/* tt_profile.h -- by-stage wall-clock accounting for RELION's accelerated E-step.
 *
 * Why this file exists. RELION's ALTCPU path has no observable stage breakdown, for two independent
 * reasons, and both had to be found by reading the source:
 *
 *   1. `src/ml_optimiser.h` registers TIMING_ESP_DIFF1 / TIMING_ESP_DIFF2 / TIMING_ESP_WSUM, and
 *      `src/acc/acc_ml_optimiser_impl.h` does tic/toc them -- but only when
 *      `op.part_id == baseMLO->exp_my_first_part_id` (or `thread_id == 0`), which never fires on this
 *      code path, so `Timer::printTimes` skips them: it only prints tags with `counts[i] > 0`.
 *   2. The CTIC/CTOC macros that bracket every interesting region are defined empty at the top of
 *      `src/acc/cpu/cpu_benchmark_utils.h`, and the block that would give them a body is inside a
 *      `/* ... *\/` comment.
 *
 * So a -DTIMING ALTCPU build reports `expectationSomeParticles` as one opaque number (156.746 s of a
 * 161.2 s iteration) and nothing inside it. Every stage share in this lineage was therefore derived
 * from sampling counts rather than measured.
 *
 * What this does. Gives CTIC/CTOC a thread-safe body, so every region RELION already brackets is
 * accumulated. Regions are keyed on the `const char*` of the label, which is always a string literal
 * at every call site, so the pointer is stable and unique per site and there is no hashing per call.
 * Per-thread open regions live on a thread_local stack, so nesting works and a missing CTOC truncates
 * one thread's stack instead of corrupting the totals.
 *
 * The accumulated total is a sum over threads, so it exceeds the elapsed wall by roughly the thread
 * count. That is intended: the question is which stage owns the work, and a share of the CPU-time sum
 * answers it without needing the regions to be serialised. `oneParticle` is the denominator.
 *
 * Set TT_RELION_PROFILE=1 to print the table at exit. Accumulation is always on: it is a handful of
 * relaxed atomic adds per particle per region against a stage that costs tens of milliseconds.
 *
 * Two columns. `wall-seconds` is steady_clock time inside the region, summed over threads; it includes
 * time a thread sleeps there (a thread waiting in the bridge for the chip counts in full). `cpu-seconds`
 * is the thread's own CPU time (CLOCK_THREAD_CPUTIME_ID) and is what a rank actually burns; it is
 * sampled only when TT_RELION_PROFILE is set, because it is a system call per tic and per toc.
 *
 * TT_RELION_PROFILE_FILE=<prefix> also rewrites the table to <prefix>.<pid> every 256 particles, so an
 * arm that is stopped by a signal after its iteration (auto-refine ignores --iter) still leaves one.
 */
#ifndef TT_PROFILE_H_
#define TT_PROFILE_H_

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <time.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace TTProf
{

struct Rec
{
	const char *label;
	std::atomic<unsigned long long> ns;
	std::atomic<unsigned long long> cpu_ns;
	std::atomic<unsigned long long> n;
	Rec(const char *l) : label(l), ns(0), cpu_ns(0), n(0) {}
};

// A vector of pointers, never of objects: the vector may reallocate and a Rec holds atomics.
// Both statics are deliberately leaked. The report runs from an atexit handler, and atexit handlers
// interleave with static destructors in one LIFO list: a `static std::vector` here is constructed
// AFTER the handler is registered and therefore destroyed BEFORE it runs, so the report reads freed
// memory and the process exits 139 after printing a correct table. Measured that way once.
inline std::vector<Rec *> &table()
{
	static std::vector<Rec *> *t = new std::vector<Rec *>();
	return *t;
}

inline std::mutex &table_mutex()
{
	static std::mutex *m = new std::mutex();
	return *m;
}

inline void report();
inline void write_table(std::FILE *f);

inline bool enabled()
{
	static const bool on = [] {
		const char *e = std::getenv("TT_RELION_PROFILE");
		return e && e[0] != '0' && e[0] != '\0';
	}();
	return on;
}

inline unsigned long long thread_cpu_ns()
{
	if (!enabled())
		return 0;
	struct timespec ts;
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
	return (unsigned long long)ts.tv_sec * 1000000000ull + (unsigned long long)ts.tv_nsec;
}

inline Rec *rec_for(const char *label)
{
	// One registration per call site, then pointer-keyed lookup. The table has a few dozen entries,
	// so the linear scan is cheaper than a hash and needs no allocation on the hot path.
	std::lock_guard<std::mutex> g(table_mutex());
	static bool registered_atexit = false;
	if (!registered_atexit)
	{
		registered_atexit = true;
		std::atexit(&report);
	}
	for (size_t i = 0; i < table().size(); i++)
		if (table()[i]->label == label)
			return table()[i];
	table().push_back(new Rec(label));
	return table().back();
}

typedef std::chrono::steady_clock clk;

struct Open
{
	Rec *rec;
	clk::time_point wall;
	unsigned long long cpu;
};

inline std::vector<Open> &open_stack()
{
	static thread_local std::vector<Open> s;
	return s;
}

// Pointer-keyed cache so only the first tic per site per thread takes the mutex.
inline Rec *cached_rec(const char *label)
{
	static thread_local std::vector<std::pair<const char *, Rec *> > cache;
	for (size_t i = 0; i < cache.size(); i++)
		if (cache[i].first == label)
			return cache[i].second;
	Rec *r = rec_for(label);
	cache.push_back(std::make_pair(label, r));
	return r;
}

inline void tic(const char *label)
{
	Open o = {cached_rec(label), clk::now(), thread_cpu_ns()};
	open_stack().push_back(o);
}

// Rewrite <TT_RELION_PROFILE_FILE>.<pid> with the table so far.
inline void dump_file()
{
	const char *prefix = std::getenv("TT_RELION_PROFILE_FILE");
	if (!prefix || !prefix[0])
		return;
	const std::string path = std::string(prefix) + "." + std::to_string((long)getpid());
	const std::string tmp = path + ".tmp";
	std::lock_guard<std::mutex> g(table_mutex());
	if (std::FILE *f = std::fopen(tmp.c_str(), "w"))
	{
		write_table(f);
		std::fclose(f);
		std::rename(tmp.c_str(), path.c_str());
	}
}

inline void toc(const char *label)
{
	std::vector<Open> &s = open_stack();
	const clk::time_point now = clk::now();
	const unsigned long long cpu = thread_cpu_ns();
	// Scan from the top for the matching open region rather than assuming the top is it: RELION has
	// early returns between some CTIC/CTOC pairs, and a mismatched pop would then charge one stage's
	// time to another for the rest of the run.
	for (size_t k = s.size(); k-- > 0;)
	{
		if (s[k].rec->label == label)
		{
			Rec *r = s[k].rec;
			const unsigned long long ns =
			    (unsigned long long)std::chrono::duration_cast<std::chrono::nanoseconds>(now - s[k].wall).count();
			r->ns.fetch_add(ns, std::memory_order_relaxed);
			r->cpu_ns.fetch_add(cpu - s[k].cpu, std::memory_order_relaxed);
			const unsigned long long n = r->n.fetch_add(1, std::memory_order_relaxed) + 1;
			s.resize(k);
			if (enabled() && n % 256 == 0 && std::strcmp(label, "oneParticle") == 0)
				dump_file();
			return;
		}
	}
	// No matching tic: a CTOC without its CTIC. Drop it rather than guess.
}

// Caller holds table_mutex().
inline void write_table(std::FILE *f)
{
	unsigned long long total = 0;
	for (size_t i = 0; i < table().size(); i++)
		if (table()[i]->label && std::strcmp(table()[i]->label, "oneParticle") == 0)
			total = table()[i]->cpu_ns.load();
	std::fprintf(f, "\nTTPROF rank_pid=%d stages=%zu denominator=oneParticle\n", (int)getpid(), table().size());
	std::fprintf(f, "TTPROF %-42s %14s %14s %12s %9s\n", "region", "wall-seconds", "cpu-seconds", "calls", "%cpu");
	for (size_t i = 0; i < table().size(); i++)
	{
		const unsigned long long ns = table()[i]->ns.load();
		const unsigned long long cpu = table()[i]->cpu_ns.load();
		const unsigned long long n = table()[i]->n.load();
		std::fprintf(f, "TTPROF %-42s %14.4f %14.4f %12llu %8.2f%%\n", table()[i]->label, ns / 1e9, cpu / 1e9, n,
		             total ? 100.0 * (double)cpu / (double)total : 0.0);
	}
	std::fflush(f);
}

inline void report()
{
	if (!enabled())
		return;
	dump_file();
	std::lock_guard<std::mutex> g(table_mutex());
	write_table(stdout);
}

} // namespace TTProf

#endif /* TT_PROFILE_H_ */
