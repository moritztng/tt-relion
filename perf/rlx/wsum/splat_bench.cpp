// splat_bench [particles] [half]: the back-projection splat of a stream of particles into one half volume, written
// directly (as RELION does, each pixel into 8 voxels of a volume far larger than the caches) and DEFERRED: every
// pixel is appended to a log, and once per batch of particles the log is partitioned by the volume tile it lands in
// and applied tile by tile, so the adds hit a tile that sits in L2 and the volume is streamed once per batch.
// Geometry of the dev cell at a global iteration: image radius 78 (size 156), padding 2, volume 158 x 315 x 315
// voxels of (re, im, weight) = 188 MB; 1-3 significant orientations per particle (mean 1.75), random orientations.
// The values are random: this measures memory traffic, the arithmetic is the same in both.
// One thread, thread CPU ms per particle. HUGE=1: 2 MB pages. EVICT=0: do not empty the caches between particles
// (RELION meets every particle cold, seconds of other work pass between two particles of a thread).
// Build: g++ -O3 -march=native -std=c++17 splat_bench.cpp -o splat_bench
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <random>
#include <vector>

#include <sys/mman.h>

static double cpu_ms()
{
	timespec t;
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
	return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

static void evict()
{
	static std::vector<char> junk(512u << 20);
	for (size_t i = 0; i < junk.size(); i += 64) junk[i]++;
}

static void *block(size_t bytes)
{
	const size_t two = 2u << 20;
	bytes = (bytes + two - 1) / two * two;
	void *p = std::aligned_alloc(two, bytes);
	if (std::getenv("HUGE")) madvise(p, bytes, MADV_HUGEPAGE);
	std::memset(p, 0, bytes);
	return p;
}

struct Pix { float xp, yp, zp, re, im, w; };          // one pixel of one slice, already folded to xp >= 0

struct Vol
{
	long X, Y, Z, initY, initZ;
	float *v;
	Vol(long r) : X(2 * r + 2), Y(4 * r + 3), Z(4 * r + 3), initY(-(2 * r + 1)), initZ(-(2 * r + 1)), v((float *)block(3 * X * Y * Z * 4)) {}
	~Vol() { std::free(v); }
	inline void add2(long i, float g0, float g1, const Pix &p)
	{
		float *q = &v[3 * i];
		q[0] += g0 * p.re; q[1] += g0 * p.im; q[2] += g0 * p.w;
		q[3] += g1 * p.re; q[4] += g1 * p.im; q[5] += g1 * p.w;
	}
	inline void splat(const Pix &p)
	{
		const int x0 = std::floor(p.xp), yy = std::floor(p.yp), zz = std::floor(p.zp);
		const float fx = p.xp - x0, fy = p.yp - yy, fz = p.zp - zz, mfx = 1 - fx, mfy = 1 - fy, mfz = 1 - fz;
		const long i = ((zz - initZ) * Y + yy - initY) * X + x0;
		add2(i, mfz * mfy * mfx, mfz * mfy * fx, p);
		add2(i + X, mfz * fy * mfx, mfz * fy * fx, p);
		add2(i + X * Y, fz * mfy * mfx, fz * mfy * fx, p);
		add2(i + X * Y + X, fz * fy * mfx, fz * fy * fx, p);
	}
	// the tile of a pixel: its lower z plane and a band of BAND rows of y
	static constexpr int BAND = 32;
	inline uint32_t tile(const Pix &p) const
	{
		const long z0 = (long)std::floor(p.zp) - initZ, y0 = (long)std::floor(p.yp) - initY;
		return z0 * ((Y + BAND - 1) / BAND) + y0 / BAND;
	}
	uint32_t tiles() const { return Z * ((Y + BAND - 1) / BAND); }
};

int main(int argc, char **argv)
{
	const int N = argc > 1 ? std::atoi(argv[1]) : 512, half = argc > 2 ? std::atoi(argv[2]) : 78;
	const bool cold = !(std::getenv("EVICT") && std::getenv("EVICT")[0] == '0');
	std::mt19937 rng(1);
	std::normal_distribution<float> g;
	std::uniform_real_distribution<float> u;
	// the slices of every particle: 1-3 orientations (1 with p 0.5, 2 with 0.25, 3 with 0.25: mean 1.75)
	std::vector<std::vector<Pix>> part(N);
	size_t total = 0;
	for (auto &pp : part)
	{
		const float r = u(rng);
		const int no = r < 0.5f ? 1 : r < 0.75f ? 2 : 3;
		for (int o = 0; o < no; o++)
		{
			float a[3] = {g(rng), g(rng), g(rng)}, b[3] = {g(rng), g(rng), g(rng)};
			float na = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
			for (float &c : a) c /= na;
			const float d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
			for (int k = 0; k < 3; k++) b[k] -= d * a[k];
			const float nb = std::sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
			for (float &c : b) c /= nb;
			for (int y = -half + 1; y < half; y++)
			{
				const int xmax = std::sqrt((float)(half * half - y * y));
				for (int x = 0; x < xmax; x++)
				{
					Pix p{(a[0] * x + b[0] * y) * 2, (a[1] * x + b[1] * y) * 2, (a[2] * x + b[2] * y) * 2, g(rng), g(rng), u(rng)};
					if (p.xp < 0) { p.xp = -p.xp; p.yp = -p.yp; p.zp = -p.zp; p.im = -p.im; }
					pp.push_back(p);
				}
			}
		}
		total += pp.size();
	}
	std::printf("%d particles, %.0f pixels per particle, volume %.0f MB, %s, %s pages\n", N, (double)total / N,
	            3.0 * (2 * half + 2) * (4 * half + 3) * (4 * half + 3) * 4 / 1048576, cold ? "cold" : "caches not emptied", std::getenv("HUGE") ? "2 MB" : "4 KB");

	// ---- direct
	Vol va(half);
	double direct = 0;
	for (const auto &pp : part)
	{
		if (cold) evict();
		const double t0 = cpu_ms();
		for (const Pix &p : pp) va.splat(p);
		direct += cpu_ms() - t0;
	}

	// ---- deferred: append, then partition by tile (counting sort, stable) and apply
	Vol vb(half);
	Pix *log = (Pix *)block(total * sizeof(Pix)), *sorted = (Pix *)block(total * sizeof(Pix));
	double append = 0, partition = 0, apply = 0;
	size_t n = 0;
	for (const auto &pp : part)
	{
		if (cold) evict();
		const double t0 = cpu_ms();
		for (const Pix &p : pp) log[n++] = p;      // in RELION the kernel writes the pixel here instead of splatting it
		append += cpu_ms() - t0;
	}
	if (cold) evict();
	double t0 = cpu_ms();
	std::vector<uint32_t> start(vb.tiles() + 1, 0);
	for (size_t i = 0; i < n; i++) start[vb.tile(log[i]) + 1]++;
	for (size_t t = 0; t < vb.tiles(); t++) start[t + 1] += start[t];
	{
		std::vector<uint32_t> at(start.begin(), start.end() - 1);
		for (size_t i = 0; i < n; i++) sorted[at[vb.tile(log[i])]++] = log[i];
	}
	partition = cpu_ms() - t0;
	if (cold) evict();
	t0 = cpu_ms();
	for (size_t i = 0; i < n; i++) vb.splat(sorted[i]);
	apply = cpu_ms() - t0;

	// the same sums, in another order per voxel
	double sa = 0, sb = 0, dmax = 0, amax = 0;
	const size_t nv = 3 * va.X * va.Y * va.Z;
	for (size_t i = 0; i < nv; i++)
	{
		sa += va.v[i]; sb += vb.v[i];
		dmax = std::max(dmax, (double)std::fabs(va.v[i] - vb.v[i])); amax = std::max(amax, (double)std::fabs(va.v[i]));
	}
	std::printf("direct %.3f ms per particle | deferred %.3f = append %.3f + partition %.3f + apply %.3f | ratio %.2f | max |diff| / max |value| %.1e\n",
	            direct / N, (append + partition + apply) / N, append / N, partition / N, apply / N, direct / (append + partition + apply), dmax / amax);
	(void)sa; (void)sb;
	return 0;
}
