// bench_prep.cpp: single-thread CPU cost of each TTPrep piece, with the transform done two ways: RELION's
// one 2D plan (what its accelerated path executes) and TTPrep::forward (rows then columns, in blocks, the
// unmasked transform's column pass stopped at the window). Also checks forward against the 2D plan.
// g++ -O3 -fopenmp -I integrations/relion/src_acc_tt perf/rlx/prep/bench_prep.cpp \
//     integrations/relion/src_acc_tt/tt_prep.cpp -lfftw3f -ltbb -o bench_prep
// usage: bench_prep [box 360] [current size 156] [particles 2000]     (TT_RELION_PREP_WISDOM=<file> to time those plans)
//        bench_prep wisdom <box> <file>                              (measure the plans once, write the file)
// Times are thread CPU seconds (CLOCK_THREAD_CPUTIME_ID), the clock the E-step profiler uses, best of 5 rounds.
#include "tt_prep.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fftw3.h>
#include <random>
#include <tbb/spin_mutex.h>

tbb::spin_mutex mkl_mutex;   // RELION defines it; tt_prep.cpp plans under it

struct Cplx { double real, imag; };
static double now() { timespec t; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

int main(int argc, char **argv)
{
	if (argc == 4 && !strcmp(argv[1], "wisdom"))
		return TTPrep::write_wisdom(atoi(argv[2]), argv[3]) ? 0 : 1;
	const int n = argc > 1 ? atoi(argv[1]) : 360, cur = argc > 2 ? atoi(argv[2]) : 156, reps = argc > 3 ? atoi(argv[3]) : 2000;
	const int ix = n / 2 + 1, ox = cur / 2 + 1;
	std::vector<double> img((size_t)n * n);
	std::mt19937 rng(1);
	std::normal_distribution<double> g;
	for (auto &v : img) v = g(rng);
	std::vector<Cplx> Fa((size_t)cur * ox), Fb((size_t)cur * ox), Ga((size_t)cur * ox), Gb((size_t)cur * ox);
	std::vector<double> spec(ix), spec2(ix);
	const TTPrep::Tables &t = TTPrep::tables(n, 134.3f, 5.f);
	TTPrep::Work &w = TTPrep::work(n);
	const float scale = 1.f / ((float)n * n);
	float *reals = fftwf_alloc_real((size_t)n * n);
	fftwf_complex *four = fftwf_alloc_complex((size_t)n * ix);
	fftwf_plan plan = fftwf_plan_dft_r2c_2d(n, n, reals, four, FFTW_ESTIMATE);

	// Same input through both: the two windows and the spectrum must agree to float rounding.
	TTPrep::shift(img.data(), n, 0.97f, 3, -2, w.shifted);
	TTPrep::center(w.shifted, n, reals);
	fftwf_execute_dft_r2c(plan, reals, four);
	TTPrep::window((const float *)four, n, cur, scale, Fa.data());
	TTPrep::mask_center(t, w.shifted, reals);
	fftwf_execute_dft_r2c(plan, reals, four);
	TTPrep::window((const float *)four, n, cur, scale, Fb.data());
	const double xa = TTPrep::power(t, (const float *)four, scale, ox, spec.data());
	TTPrep::center(w.shifted, n, w.real);
	TTPrep::forward(t, w.real, w.freq, ox);
	TTPrep::window(w.freq, n, cur, scale, Ga.data());
	TTPrep::mask_center(t, w.shifted, w.real);
	TTPrep::forward(t, w.real, w.freq, ix);
	TTPrep::window(w.freq, n, cur, scale, Gb.data());
	const double xb = TTPrep::power(t, w.freq, scale, ox, spec2.data());
	auto rel = [&](const std::vector<Cplx> &a, const std::vector<Cplx> &b) {
		double d = 0, s = 0;
		for (size_t i = 0; i < a.size(); i++)
		{
			d += (a[i].real - b[i].real) * (a[i].real - b[i].real) + (a[i].imag - b[i].imag) * (a[i].imag - b[i].imag);
			s += a[i].real * a[i].real + a[i].imag * a[i].imag;
		}
		return sqrt(d / s);
	};
	double ds = 0, ss = 0;
	for (int i = 0; i < ix; i++) { ds += (spec[i] - spec2[i]) * (spec[i] - spec2[i]); ss += spec[i] * spec[i]; }
	printf("forward vs 2D plan, relative L2: unmasked window %.2e, masked window %.2e, spectrum %.2e, highres Xi2 %.2e\n",
	       rel(Fa, Ga), rel(Fb, Gb), sqrt(ds / ss), fabs(xa - xb) / xa);

	for (int mode = 0; mode < 2; mode++)
	{
		double best[7], sink = 0;
		std::fill(best, best + 7, 1e9);
		for (int round = 0; round < 5; round++)
		{
			double part[7] = {0};
			for (int r = -20; r < reps; r++)
			{
				if (r == 0) for (double &v : part) v = 0;
				float *in = mode ? w.real : reals;
				const float *out = mode ? w.freq : (const float *)four;
				double a = now(), b;
				TTPrep::shift(img.data(), n, 0.97f, r % 5 - 2, r % 3 - 1, w.shifted);   b = now(); part[0] += b - a; a = b;
				TTPrep::center(w.shifted, n, in);                                        b = now(); part[1] += b - a; a = b;
				if (mode) TTPrep::forward(t, in, w.freq, ox); else fftwf_execute_dft_r2c(plan, in, four);
				                                                                         b = now(); part[2] += b - a; a = b;
				TTPrep::window(out, n, cur, scale, Fa.data());                           b = now(); part[4] += b - a; a = b;
				TTPrep::mask_center(t, w.shifted, in);                                   b = now(); part[5] += b - a; a = b;
				if (mode) TTPrep::forward(t, in, w.freq, ix); else fftwf_execute_dft_r2c(plan, in, four);
				                                                                         b = now(); part[3] += b - a; a = b;
				TTPrep::window(out, n, cur, scale, Fb.data());                           b = now(); part[4] += b - a; a = b;
				sink += TTPrep::power(t, out, scale, ox, spec.data());                   b = now(); part[6] += b - a;
				sink += Fa[7].real + Fb[9].imag;
			}
			for (int i = 0; i < 7; i++) best[i] = std::min(best[i], part[i]);
		}
		double total = 0;
		for (double v : best) total += v;
		printf("%-22s box %d cur %d: total %.3f ms/particle | shift %.3f center %.3f FFT unmasked %.3f FFT masked %.3f 2xwindow %.3f mask %.3f power %.3f (%g)\n",
		       mode ? (t.from_wisdom ? "blocks, wisdom" : "blocks, estimate") : "one 2D plan (RELION)", n, cur, 1e3 * total / reps, 1e3 * best[0] / reps, 1e3 * best[1] / reps,
		       1e3 * best[2] / reps, 1e3 * best[3] / reps, 1e3 * best[4] / reps, 1e3 * best[5] / reps, 1e3 * best[6] / reps, sink);
	}
	return 0;
}
