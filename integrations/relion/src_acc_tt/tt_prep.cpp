#include "tt_prep.h"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <unistd.h>

#include <fftw3.h>
#include <immintrin.h>
#include <tbb/spin_mutex.h>

// RELION's lock around the single-precision FFTW planner (src/acc/cpu/cpu_ml_optimiser.cpp). Every plan its
// accelerated path makes is made under it, and so are the ones here.
extern tbb::spin_mutex mkl_mutex;

namespace TTPrep
{

bool enabled()
{
	static const bool on = [] { const char *e = getenv("TT_RELION_PREP"); return !(e && e[0] == '0'); }();
	return on;
}

// The plans of forward(): BLOCK rows, the n % BLOCK rows that remain, and k columns for k up to BLOCK. A
// block of BLOCK rows or columns starts a multiple of 64 bytes into its buffer, so every call sees the
// alignment its plan was made for. With FFTW_WISDOM_ONLY a plan the wisdom lacks is estimated instead.
// Returns how many were not. Call under mkl_mutex.
static int plan(Tables &t, unsigned flags)
{
	int n = t.n, ix = n / 2 + 1, made = 0;
	const int B = Tables::BLOCK;
	float *r = fftwf_alloc_real((size_t)n * n);
	fftwf_complex *c = fftwf_alloc_complex((size_t)n * ix);
	for (int i = 0; i < 2; i++)
	{
		const int k = i ? n % B : (n >= B ? B : 0);
		t.rows[i] = k ? fftwf_plan_many_dft_r2c(1, &n, k, r, NULL, 1, n, c, NULL, 1, ix, flags) : NULL;
		if (!k)
			continue;
		if (!t.rows[i])
			t.rows[i] = fftwf_plan_many_dft_r2c(1, &n, k, r, NULL, 1, n, c, NULL, 1, ix, FFTW_ESTIMATE);
		else
			made++;
	}
	t.columns[0] = NULL;
	for (int k = 1; k <= B; k++)
	{
		t.columns[k] = fftwf_plan_many_dft(1, &n, k, c, NULL, ix, 1, c, NULL, ix, 1, FFTW_FORWARD, flags);
		if (!t.columns[k])
			t.columns[k] = fftwf_plan_many_dft(1, &n, k, c, NULL, ix, 1, c, NULL, ix, 1, FFTW_FORWARD, FFTW_ESTIMATE);
		else
			made++;
	}
	fftwf_free(r);
	fftwf_free(c);
	return made;
}

// The expressions of CpuKernels::softMaskBackgroundValue / cosineFilter and CpuKernels::powerClass,
// evaluated once per pixel instead of once per pixel per particle.
static std::unique_ptr<Tables> build(int n, float radius, float edge)
{
	std::unique_ptr<Tables> t(new Tables);
	t->n = n; t->radius = radius; t->edge = edge;
	const float radius_p = radius + edge;
	t->weight.resize((size_t)n * n);
	t->weight_sum = 0.;
	for (int i = 0; i < n * n; i++)
	{
		const int x = i % n - n / 2, y = i / n - n / 2;
		const float r = sqrtf(float(x * x + y * y));
		float w;
		if (r < radius) w = 0.f;
		else if (r > radius_p) w = 1.f;
		else w = 0.5 + 0.5 * cosf((radius_p - r) / edge * M_PI);
		t->weight[i] = w;
		t->weight_sum += w;
	}
	const int ix = n / 2 + 1;
	t->shell.resize((size_t)n * ix);
	for (int v = 0; v < n * ix; v++)
	{
		const int x = v % ix;
		int y = v / ix;
		y = y < ix ? y : y - n;
		const size_t s = (size_t)(sqrtf((float)(x * x + y * y)) + 0.5f);
		t->shell[v] = (s < (size_t)ix && !(x == 0 && y < 0)) ? s : ix;
	}
	const char *wisdom = getenv("TT_RELION_PREP_WISDOM");
	tbb::spin_mutex::scoped_lock lock(mkl_mutex);
	const bool have = wisdom && wisdom[0] && fftwf_import_wisdom_from_filename(wisdom);
	t->from_wisdom = 0;
	if (have)
		t->from_wisdom = plan(*t, FFTW_MEASURE | FFTW_WISDOM_ONLY);
	else
		plan(*t, FFTW_ESTIMATE);
	// The file is for these plans only: RELION's own keep planning as they did without it.
	if (have) fftwf_forget_wisdom();
	lock.release();
	if (wisdom && wisdom[0])
		fprintf(stderr, "TTPrep: box %d, %d of %d FFT plans from %s\n", n, t->from_wisdom, Tables::BLOCK + 2, wisdom);
	return t;
}

bool write_wisdom(int n, const char *path)
{
	Tables t;
	t.n = n;
	tbb::spin_mutex::scoped_lock lock(mkl_mutex);
	fftwf_forget_wisdom();
	plan(t, FFTW_MEASURE);
	const bool ok = fftwf_export_wisdom_to_filename(path);
	fftwf_forget_wisdom();
	return ok;
}

const Tables &tables(int n, float radius, float edge)
{
	static thread_local const Tables *last = nullptr;
	if (last && last->n == n && last->radius == radius && last->edge == edge)
		return *last;
	static std::mutex m;
	static std::vector<std::unique_ptr<Tables>> all;
	std::lock_guard<std::mutex> lock(m);
	for (auto &t : all)
		if (t->n == n && t->radius == radius && t->edge == edge)
			return *(last = t.get());
	all.push_back(build(n, radius, edge));
	return *(last = all.back().get());
}

namespace
{
struct Buffers : Work
{
	int n;
	Buffers() : Work{nullptr, nullptr, nullptr}, n(0) {}
	~Buffers() { release(); }
	void release() { fftwf_free(shifted); fftwf_free(real); fftwf_free(freq); }
};
}

Work &work(int n)
{
	static thread_local Buffers b;
	if (b.n != n)
	{
		b.release();
		b.shifted = fftwf_alloc_real((size_t)n * n);
		b.real = fftwf_alloc_real((size_t)n * n);
		b.freq = (float*) fftwf_alloc_complex((size_t)n * (n / 2 + 1));
		b.n = n;
	}
	return b;
}

void forward(const Tables &t, const float *real, float *freq, int columns)
{
	const int n = t.n, ix = n / 2 + 1, B = Tables::BLOCK;
	fftwf_complex *c = (fftwf_complex*) freq;
	int y = 0;
	for (; y + B <= n; y += B)
		fftwf_execute_dft_r2c((fftwf_plan) t.rows[0], (float*) real + (size_t)y * n, c + (size_t)y * ix);
	if (y < n)
		fftwf_execute_dft_r2c((fftwf_plan) t.rows[1], (float*) real + (size_t)y * n, c + (size_t)y * ix);
	int x = 0;
	for (; x + B <= columns; x += B)
		fftwf_execute_dft((fftwf_plan) t.columns[B], c + x, c + x);
	if (x < columns)
		fftwf_execute_dft((fftwf_plan) t.columns[columns - x], c + x, c + x);
}

void center(const float *in, int n, float *out)
{
	const int h = n / 2;
	for (int y = 0; y < n; y++)
	{
		const float *s = in + (size_t)y * n;
		float *d = out + (size_t)((y + h) % n) * n;
		memcpy(d + h, s, sizeof(float) * h);
		memcpy(d, s + h, sizeof(float) * h);
	}
}

void mask_center(const Tables &t, const float *in, float *out)
{
	const int n = t.n, h = n / 2;
	const float *w = t.weight.data();
	double sum = 0.;
	#pragma omp simd reduction(+:sum)
	for (int i = 0; i < n * n; i++)
		sum += w[i] * in[i];
	const float bg = sum / t.weight_sum;
	for (int y = 0; y < n; y++)
	{
		const float *s = in + (size_t)y * n, *wy = w + (size_t)y * n;
		float *d = out + (size_t)((y + h) % n) * n;
		for (int x = 0; x < h; x++) d[x + h] = s[x] * (1 - wy[x]) + bg * wy[x];
		for (int x = h; x < n; x++) d[x - h] = s[x] * (1 - wy[x]) + bg * wy[x];
	}
}

double power(const Tables &t, const float *f, float scale, int first, double *spectrum)
{
	const int ix = t.n / 2 + 1;
	const uint16_t *shell = t.shell.data();
	static thread_local std::vector<double> acc;
	acc.assign(ix + 1, 0.);
	for (int v = 0; v < t.n * ix; v++)
	{
		const float re = f[2 * v] * scale, im = f[2 * v + 1] * scale;
		acc[shell[v]] += re * re + im * im;
	}
	double high = 0.;
	for (int s = 0; s < ix; s++)
	{
		spectrum[s] = acc[s];
		if (s >= first) high += acc[s];
	}
	return high;
}

// glibc's vector math (libmvec, linked through -lm): sin and exp of four doubles.
extern "C" { __m256d _ZGVdN4v_sin(__m256d); __m256d _ZGVdN4v_exp(__m256d); }

__attribute__((target("avx2")))
static void sin_exp4(const double *phase, const double *arg, int n, double *s, double *e)
{
	for (int j = 0; j < n; j += 4)
		_mm256_storeu_pd(s + j, _ZGVdN4v_sin(_mm256_loadu_pd(phase + j)));
	if (e)
		for (int j = 0; j < n; j += 4)
			_mm256_storeu_pd(e + j, _ZGVdN4v_exp(_mm256_loadu_pd(arg + j)));
}

void ctf(const Ctf &c, double xs, double ys, bool do_abs, bool only_flip_phases, bool intact_first_peak,
         bool damping, int ydim, int xdim, double *out)
{
	static const bool avx2 = __builtin_cpu_supports("avx2");
	static thread_local std::vector<double> buf;
	const int m = (xdim + 3) & ~3;
	buf.assign((size_t)4 * m, 0.);
	double *phase = buf.data(), *arg = phase + m, *s = arg + m, *e = s + m;
	const bool decay = damping && c.K4 != 0.;   // exp(K4 * u2) is exactly 1 when K4 is 0
	for (int i = 0; i < ydim; i++)
	{
		// FOR_ALL_ELEMENTS_IN_FFTW_TRANSFORM2D, and the body of CTF::getCTF up to its sin
		const double Y = (double)(i < xdim ? i : i - ydim) / ys;
		for (int j = 0; j < xdim; j++)
		{
			const double X = (double)j / xs;
			const double u2 = X * X + Y * Y, u4 = u2 * u2;
			phase[j] = c.K1 * (c.Axx * X * X + 2.0 * c.Axy * X * Y + c.Ayy * Y * Y) + c.K2 * u4 - c.K5 - c.K3 + 0.0;
			arg[j] = c.K4 * u2;
		}
		if (avx2)
			sin_exp4(phase, arg, m, s, decay ? e : nullptr);
		else
			for (int j = 0; j < xdim; j++)
			{
				s[j] = sin(phase[j]);
				if (decay) e[j] = exp(arg[j]);
			}
		// the rest of CTF::getCTF
		double *o = out + (size_t)i * xdim;
		for (int j = 0; j < xdim; j++)
		{
			double v = (intact_first_peak && fabs(phase[j]) < M_PI / 2.) ? 1. : -s[j];
			if (decay) v *= e[j];
			if (do_abs) v = fabs(v);
			else if (only_flip_phases) v = (v < 0.) ? -1. : 1.;
			v *= c.scale;
			if (fabs(v) < 1e-8) v = (v >= 0 ? 1 : -1) * 1e-8;
			o[j] = v;
		}
	}
}

Record::Record(long part) : f(nullptr)
{
	static const char *dir = getenv("TT_RELION_RECORD_PREP");
	static std::atomic<int> left([] { const char *e = getenv("TT_RELION_RECORD_PREP_N"); return e ? atoi(e) : 8; }());
	if (dir && left.fetch_sub(1) > 0)
		f = fopen((std::string(dir) + "/prep." + std::to_string(getpid()) + "." + std::to_string(part) + ".bin").c_str(), "wb");
}

void Record::put(const char *name, const char *dtype, const void *data, size_t item, long ydim, long xdim)
{
	fprintf(f, "%s %s %ld %ld\n", name, dtype, ydim, xdim);
	fwrite(data, item, (size_t)ydim * xdim, f);
}

}
