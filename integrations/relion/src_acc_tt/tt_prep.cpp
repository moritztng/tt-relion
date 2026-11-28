#include "tt_prep.h"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <unistd.h>

namespace TTPrep
{

bool enabled()
{
	static const bool on = [] { const char *e = getenv("TT_RELION_PREP"); return !(e && e[0] == '0'); }();
	return on;
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
	return t;
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

float *work(int n)
{
	static thread_local std::vector<float> buf;
	if (buf.size() < (size_t)n * n)
		buf.resize((size_t)n * n);
	return buf.data();
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
