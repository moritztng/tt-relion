// gather_bench [particles] [half]: the projection gathers of the weighted sums. After the sums skip dead samples a
// particle projects only its 1-3 significant orientations, and in a chip arm the host projects for nothing else, so
// each slice meets the reference volume cold: 8 voxels per pixel on 4 cache lines of a volume far larger than the
// caches. PACKED stores, per cell, the 8 corner values a pixel interpolates between in one 64-byte line (8 times the
// memory, written once per iteration), so a pixel costs one line instead of four. Same values in the same
// arithmetic, so the projections are equal to the bit; the bench checks that.
// Geometry of the dev cell at a global iteration: image radius 78, padding 2, volume 158 x 315 x 315 complex
// floats = 125 MB (packed 1 GB). One thread, thread CPU ms per particle. HUGE=1: 2 MB pages. EVICT=0: caches not
// emptied between particles. PF=<n>: software prefetch of the lines of the pixel n ahead, in both forms.
// Measured (128 particles, cold, 2 MB pages, one thread at load1 30): direct 0.93 ms per particle, direct with PF=8
// 0.57, packed 0.73, packed with PF=16 0.26. With 4 KB pages the packed form gains nothing (0.61 vs 0.62).
// Build: g++ -O3 -march=native -ffp-contract=off -std=c++17 gather_bench.cpp -o gather_bench
// (with contraction on, the compiler fuses the two loops differently and 42 % of the projections differ in the last bit)
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <random>
#include <vector>

#include <sys/mman.h>

typedef std::complex<float> C;

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

struct Pt { float xp, yp, zp; };                      // one pixel of one slice in volume coordinates, xp >= 0

static inline C lin(C a, C b, float f) { return a + (b - a) * f; }

int main(int argc, char **argv)
{
	const int N = argc > 1 ? std::atoi(argv[1]) : 256, half = argc > 2 ? std::atoi(argv[2]) : 78;
	const bool cold = !(std::getenv("EVICT") && std::getenv("EVICT")[0] == '0');
	const long X = 2 * half + 2, Y = 4 * half + 3, Z = 4 * half + 3, initY = -(2 * half + 1), initZ = -(2 * half + 1);
	std::mt19937 rng(1);
	std::normal_distribution<float> g;
	std::uniform_real_distribution<float> u;

	C *vol = (C *)block(X * Y * Z * sizeof(C));
	for (long i = 0; i < X * Y * Z; i++) vol[i] = C(g(rng), g(rng));

	std::vector<std::vector<Pt>> part(N);
	size_t total = 0;
	for (auto &pp : part)
	{
		const float r = u(rng);
		const int no = r < 0.5f ? 1 : r < 0.75f ? 2 : 3;
		for (int o = 0; o < no; o++)
		{
			float a[3] = {g(rng), g(rng), g(rng)}, b[3] = {g(rng), g(rng), g(rng)};
			const float na = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
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
					Pt p{(a[0] * x + b[0] * y) * 2, (a[1] * x + b[1] * y) * 2, (a[2] * x + b[2] * y) * 2};
					if (p.xp < 0) { p.xp = -p.xp; p.yp = -p.yp; p.zp = -p.zp; }
					pp.push_back(p);
				}
			}
		}
		total += pp.size();
	}
	std::printf("%d particles, %.0f pixels per particle, volume %.0f MB, packed %.0f MB, %s, %s pages\n", N, (double)total / N,
	            (double)X * Y * Z * 8 / 1048576, (double)X * Y * Z * 64 / 1048576, cold ? "cold" : "caches not emptied", std::getenv("HUGE") ? "2 MB" : "4 KB");

	std::vector<C> outa(total), outb(total);
	const size_t PF = std::getenv("PF") ? std::atoi(std::getenv("PF")) : 0;   // prefetch the lines of the pixel PF ahead

	// ---- direct: RELION's trilinear read, 2 voxels on each of 4 rows
	double direct = 0;
	size_t n = 0;
	for (const auto &pp : part)
	{
		if (cold) evict();
		const double t0 = cpu_ms();
		for (size_t i = 0; i < pp.size(); i++)
		{
			const Pt &p = pp[i];
			if (PF && i + PF < pp.size())
			{
				const Pt &f = pp[i + PF];
				const C *q = &vol[(((long)std::floor(f.zp) - initZ) * Y + (long)std::floor(f.yp) - initY) * X + (long)std::floor(f.xp)];
				__builtin_prefetch(q); __builtin_prefetch(q + X); __builtin_prefetch(q + X * Y); __builtin_prefetch(q + X * Y + X);
			}
			const int x0 = std::floor(p.xp), y0 = (int)std::floor(p.yp), z0 = (int)std::floor(p.zp);
			const float fx = p.xp - x0, fy = p.yp - y0, fz = p.zp - z0;
			const C *q = &vol[((z0 - initZ) * Y + y0 - initY) * X + x0];
			const C dx00 = lin(q[0], q[1], fx), dx10 = lin(q[X], q[X + 1], fx);
			const C dx01 = lin(q[X * Y], q[X * Y + 1], fx), dx11 = lin(q[X * Y + X], q[X * Y + X + 1], fx);
			outa[n++] = lin(lin(dx00, dx10, fy), lin(dx01, dx11, fy), fz);
		}
		direct += cpu_ms() - t0;
	}

	// ---- packed: the 8 corners of a cell in one line
	double t0 = cpu_ms();
	C *cell = (C *)block(X * Y * Z * 8 * sizeof(C));
	for (long z = 0; z + 1 < Z; z++)
		for (long y = 0; y + 1 < Y; y++)
		{
			const C *q = &vol[(z * Y + y) * X];
			C *c = &cell[(z * Y + y) * X * 8];
			for (long x = 0; x + 1 < X; x++, c += 8)
			{
				c[0] = q[x]; c[1] = q[x + 1]; c[2] = q[x + X]; c[3] = q[x + X + 1];
				c[4] = q[x + X * Y]; c[5] = q[x + X * Y + 1]; c[6] = q[x + X * Y + X]; c[7] = q[x + X * Y + X + 1];
			}
		}
	const double build = cpu_ms() - t0;
	double packed = 0;
	n = 0;
	for (const auto &pp : part)
	{
		if (cold) evict();
		t0 = cpu_ms();
		for (size_t i = 0; i < pp.size(); i++)
		{
			const Pt &p = pp[i];
			if (PF && i + PF < pp.size())
			{
				const Pt &f = pp[i + PF];
				__builtin_prefetch(&cell[((((long)std::floor(f.zp) - initZ) * Y + (long)std::floor(f.yp) - initY) * X + (long)std::floor(f.xp)) * 8]);
			}
			const int x0 = std::floor(p.xp), y0 = (int)std::floor(p.yp), z0 = (int)std::floor(p.zp);
			const float fx = p.xp - x0, fy = p.yp - y0, fz = p.zp - z0;
			const C *c = &cell[(((z0 - initZ) * Y + y0 - initY) * X + x0) * 8];
			const C dx00 = lin(c[0], c[1], fx), dx10 = lin(c[2], c[3], fx);
			const C dx01 = lin(c[4], c[5], fx), dx11 = lin(c[6], c[7], fx);
			outb[n++] = lin(lin(dx00, dx10, fy), lin(dx01, dx11, fy), fz);
		}
		packed += cpu_ms() - t0;
	}

	const bool same = std::memcmp(outa.data(), outb.data(), total * sizeof(C)) == 0;
	double dmax = 0, amax = 0;
	size_t nd = 0;
	for (size_t i = 0; i < total; i++)
	{
		dmax = std::max(dmax, (double)std::abs(outa[i] - outb[i])); amax = std::max(amax, (double)std::abs(outa[i]));
		nd += outa[i] != outb[i];
	}
	if (!same) std::printf("%zu of %zu projections differ, max |diff| / max |value| %.1e\n", nd, total, dmax / amax);
	std::printf("direct %.3f ms per particle | packed %.3f | ratio %.2f | packing the volume %.0f ms once | projections %s\n",
	            direct / N, packed / N, direct / packed, build, same ? "equal to the bit" : "DIFFER");
	return same ? 0 : 1;
}
