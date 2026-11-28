// kernel_bench <sums.*.bin> ...: where storeWeightedSums' CPU goes for recorded particles, and what a one-pass
// kernel over the significant samples alone costs. One thread, CPU ms per particle.
//
//   relion  the loop order of RELION's two CPU kernels: wavg projects EVERY orientation and walks the pairs,
//           backproject builds the phase tables again, walks every orientation's rows and splats under row locks
//   sparse  one pass per orientation that has a significant pair: phase tables for the translations in use only,
//           Z = sum_t w' S_t once per row and shared by the three per-pixel sums and the back-projected image
//   splat   the splat alone (the part that stays on the host when the chip returns per-orientation sums), into
//           RELION's three arrays with and without row locks, and into one interleaved (re, im, weight) array
//
// Each variant's sums are compared with what RELION recorded (relative L2), so a variant that does other work
// shows. Build: g++ -O3 -march=native -std=c++17 kernel_bench.cpp -o kernel_bench
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <map>
#include <string>
#include <vector>

#include <sys/mman.h>

typedef std::map<std::string, std::vector<char>> Rec;

static Rec load(const std::string &path)
{
	Rec r;
	std::FILE *f = std::fopen(path.c_str(), "rb");
	char magic[8];
	if (!f || std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "TTSUMS1\n", 8)) { std::fprintf(stderr, "%s: not a sums record\n", path.c_str()); std::exit(1); }
	uint16_t n;
	while (std::fread(&n, 2, 1, f) == 1)
	{
		std::string name(n, ' ');
		char type;
		uint64_t count;
		if (std::fread(&name[0], 1, n, f) != n || std::fread(&type, 1, 1, f) != 1 || std::fread(&count, 8, 1, f) != 1) std::exit(1);
		std::vector<char> &v = r[name];
		v.resize(count * ((type == 'f' || type == 'i') ? 4 : 8));
		if (std::fread(v.data(), 1, v.size(), f) != v.size()) std::exit(1);
	}
	std::fclose(f);
	return r;
}
static long I(const Rec &r, const char *k) { return (long)*(const int64_t *)r.at(k).data(); }
static double D(const Rec &r, const char *k) { return *(const double *)r.at(k).data(); }
static const float *F(const Rec &r, const char *k) { return (const float *)r.at(k).data(); }

static double cpu_ms()
{
	timespec t;
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
	return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

// RELION meets every particle's slices cold (seconds of other work pass between two particles of a thread), so
// every timed phase here starts from an evicted cache too.
static void evict()
{
	static std::vector<char> junk(512u << 20);
	for (size_t i = 0; i < junk.size(); i += 64) junk[i]++;
}

// Zeroed floats on a 2 MB boundary; HUGE=1 asks for 2 MB pages (the volumes are 65-200 MB and every pixel lands on
// another 4 KB page). PREFETCH=1: the splat-alone loops fetch a row's voxels before adding into them.
static const bool HUGE = std::getenv("HUGE"), PREFETCH = std::getenv("PREFETCH");
static float *floats(size_t n)
{
	const size_t two = 2u << 20, bytes = (n * 4 + two - 1) / two * two;
	void *p = std::aligned_alloc(two, bytes);
	if (HUGE) madvise(p, bytes, MADV_HUGEPAGE);
	std::memset(p, 0, bytes);
	return (float *)p;
}

struct Model { const float *v; long X, Y, Z, initY, initZ, maxR2; float pf; };

// AccProjectorKernel::project3Dmodel for 2D data, without textures
static inline void project(const Model &m, int x, int y, const float *e, float &re, float &im)
{
	float xp = (e[0] * x + e[1] * y) * m.pf, yp = (e[3] * x + e[4] * y) * m.pf, zp = (e[6] * x + e[7] * y) * m.pf;
	if ((int)(xp * xp + yp * yp + zp * zp) > m.maxR2) { re = im = 0; return; }
	const bool inv = xp < 0;
	if (inv) { xp = -xp; yp = -yp; zp = -zp; }
	const int x0 = std::floor(xp), y0 = std::floor(yp), z0 = std::floor(zp);
	const float fx = xp - x0, fy = yp - y0, fz = zp - z0;
	const float *c = m.v + 2 * ((z0 - m.initZ) * m.X * m.Y + (y0 - m.initY) * m.X + x0);
	const long dy = 2 * m.X, dz = 2 * m.X * m.Y;
	float o[2];
	for (int k = 0; k < 2; k++)
	{
		const float a00 = c[k] + (c[k + 2] - c[k]) * fx, a10 = c[dy + k] + (c[dy + k + 2] - c[dy + k]) * fx;
		const float a01 = c[dz + k] + (c[dz + k + 2] - c[dz + k]) * fx, a11 = c[dz + dy + k] + (c[dz + dy + k + 2] - c[dz + dy + k]) * fx;
		const float b0 = a00 + (a10 - a00) * fy, b1 = a01 + (a11 - a01) * fy;
		o[k] = b0 + (b1 - b0) * fz;
	}
	re = o[0];
	im = inv ? -o[1] : o[1];
}

struct Vol
{
	long X, Y, Z, initY, initZ;
	size_t n;
	float *re, *im, *w;
	std::vector<std::atomic_flag> locks;
	Vol(long x, long y, long z, long iy, long iz) : X(x), Y(y), Z(z), initY(iy), initZ(iz), n(x * y * z), re(floats(n)), im(floats(n)), w(floats(n)), locks(y * z) {}
	~Vol() { std::free(re); std::free(im); std::free(w); }
	void zero() { std::memset(re, 0, n * 4); std::memset(im, 0, n * 4); std::memset(w, 0, n * 4); }
	double weight() const { double s = 0; for (size_t i = 0; i < n; i++) s += w[i]; return s; }
	inline void fetch(float xp, float yp, float zp) const
	{
		const long i = (((long)std::floor(zp) - initZ) * Y + (long)std::floor(yp) - initY) * X + (long)std::floor(xp);
		for (const float *a : {re, im, w})
			for (long d : {0L, X, X * Y, X * Y + X}) __builtin_prefetch(a + i + d, 1);
	}
	inline void add2(long row, long i, float g0, float g1, float r, float m, float f, bool lock)
	{
		if (lock) while (locks[row].test_and_set(std::memory_order_acquire)) {}
		re[i] += g0 * r; im[i] += g0 * m; w[i] += g0 * f;
		re[i + 1] += g1 * r; im[i + 1] += g1 * m; w[i + 1] += g1 * f;
		if (lock) locks[row].clear(std::memory_order_release);
	}
	// one pixel of backprojectRef3D's splat
	inline void splat(float xp, float yp, float zp, float r, float m, float f, bool lock)
	{
		const int x0 = std::floor(xp), yy = std::floor(yp), zz = std::floor(zp);
		const float fx = xp - x0, fy = yp - yy, fz = zp - zz, mfx = 1 - fx, mfy = 1 - fy, mfz = 1 - fz;
		const long y0 = yy - initY, z0 = zz - initZ, i = (z0 * Y + y0) * X + x0;
		add2(z0 * Y + y0, i, mfz * mfy * mfx, mfz * mfy * fx, r, m, f, lock);
		add2(z0 * Y + y0 + 1, i + X, mfz * fy * mfx, mfz * fy * fx, r, m, f, lock);
		add2((z0 + 1) * Y + y0, i + X * Y, fz * mfy * mfx, fz * mfy * fx, r, m, f, lock);
		add2((z0 + 1) * Y + y0 + 1, i + X * Y + X, fz * fy * mfx, fz * fy * fx, r, m, f, lock);
	}
};

// The same splat into ONE array of (re, im, weight) per voxel: 4 cache lines per pixel instead of 12.
struct Vol3
{
	long X, Y, Z, initY, initZ;
	size_t n;
	float *v;
	Vol3(long x, long y, long z, long iy, long iz) : X(x), Y(y), Z(z), initY(iy), initZ(iz), n(x * y * z), v(floats(3 * n)) {}
	~Vol3() { std::free(v); }
	double weight() const { double s = 0; for (size_t i = 0; i < n; i++) s += v[3 * i + 2]; return s; }
	inline void fetch(float xp, float yp, float zp) const
	{
		const long i = (((long)std::floor(zp) - initZ) * Y + (long)std::floor(yp) - initY) * X + (long)std::floor(xp);
		for (long d : {0L, X, X * Y, X * Y + X}) { __builtin_prefetch(v + 3 * (i + d), 1); __builtin_prefetch(v + 3 * (i + d) + 5, 1); }
	}
	inline void add2(long i, float g0, float g1, float r, float m, float f)
	{
		float *q = &v[3 * i];
		q[0] += g0 * r; q[1] += g0 * m; q[2] += g0 * f;
		q[3] += g1 * r; q[4] += g1 * m; q[5] += g1 * f;
	}
	inline void splat(float xp, float yp, float zp, float r, float m, float f)
	{
		const int x0 = std::floor(xp), yy = std::floor(yp), zz = std::floor(zp);
		const float fx = xp - x0, fy = yp - yy, fz = zp - zz, mfx = 1 - fx, mfy = 1 - fy, mfz = 1 - fz;
		const long i = ((zz - initZ) * Y + yy - initY) * X + x0;
		add2(i, mfz * mfy * mfx, mfz * mfy * fx, r, m, f);
		add2(i + X, mfz * fy * mfx, mfz * fy * fx, r, m, f);
		add2(i + X * Y, fz * mfy * mfx, fz * mfy * fx, r, m, f);
		add2(i + X * Y + X, fz * fy * mfx, fz * fy * fx, r, m, f);
	}
};

struct Particle
{
	Rec r;
	Rec mdl;
	long nx, ny, P, O, T, maxR;
	Model m;
	const float *e, *xr, *xi, *nr, *ni, *tx, *ty, *w, *ctf, *minv;
	float sig, wn_inv, pf_bp;
	long bp_maxR2;
	bool premult;
	std::vector<float> sx, cx, sy, cy;         // [T, nx], [T, ny]
	void tables(const std::vector<int> &ts)
	{
		sx.resize(T * nx); cx.resize(T * nx); sy.resize(T * ny); cy.resize(T * ny);
		for (int t : ts)
		{
			for (int x = 0; x < nx; x++) { sx[t * nx + x] = std::sin(x * tx[t]); cx[t * nx + x] = std::cos(x * tx[t]); }
			for (int y = 0; y < ny; y++) { sy[t * ny + y] = std::sin(y * ty[t]); cy[t * ny + y] = std::cos(y * ty[t]); }
		}
	}
	inline void phase_y(int t, int y, float &c, float &s) const
	{
		if (y < 0) { c = cy[t * ny - y]; s = -sy[t * ny - y]; } else { c = cy[t * ny + y]; s = sy[t * ny + y]; }
	}
};

static double rel(const std::vector<double> &a, const float *b, size_t n)
{
	double d = 0, s = 0;
	for (size_t i = 0; i < n; i++) { d += (a[i] - b[i]) * (a[i] - b[i]); s += (double)b[i] * b[i]; }
	return std::sqrt(d / (s > 0 ? s : 1));
}

int main(int argc, char **argv)
{
	std::printf("# particle O O_sig T T_sig pairs P | relion: tables project pairs bp(locks) bp(no locks) | sparse: all(locks) all(no locks) "
	            "splat(locks) splat(no locks) splat(interleaved) | rel err vs record: relion parts XA AA, sparse parts XA AA, bp weight relion sparse interleaved\n");
	double tot[10] = {0};
	int count = 0;
	for (int a = 1; a < argc; a++)
	{
		Particle p;
		p.r = load(argv[a]);
		const Rec &r = p.r;
		if (!r.count("bp_index")) continue;
		std::string path = argv[a];
		const size_t slash = path.rfind('/'), d1 = path.find('.', slash == std::string::npos ? 0 : slash), d2 = path.find('.', d1 + 1);
		p.mdl = load(path.substr(0, slash == std::string::npos ? 0 : slash + 1) + "model." + path.substr(d1 + 1, d2 - d1 - 1) + "." + std::to_string(I(r, "model")) + ".bin");
		p.nx = I(r, "imgX"); p.ny = I(r, "imgY"); p.P = I(r, "image_size"); p.O = I(r, "orientation_num"); p.T = I(r, "translation_num"); p.maxR = I(r, "maxR");
		const size_t mdl_floats = p.mdl.at("mdlComplex").size() / 4;
		float *mdl = floats(mdl_floats);
		std::memcpy(mdl, F(p.mdl, "mdlComplex"), mdl_floats * 4);
		p.m = Model{mdl, I(r, "mdlX"), I(r, "mdlY"), I(r, "mdlZ"), I(r, "mdlInitY"), I(r, "mdlInitZ"), I(r, "maxR2_padded"), (float)D(r, "padding_factor")};
		p.e = F(r, "eulers"); p.xr = F(r, "Fimg_real"); p.xi = F(r, "Fimg_imag"); p.nr = F(r, "Fimg_nomask_real"); p.ni = F(r, "Fimg_nomask_imag");
		p.tx = F(r, "trans_x"); p.ty = F(r, "trans_y"); p.w = F(r, "weights"); p.ctf = F(r, "ctfs"); p.minv = F(r, "Minvsigma2s");
		p.sig = (float)D(r, "significant_weight"); p.wn_inv = 1.0f / (float)D(r, "weight_norm"); p.pf_bp = (float)D(r, "bp_padding_factor");
		p.bp_maxR2 = I(r, "bp_maxR2"); p.premult = I(r, "ctf_premultiplied");
		const bool refctf = I(r, "refs_are_ctf_corrected");
		const float part_scale = (float)D(r, "part_scale");
		const long nx = p.nx, ny = p.ny, P = p.P, O = p.O, T = p.T;
		Vol vol(I(r, "bp_mdlX"), I(r, "bp_mdlY"), I(r, "bp_mdlZ"), I(r, "bp_mdlInitY"), I(r, "bp_mdlInitZ"));
		const int max_r2_vol = p.bp_maxR2 * p.pf_bp * p.pf_bp;
		const int half = ny / 2;

		std::vector<int> all_t(T), sig_t;
		for (int t = 0; t < T; t++) all_t[t] = t;
		std::vector<std::vector<int>> ts(O);
		std::vector<int> sig_o;
		std::vector<char> t_used(T, 0);
		long pairs = 0;
		for (long o = 0; o < O; o++)
		{
			for (int t = 0; t < T; t++) if (p.w[o * T + t] >= p.sig) { ts[o].push_back(t); t_used[t] = 1; pairs++; }
			if (!ts[o].empty()) sig_o.push_back(o);
		}
		for (int t = 0; t < T; t++) if (t_used[t]) sig_t.push_back(t);

		// ---- relion: wavg
		evict();
		double t0 = cpu_ms();
		p.tables(all_t);
		const double a_tables = cpu_ms() - t0;
		std::vector<float> ref_r(nx), ref_i(nx);
		std::vector<double> parts(P, 0), XA(P, 0), AA(P, 0);
		double a_project = 0, a_all = 0;
		for (int pass = 0; pass < 2; pass++)        // pass 0: projections alone, pass 1: the kernel
		{
			evict();
			t0 = cpu_ms();
			float sink = 0;
			for (long o = 0; o < O; o++)
				for (int iy = 0; iy < ny; iy++)
				{
					const int y = iy > p.maxR ? iy - ny : iy;
					const long px = iy * nx;
					for (int x = 0; x < nx; x++)
					{
						project(p.m, x, y, p.e + 9 * o, ref_r[x], ref_i[x]);
						const float s = refctf ? p.ctf[px + x] : part_scale;
						ref_r[x] *= s; ref_i[x] *= s;
					}
					if (pass == 0) { sink += ref_r[nx / 2]; continue; }
					for (int t : ts[o])
					{
						const float wt = p.w[o * T + t] * p.wn_inv;
						float c, s;
						p.phase_y(t, y, c, s);
						const float *sxr = &p.sx[t * nx], *cxr = &p.cx[t * nx];
						for (int x = 0; x < nx; x++)
						{
							const float ss = sxr[x] * c + cxr[x] * s, cc = cxr[x] * c - sxr[x] * s;
							const float tr = cc * p.xr[px + x] - ss * p.xi[px + x], ti = cc * p.xi[px + x] + ss * p.xr[px + x];
							const float dr = ref_r[x] - tr, di = ref_i[x] - ti;
							parts[px + x] += wt * (dr * dr + di * di);
							XA[px + x] += wt * (ref_r[x] * tr + ref_i[x] * ti);
							AA[px + x] += wt * (ref_r[x] * ref_r[x] + ref_i[x] * ref_i[x]);
						}
					}
				}
			(pass ? a_all : a_project) = cpu_ms() - t0;
			if (sink != sink) std::printf("nan\n");
		}
		const double ea[3] = {rel(parts, F(r, "wdiff2s_parts"), P), rel(XA, F(r, "wdiff2s_XA"), P), rel(AA, F(r, "wdiff2s_AA"), P)};

		// ---- relion: backproject, with and without the row locks
		double b_ms[2], wsum_relion = 0;
		std::vector<float> re(nx), im(nx), fw(nx);
		for (int lock = 1; lock >= 0; lock--)
		{
			vol.zero();
			evict();
			t0 = cpu_ms();
			p.tables(all_t);
			for (long o = 0; o < O; o++)
			{
				const float *e = p.e + 9 * o;
				for (int iy = 0; iy < ny; iy++)
				{
					const int y = iy > half ? iy - ny : iy;
					const int xmax = std::sqrt((float)(half * half - y * y));
					const long px = iy * nx;
					std::fill(re.begin(), re.end(), 0); std::fill(im.begin(), im.end(), 0); std::fill(fw.begin(), fw.end(), 0);
					for (int t : ts[o])
					{
						const float wt = p.w[o * T + t];
						float c, s;
						p.phase_y(t, y, c, s);
						const float *sxr = &p.sx[t * nx], *cxr = &p.cx[t * nx];
						for (int x = 0; x < xmax; x++)
						{
							const float g = wt * p.wn_inv * p.minv[px + x] * (p.premult ? 1.0f : p.ctf[px + x]);
							fw[x] += g * p.ctf[px + x];
							const float ss = sxr[x] * c + cxr[x] * s, cc = cxr[x] * c - sxr[x] * s;
							re[x] += g * (cc * p.nr[px + x] - ss * p.ni[px + x]);
							im[x] += g * (cc * p.ni[px + x] + ss * p.nr[px + x]);
						}
					}
					for (int x = 0; x < xmax; x++)
					{
						float xp = (e[0] * x + e[1] * y) * p.pf_bp, yp = (e[3] * x + e[4] * y) * p.pf_bp, zp = (e[6] * x + e[7] * y) * p.pf_bp;
						if (xp * xp + yp * yp + zp * zp > max_r2_vol || fw[x] <= 0) continue;
						float m = im[x];
						if (xp < 0) { xp = -xp; yp = -yp; zp = -zp; m = -m; }
						vol.splat(xp, yp, zp, re[x], m, fw[x], lock);
					}
				}
			}
			b_ms[lock] = cpu_ms() - t0;
			if (lock) wsum_relion = vol.weight();
		}

		// ---- sparse: one pass per significant orientation; lock 1/0, then the splat alone from kept rows
		double c_ms[2], s_ms[2], wsum_sparse = 0, eb[3] = {0, 0, 0};
		std::vector<float> zr(nx), zi(nx);
		std::vector<float> keep;                      // per significant orientation and pixel: re, im, weight
		for (int lock = 1; lock >= 0; lock--)
		{
			vol.zero();
			std::fill(parts.begin(), parts.end(), 0); std::fill(XA.begin(), XA.end(), 0); std::fill(AA.begin(), AA.end(), 0);
			keep.assign(sig_o.size() * P * 3, 0);
			evict();
			t0 = cpu_ms();
			p.tables(sig_t);
			double sw_all = 0;
			size_t ko = 0;
			for (long o : sig_o)
			{
				const float *e = p.e + 9 * o;
				float sw = 0;
				for (int t : ts[o]) sw += p.w[o * T + t] * p.wn_inv;
				sw_all += sw;
				for (int iy = 0; iy < ny; iy++)
				{
					const int y = iy > p.maxR ? iy - ny : iy;
					const int xmax = std::sqrt((float)(half * half - y * y));
					const long px = iy * nx;
					std::fill(zr.begin(), zr.end(), 0); std::fill(zi.begin(), zi.end(), 0);
					for (int t : ts[o])
					{
						const float wt = p.w[o * T + t] * p.wn_inv;
						float c, s;
						p.phase_y(t, y, c, s);
						const float *sxr = &p.sx[t * nx], *cxr = &p.cx[t * nx];
						for (int x = 0; x < nx; x++)
						{
							zr[x] += wt * (cxr[x] * c - sxr[x] * s);
							zi[x] += wt * (sxr[x] * c + cxr[x] * s);
						}
					}
					for (int x = 0; x < nx; x++)
					{
						float rr, ri;
						project(p.m, x, y, e, rr, ri);
						const float s = refctf ? p.ctf[px + x] : part_scale;
						rr *= s; ri *= s;
						const float yr = zr[x] * p.xr[px + x] - zi[x] * p.xi[px + x], yi = zr[x] * p.xi[px + x] + zi[x] * p.xr[px + x];
						XA[px + x] += rr * yr + ri * yi;
						AA[px + x] += sw * (rr * rr + ri * ri);
					}
					float *k = &keep[(ko * P + px) * 3];
					for (int x = 0; x < xmax; x++)
					{
						float xp = (e[0] * x + e[1] * y) * p.pf_bp, yp = (e[3] * x + e[4] * y) * p.pf_bp, zp = (e[6] * x + e[7] * y) * p.pf_bp;
						if (xp * xp + yp * yp + zp * zp > max_r2_vol) continue;
						const float g = p.minv[px + x] * (p.premult ? 1.0f : p.ctf[px + x]);
						const float f = sw * g * p.ctf[px + x];
						if (f <= 0) continue;
						const float br = g * (zr[x] * p.nr[px + x] - zi[x] * p.ni[px + x]);
						float bi = g * (zr[x] * p.ni[px + x] + zi[x] * p.nr[px + x]);
						k[3 * x] = br; k[3 * x + 1] = bi; k[3 * x + 2] = f;
						if (xp < 0) { xp = -xp; yp = -yp; zp = -zp; bi = -bi; }
						vol.splat(xp, yp, zp, br, bi, f, lock);
					}
				}
				ko++;
			}
			for (long i = 0; i < P; i++) parts[i] = AA[i] - 2 * XA[i] + sw_all * ((double)p.xr[i] * p.xr[i] + (double)p.xi[i] * p.xi[i]);
			c_ms[lock] = cpu_ms() - t0;
			if (lock)
			{
				wsum_sparse = vol.weight();
				eb[0] = rel(parts, F(r, "wdiff2s_parts"), P); eb[1] = rel(XA, F(r, "wdiff2s_XA"), P); eb[2] = rel(AA, F(r, "wdiff2s_AA"), P);
			}
			vol.zero();
			evict();
			t0 = cpu_ms();
			ko = 0;
			for (long o : sig_o)
			{
				const float *e = p.e + 9 * o;
				for (int iy = 0; iy < ny; iy++)
				{
					const int y = iy > half ? iy - ny : iy;
					const int xmax = std::sqrt((float)(half * half - y * y));
					const float *k = &keep[(ko * P + iy * nx) * 3];
					if (PREFETCH)
						for (int x = 0; x < xmax; x++)
						{
							if (k[3 * x + 2] <= 0) continue;
							const float xp = (e[0] * x + e[1] * y) * p.pf_bp, yp = (e[3] * x + e[4] * y) * p.pf_bp, zp = (e[6] * x + e[7] * y) * p.pf_bp;
							if (xp < 0) vol.fetch(-xp, -yp, -zp); else vol.fetch(xp, yp, zp);
						}
					for (int x = 0; x < xmax; x++)
					{
						if (k[3 * x + 2] <= 0) continue;
						float xp = (e[0] * x + e[1] * y) * p.pf_bp, yp = (e[3] * x + e[4] * y) * p.pf_bp, zp = (e[6] * x + e[7] * y) * p.pf_bp;
						float bi = k[3 * x + 1];
						if (xp < 0) { xp = -xp; yp = -yp; zp = -zp; bi = -bi; }
						vol.splat(xp, yp, zp, k[3 * x], bi, k[3 * x + 2], lock);
					}
				}
				ko++;
			}
			s_ms[lock] = cpu_ms() - t0;
		}
		// ---- the splat alone into the interleaved volume
		double s3_ms, wsum3 = 0;
		{
			Vol3 v3(vol.X, vol.Y, vol.Z, vol.initY, vol.initZ);
			evict();
			const double t3 = cpu_ms();
			size_t ko = 0;
			for (long o : sig_o)
			{
				const float *e = p.e + 9 * o;
				for (int iy = 0; iy < ny; iy++)
				{
					const int y = iy > half ? iy - ny : iy;
					const int xmax = std::sqrt((float)(half * half - y * y));
					const float *k = &keep[(ko * P + iy * nx) * 3];
					if (PREFETCH)
						for (int x = 0; x < xmax; x++)
						{
							if (k[3 * x + 2] <= 0) continue;
							const float xp = (e[0] * x + e[1] * y) * p.pf_bp, yp = (e[3] * x + e[4] * y) * p.pf_bp, zp = (e[6] * x + e[7] * y) * p.pf_bp;
							if (xp < 0) v3.fetch(-xp, -yp, -zp); else v3.fetch(xp, yp, zp);
						}
					for (int x = 0; x < xmax; x++)
					{
						if (k[3 * x + 2] <= 0) continue;
						float xp = (e[0] * x + e[1] * y) * p.pf_bp, yp = (e[3] * x + e[4] * y) * p.pf_bp, zp = (e[6] * x + e[7] * y) * p.pf_bp;
						float bi = k[3 * x + 1];
						if (xp < 0) { xp = -xp; yp = -yp; zp = -zp; bi = -bi; }
						v3.splat(xp, yp, zp, k[3 * x], bi, k[3 * x + 2]);
					}
				}
				ko++;
			}
			s3_ms = cpu_ms() - t3;
			wsum3 = v3.weight();
		}
		std::free(mdl);
		double wsum_rec = 0;
		const float *bw = F(r, "bp_weight");
		for (size_t i = 0; i < r.at("bp_weight").size() / 4; i++) wsum_rec += bw[i];
		std::printf("%ld %ld %zu %ld %zu %ld %ld | %.2f %.2f %.2f %.2f %.2f | %.2f %.2f %.2f %.2f %.2f | %.1e %.1e %.1e, %.1e %.1e %.1e, %.1e %.1e %.1e\n",
		            I(r, "part_id"), O, sig_o.size(), T, sig_t.size(), pairs, P,
		            a_tables, a_project, a_all - a_project, b_ms[1], b_ms[0], c_ms[1], c_ms[0], s_ms[1], s_ms[0], s3_ms,
		            ea[0], ea[1], ea[2], eb[0], eb[1], eb[2], std::fabs(wsum_relion / wsum_rec - 1), std::fabs(wsum_sparse / wsum_rec - 1),
		            std::fabs(wsum3 / wsum_rec - 1));
		const double v[10] = {a_tables, a_project, a_all - a_project, b_ms[1], b_ms[0], c_ms[1], c_ms[0], s_ms[1], s_ms[0], s3_ms};
		for (int i = 0; i < 10; i++) tot[i] += v[i];
		count++;
	}
	if (count)
		std::printf("%s%smean of %d: relion tables %.2f project %.2f pairs %.2f bp %.2f (no locks %.2f) = %.2f | sparse %.2f (no locks %.2f) | splat alone %.2f (no locks %.2f, one interleaved volume %.2f)\n",
		            HUGE ? "HUGE " : "", PREFETCH ? "PREFETCH " : "", count, tot[0] / count, tot[1] / count, tot[2] / count, tot[3] / count, tot[4] / count, (tot[0] + tot[1] + tot[2] + tot[3]) / count,
		            tot[5] / count, tot[6] / count, tot[7] / count, tot[8] / count, tot[9] / count);
	return 0;
}
