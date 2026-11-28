/* tt_prep.h -- a particle's masked and unmasked transforms without per-particle setup.
 *
 * getFourierTransformsAndCtfs does, per particle and per iteration: scale by the norm correction, move by
 * the rounded old offset, transform (unmasked, for the reconstruction), soft-mask, transform again (masked,
 * for alignment), window both to the current size and sum the masked power per shell. RELION's accelerated
 * path spends most of that time around the work: it allocates and zeroes seven arrays per particle,
 * recomputes sqrt and cosf for every pixel of the mask twice, and copies each transform three times.
 *
 * This keeps all of the work and drops the setup. What depends only on the box, the mask radius and the
 * edge width (mask weight per pixel, shell of every Fourier pixel, the FFT plans) is made once per process;
 * the shifted image and the FFT buffers are per thread and kept across particles.
 *
 * The transform is the same single-precision FFTW, called as its two passes: every row real to complex,
 * then the columns, in blocks of 32. The unmasked transform is only ever read inside the current-size
 * window, so its column pass runs over the window's columns and no others. Every value RELION keeps is
 * computed; the columns it throws away right after the transform are not. With RELION's planner flag
 * (FFTW_ESTIMATE, the default here) the passes are the ones its 2D plan runs and the output is the same
 * bits (perf/rlx/prep/bench_prep.cpp prints the difference: 0).
 *
 * The two transforms are most of what is left of the function, and an estimated plan for a 360 box is about
 * 1.6x slower than a measured one. A measured plan depends on timing, so two ranks or two runs would not
 * round alike. TT_RELION_PREP_WISDOM=<file> takes the plans from a file instead: written once per machine
 * by write_wisdom (bench_prep wisdom <box> <file>), it gives every rank and every run the same measured
 * plans. The result then differs from RELION's by float rounding in the FFT (about 1e-7 relative) and is
 * graded against the float64 reference like everything else. A plan the file does not hold is estimated.
 *
 * Arithmetic outside the FFT is RELION's, operation for operation, in single precision where RELION is
 * single, with two exceptions that are sums: the background value under the mask and the shell sums are
 * accumulated in double here, where RELION accumulates float32 in 128 interleaved stripes.
 *
 * The shift drops what leaves the box and leaves zeros behind (cpu_translate2D). It is not circular, so a
 * Fourier phase ramp on one kept transform is not the same image and is not used.
 *
 * Covers 2D images with an even square box and --zero_mask. The caller falls back to RELION's code for
 * everything else. TT_RELION_PREP=0 switches it off (the N arm of an A/B).
 */
#ifndef TT_PREP_H_
#define TT_PREP_H_

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace TTPrep
{

bool enabled();

struct Tables
{
	int n;                        // box edge
	float radius, edge;           // mask radius and width of its cosine edge, pixels
	std::vector<float> weight;    // n*n: 0 inside the mask, raised cosine across the edge, 1 outside
	double weight_sum;
	std::vector<uint16_t> shell;  // n*(n/2+1): shell of each Fourier pixel, n/2+1 for the ones RELION skips
	enum { BLOCK = 32 };          // rows or columns per FFT call; a block starts on a 64-byte boundary
	void *rows[2];                // fftwf plans: BLOCK rows real to complex, and the n % BLOCK that remain
	void *columns[BLOCK + 1];     // columns[k]: k neighbouring columns, complex, in place
	int from_wisdom;              // how many of the plans came from TT_RELION_PREP_WISDOM
};

// Built on first use, shared by all threads, never freed.
const Tables &tables(int n, float radius, float edge);

// Buffers owned by the calling thread, kept across particles.
struct Work
{
	float *shifted;               // n*n: the image after norm correction and shift
	float *real;                  // n*n: FFT input
	float *freq;                  // n*(n/2+1) complex: FFT output
};
Work &work(int n);

// freq = unnormalised transform of real, as fftwf_plan_dft_r2c_2d lays it out. Only the first `columns`
// columns are finished; the rest hold the row pass alone and must not be read.
void forward(const Tables &t, const float *real, float *freq, int columns);

// out = img * norm moved by (dx, dy) whole pixels, zero where nothing lands.
template <typename R>
inline void shift(const R *img, int n, float norm, int dx, int dy, float *out)
{
	memset(out, 0, sizeof(float) * (size_t)n * n);
	const int x0 = dx < 0 ? -dx : 0, x1 = dx > 0 ? n - dx : n;
	const int y0 = dy < 0 ? -dy : 0, y1 = dy > 0 ? n - dy : n;
	for (int y = y0; y < y1; y++)
	{
		const R *s = img + (size_t)y * n;
		float *d = out + (size_t)(y + dy) * n + dx;
		if (norm != 1)
			for (int x = x0; x < x1; x++) d[x] = (float)s[x] * norm;
		else
			for (int x = x0; x < x1; x++) d[x] = (float)s[x];
	}
}

// Measures the plans of an n box (FFTW_MEASURE, about a second) and writes them as FFTW wisdom.
bool write_wisdom(int n, const char *path);

// The image with its quadrants swapped (RELION's centerFFT), as FFT input.
void center(const float *in, int n, float *out);

// The same after the soft mask: outside the mask the image is replaced by its own weighted mean there.
void mask_center(const Tables &t, const float *in, float *out);

// The cur x (cur/2+1) low-frequency window of an n x (n/2+1) transform, times scale. C is RELION's Complex.
// Reads columns 0..cur/2 only.
template <typename C>
inline void window(const float *f, int n, int cur, float scale, C *out)
{
	const int ix = n / 2 + 1, ox = cur / 2 + 1;
	for (int i = 0; i < cur; i++)
	{
		const float *s = f + 2 * (size_t)(i < ox ? i : i - cur + n) * ix;
		C *d = out + (size_t)i * ox;
		for (int j = 0; j < ox; j++)
		{
			d[j].real = s[2 * j] * scale;
			d[j].imag = s[2 * j + 1] * scale;
		}
	}
}

// spectrum[s] = sum of |f * scale|^2 over shell s, for s in 0..n/2. Returns the sum over shells >= first.
double power(const Tables &t, const float *f, float scale, int first, double *spectrum);

// TT_RELION_RECORD_PREP=<dir>: what went into and came out of the first TT_RELION_RECORD_PREP_N (default 8)
// particles of this process, one file each, for perf/rlx/prep/check_prep.py (float64 reference). A file is
// a sequence of items: a text line "name dtype ydim xdim" followed by ydim*xdim raw values.
class Record
{
	FILE *f;
public:
	explicit Record(long part);
	~Record() { if (f) fclose(f); }
	Record(const Record &) = delete;
	explicit operator bool() const { return f; }
	void put(const char *name, double v) { put(name, "f8", &v, sizeof v, 1, 1); }
	void put(const char *name, const char *dtype, const void *data, size_t item, long ydim, long xdim);
};

}

#endif
