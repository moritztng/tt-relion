#ifndef ACC_TT_SUMS_RECORD_H_
#define ACC_TT_SUMS_RECORD_H_

// Recorder for storeWeightedSums: what RELION hands its two sums kernels for one particle, and what they
// and the code around them produce. perf/rlx/wsum/ reads the files back and checks a float64 reference
// against them; the same files grade a device op without RELION in the loop.
//
// TT_RELION_RECORD_SUMS=<dir> turns it on, TT_RELION_RECORD_SUMS_N=<n> (default 8) is the number of
// particles per process. One file per particle, <dir>/sums.<pid>.<part_id>.bin, plus the padded Fourier
// reference once per process in <dir>/model.<pid>.<k>.bin. Off, every entry point is one branch.
//
// The three calls of a particle come from the same thread in this order: wavg, backproject, particle.
// Single class, 3D reference, 2D data only; anything else is not recorded.

#include <cstddef>

namespace TTSumsRecord
{
	// True once wavg() has opened a record on this thread and until particle() closes it.
	bool active();

	// After runWavgKernel. The three outputs are per pixel and were zero before the call.
	void wavg(
			const float *mdlComplex, int mdlX, int mdlY, int mdlZ, int mdlInitY, int mdlInitZ,
			int maxR, int maxR2_padded, float padding_factor, int imgX, int imgY,
			const float *eulers, const float *Fimg_real, const float *Fimg_imag,
			const float *trans_x, const float *trans_y, const float *weights, const float *ctfs,
			const float *wdiff2s_parts, const float *wdiff2s_AA, const float *wdiff2s_XA,
			long orientation_num, long translation_num, long image_size,
			double weight_norm, double significant_weight, double part_scale,
			bool refs_are_ctf_corrected, bool ctf_premultiplied);

	// After runBackProjectKernel. model_* is what the same kernel adds to an all-zero volume for this
	// particle (the caller runs it a second time into scratch); stored sparse. stride 3 = the three arrays
	// are one interleaved block (relion5-sums-memory.patch).
	void backproject(
			const float *Fimg_nomask_real, const float *Fimg_nomask_imag, const float *Minvsigma2s, long image_size,
			int bpMaxR, int bpMaxR2, float bp_padding_factor,
			int bpMdlX, int bpMdlY, int bpMdlZ, int bpMdlInitY, int bpMdlInitZ,
			const float *model_real, const float *model_imag, const float *model_weight, long stride = 1);

	// At the end of storeWeightedSums, before the sums are added to wsum_model. Closes the record.
	void particle(
			long part_id, int image_full_size, int image_current_size,
			double sum_weight, double significant_weight, double min_diff2,
			double old_offset_x, double old_offset_y, double prior_x, double prior_y,
			const int *Mresol_fine, long n_resol,
			const double *data_vs_prior, long n_dvp,
			const float *fine_weights, const size_t *rot_id, const size_t *rot_idx,
			const size_t *trans_idx, const size_t *ihidden_overs, long n_fine,
			const double *wsum_sigma2_noise, long n_sigma2,
			const double *wsum_pdf_direction, long n_dir,
			double wsum_norm_correction, double wsum_scale_XA, double wsum_scale_AA,
			double wsum_sigma2_offset, double sumw_group, double dLL, double norm_correction, double pmax);
}

#endif /* ACC_TT_SUMS_RECORD_H_ */
