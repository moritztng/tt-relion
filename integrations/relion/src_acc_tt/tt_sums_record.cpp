#ifdef _TT_ENABLED

#include "src/acc/tt/tt_sums_record.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

// File format, read by perf/rlx/wsum/sums_io.py: the 8 bytes "TTSUMS1\n", then entries of
//   u16 name length, name, 1 byte type (f float32, d float64, i int32, q int64, Q uint64), u64 count, raw data.
// A scalar is an entry of count 1.

namespace
{
	const char *g_dir = nullptr;
	long g_max = 8;
	std::once_flag g_once;
	std::atomic<long> g_taken{0};

	std::mutex g_model_mutex;
	std::vector<std::pair<const float *, size_t> > g_models;   // written references: pointer and size

	thread_local std::FILE *t_file = nullptr;
	thread_local std::string t_path;

	void init()
	{
		const char *d = std::getenv("TT_RELION_RECORD_SUMS");
		if (d && d[0])
			g_dir = d;
		const char *n = std::getenv("TT_RELION_RECORD_SUMS_N");
		if (n && n[0])
			g_max = std::atol(n);
	}

	void put(std::FILE *f, const char *name, char type, const void *data, size_t count, size_t elem)
	{
		const uint16_t len = (uint16_t)std::strlen(name);
		const uint64_t cnt = count;
		std::fwrite(&len, sizeof len, 1, f);
		std::fwrite(name, 1, len, f);
		std::fwrite(&type, 1, 1, f);
		std::fwrite(&cnt, sizeof cnt, 1, f);
		std::fwrite(data, elem, count, f);
	}
	void f32(std::FILE *f, const char *name, const float *p, size_t n) { put(f, name, 'f', p, n, 4); }
	void f64(std::FILE *f, const char *name, const double *p, size_t n) { put(f, name, 'd', p, n, 8); }
	void i32(std::FILE *f, const char *name, const int *p, size_t n) { put(f, name, 'i', p, n, 4); }
	void u64(std::FILE *f, const char *name, const size_t *p, size_t n) { put(f, name, 'Q', p, n, 8); }
	void num(std::FILE *f, const char *name, double v) { put(f, name, 'd', &v, 1, 8); }
	void integer(std::FILE *f, const char *name, long v)
	{
		const int64_t x = v;
		put(f, name, 'q', &x, 1, 8);
	}

	// Index of this reference among the ones written by this process; written on first sight.
	long modelIndex(const float *mdl, size_t floats)
	{
		std::lock_guard<std::mutex> g(g_model_mutex);
		for (size_t k = 0; k < g_models.size(); k++)
			if (g_models[k].first == mdl && g_models[k].second == floats)
				return (long)k;
		const long k = (long)g_models.size();
		const std::string path = std::string(g_dir) + "/model." + std::to_string((long)getpid()) + "." + std::to_string(k) + ".bin";
		if (std::FILE *f = std::fopen(path.c_str(), "wb"))
		{
			std::fwrite("TTSUMS1\n", 1, 8, f);
			f32(f, "mdlComplex", mdl, floats);
			std::fclose(f);
		}
		g_models.push_back(std::make_pair(mdl, floats));
		return k;
	}
}

namespace TTSumsRecord
{

bool active()
{
	return t_file != nullptr;
}

void wavg(
		const float *mdlComplex, int mdlX, int mdlY, int mdlZ, int mdlInitY, int mdlInitZ,
		int maxR, int maxR2_padded, float padding_factor, int imgX, int imgY,
		const float *eulers, const float *Fimg_real, const float *Fimg_imag,
		const float *trans_x, const float *trans_y, const float *weights, const float *ctfs,
		const float *wdiff2s_parts, const float *wdiff2s_AA, const float *wdiff2s_XA,
		long orientation_num, long translation_num, long image_size,
		double weight_norm, double significant_weight, double part_scale,
		bool refs_are_ctf_corrected, bool ctf_premultiplied)
{
	std::call_once(g_once, init);
	if (!g_dir || t_file || g_taken.load(std::memory_order_relaxed) >= g_max)
		return;
	if (g_taken.fetch_add(1) >= g_max)
		return;
	// The particle id is only known at the end, so the file is renamed there.
	t_path = std::string(g_dir) + "/open." + std::to_string((long)getpid()) + "." + std::to_string((long)g_taken.load());
	t_file = std::fopen(t_path.c_str(), "wb");
	if (!t_file)
		return;
	std::FILE *f = t_file;
	std::fwrite("TTSUMS1\n", 1, 8, f);
	integer(f, "model", modelIndex(mdlComplex, (size_t)mdlX * mdlY * mdlZ * 2));
	integer(f, "mdlX", mdlX);
	integer(f, "mdlY", mdlY);
	integer(f, "mdlZ", mdlZ);
	integer(f, "mdlInitY", mdlInitY);
	integer(f, "mdlInitZ", mdlInitZ);
	integer(f, "maxR", maxR);
	integer(f, "maxR2_padded", maxR2_padded);
	num(f, "padding_factor", padding_factor);
	integer(f, "imgX", imgX);
	integer(f, "imgY", imgY);
	integer(f, "orientation_num", orientation_num);
	integer(f, "translation_num", translation_num);
	integer(f, "image_size", image_size);
	num(f, "weight_norm", weight_norm);
	num(f, "significant_weight", significant_weight);
	num(f, "part_scale", part_scale);
	integer(f, "refs_are_ctf_corrected", refs_are_ctf_corrected);
	integer(f, "ctf_premultiplied", ctf_premultiplied);
	f32(f, "eulers", eulers, (size_t)orientation_num * 9);
	f32(f, "Fimg_real", Fimg_real, image_size);
	f32(f, "Fimg_imag", Fimg_imag, image_size);
	f32(f, "trans_x", trans_x, translation_num);
	f32(f, "trans_y", trans_y, translation_num);
	f32(f, "weights", weights, (size_t)orientation_num * translation_num);
	f32(f, "ctfs", ctfs, image_size);
	f32(f, "wdiff2s_parts", wdiff2s_parts, image_size);
	f32(f, "wdiff2s_AA", wdiff2s_AA, image_size);
	f32(f, "wdiff2s_XA", wdiff2s_XA, image_size);
}

void backproject(
		const float *Fimg_nomask_real, const float *Fimg_nomask_imag, const float *Minvsigma2s, long image_size,
		int bpMaxR, int bpMaxR2, float bp_padding_factor,
		int bpMdlX, int bpMdlY, int bpMdlZ, int bpMdlInitY, int bpMdlInitZ,
		const float *model_real, const float *model_imag, const float *model_weight, long stride)
{
	std::FILE *f = t_file;
	if (!f)
		return;
	f32(f, "Fimg_nomask_real", Fimg_nomask_real, image_size);
	f32(f, "Fimg_nomask_imag", Fimg_nomask_imag, image_size);
	f32(f, "Minvsigma2s", Minvsigma2s, image_size);
	integer(f, "bp_maxR", bpMaxR);
	integer(f, "bp_maxR2", bpMaxR2);
	num(f, "bp_padding_factor", bp_padding_factor);
	integer(f, "bp_mdlX", bpMdlX);
	integer(f, "bp_mdlY", bpMdlY);
	integer(f, "bp_mdlZ", bpMdlZ);
	integer(f, "bp_mdlInitY", bpMdlInitY);
	integer(f, "bp_mdlInitZ", bpMdlInitZ);
	const size_t n = (size_t)bpMdlX * bpMdlY * bpMdlZ;
	std::vector<size_t> idx;
	std::vector<float> re, im, w;
	for (size_t i = 0, k = 0; i < n; i++, k += stride)
		if (model_weight[k] != 0.f || model_real[k] != 0.f || model_imag[k] != 0.f)
		{
			idx.push_back(i);
			re.push_back(model_real[k]);
			im.push_back(model_imag[k]);
			w.push_back(model_weight[k]);
		}
	u64(f, "bp_index", idx.data(), idx.size());
	f32(f, "bp_real", re.data(), re.size());
	f32(f, "bp_imag", im.data(), im.size());
	f32(f, "bp_weight", w.data(), w.size());
}

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
		double wsum_sigma2_offset, double sumw_group, double dLL, double norm_correction, double pmax)
{
	std::FILE *f = t_file;
	if (!f)
		return;
	integer(f, "part_id", part_id);
	integer(f, "image_full_size", image_full_size);
	integer(f, "image_current_size", image_current_size);
	num(f, "sum_weight", sum_weight);
	num(f, "op_significant_weight", significant_weight);
	num(f, "min_diff2", min_diff2);
	num(f, "old_offset_x", old_offset_x);
	num(f, "old_offset_y", old_offset_y);
	num(f, "prior_x", prior_x);
	num(f, "prior_y", prior_y);
	i32(f, "Mresol_fine", Mresol_fine, n_resol);
	f64(f, "data_vs_prior", data_vs_prior, n_dvp);
	f32(f, "fine_weights", fine_weights, n_fine);
	u64(f, "fine_rot_id", rot_id, n_fine);
	u64(f, "fine_rot_idx", rot_idx, n_fine);
	u64(f, "fine_trans_idx", trans_idx, n_fine);
	u64(f, "fine_ihidden_overs", ihidden_overs, n_fine);
	f64(f, "wsum_sigma2_noise", wsum_sigma2_noise, n_sigma2);
	f64(f, "wsum_pdf_direction", wsum_pdf_direction, n_dir);
	num(f, "wsum_norm_correction", wsum_norm_correction);
	num(f, "wsum_scale_XA", wsum_scale_XA);
	num(f, "wsum_scale_AA", wsum_scale_AA);
	num(f, "wsum_sigma2_offset", wsum_sigma2_offset);
	num(f, "sumw_group", sumw_group);
	num(f, "dLL", dLL);
	num(f, "norm_correction", norm_correction);
	num(f, "pmax", pmax);
	std::fclose(f);
	t_file = nullptr;
	const std::string path = std::string(g_dir) + "/sums." + std::to_string((long)getpid()) + "." + std::to_string(part_id) + ".bin";
	std::rename(t_path.c_str(), path.c_str());
}

}

#endif /* _TT_ENABLED */
