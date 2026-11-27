// oz_combine compute: per output tile, (hi, lo) = TwoSum-accumulate of b_k = S_k * 2^-(2k+2) over k (exact digit-pair
// integers scaled by powers of two), then both times E = 2^(ea[r] + eb[c]) (exact). SFPU fp32, ONE dest acquire per
// tile and only slots 0..3: slot 4+ corrupts here (runs/ozx1, dst_full_sync_en through the descriptor does not give
// 8 fp32 slots), and an in-core fp32 CB spill lost its wrapped write (runs/ozc1). So all n S tiles stay in the CB and
// b_k is reloaded instead of kept (mul_unary_tile gave NaN for k >= 2 scales, so the
// reader applies 2^-(2k+2) as an exponent subtract): Knuth TwoSum, lo += da then lo += db (the exact error split in two fp32 adds).
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"

namespace {
constexpr uint32_t cb_s = 0, cb_e = 2, cb_hi = 16, cb_lo = 17;
inline void ld(uint32_t k, uint32_t d) {   // b_k (the reader already scaled S_k by 2^-(2k+2), exactly)
    copy_tile_to_dst_init_short(cb_s);
    copy_tile(cb_s, k, d);
}
inline void add(uint32_t a, uint32_t b, uint32_t o) { add_binary_tile_init(); add_binary_tile(a, b, o); }
inline void sub(uint32_t a, uint32_t b, uint32_t o) { sub_binary_tile_init(); sub_binary_tile(a, b, o); }
}  // namespace

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    const uint32_t nu = get_arg_val<uint32_t>(1);
    binary_op_init_common(cb_s, cb_e, cb_hi);
    constexpr uint32_t L = 1, X = 2;
    for (uint32_t u = 0; u < nu; ++u) {
        cb_wait_front(cb_s, N);
        cb_wait_front(cb_e, 1);
        tile_regs_acquire();
        uint32_t h = 0, t = 3;
        ld(0, h);
        sub(h, h, L);                    // lo = 0
        for (uint32_t k = 1; k < N; ++k) {
            ld(k, X);                    // b
            add(h, X, t);                // s = a + b
            sub(t, X, X);                // a' = s - b
            sub(h, X, h);                // da = a - a'
            sub(t, X, X);                // b' = s - a'
            add(L, h, L);                // lo += da
            ld(k, h);                    // b again
            sub(h, X, h);                // db = b - b'
            add(L, h, L);                // lo += db
            const uint32_t w = h; h = t; t = w;
        }
        copy_tile_to_dst_init_short(cb_e);
        copy_tile(cb_e, 0, X);
        mul_binary_tile_init();
        mul_binary_tile(h, X, h);
        mul_binary_tile(L, X, L);
        tile_regs_commit();
        cb_pop_front(cb_s, N);
        cb_pop_front(cb_e, 1);
        tile_regs_wait();
        cb_reserve_back(cb_hi, 1);
        cb_reserve_back(cb_lo, 1);
        pack_tile(h, cb_hi);
        pack_tile(L, cb_lo);
        cb_push_back(cb_hi, 1);
        cb_push_back(cb_lo, 1);
        tile_regs_release();
    }
}
