// oz_combine compute: per output tile, (hi, lo) = TwoSum-accumulate of S_k * 2^-(2k+2) over k (exact digit-pair
// integers scaled by powers of two), then both times E = 2^(ea[r] + eb[c]) (exact). SFPU fp32, full-sync dest (8 slots).
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"

namespace {
constexpr uint32_t cb_s = 0, cb_c = 1, cb_e = 2, cb_hi = 16, cb_lo = 17;
inline void cp(uint32_t cb, uint32_t t, uint32_t d) {
    copy_tile_to_dst_init_short(cb);
    copy_tile(cb, t, d);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    const uint32_t nu = get_arg_val<uint32_t>(1);
    binary_op_init_common(cb_s, cb_c, cb_hi);
    cb_wait_front(cb_c, N);
    for (uint32_t u = 0; u < nu; ++u) {
        tile_regs_acquire();
        // slots: H hi, 1 lo, 2 b, S s, 4 bb; H and S swap each step
        uint32_t H = 0, S = 3;
        for (uint32_t k = 0; k < N; ++k) {
            cb_wait_front(cb_s, 1);
            cp(cb_s, 0, k == 0 ? H : 2);
            cp(cb_c, k, 4);
            cb_pop_front(cb_s, 1);
            mul_binary_tile_init();
            if (k == 0) {
                mul_binary_tile(H, 4, H);
                sub_binary_tile_init();
                sub_binary_tile(H, H, 1);   // lo = 0
                continue;
            }
            mul_binary_tile(2, 4, 2);
            add_binary_tile_init();
            add_binary_tile(H, 2, S);        // s = a + b
            sub_binary_tile_init();
            sub_binary_tile(S, H, 4);        // bb = s - a
            sub_binary_tile(2, 4, 2);        // db = b - bb
            sub_binary_tile(S, 4, 4);        // a' = s - bb
            sub_binary_tile(H, 4, H);        // da = a - a'
            add_binary_tile_init();
            add_binary_tile(H, 2, H);        // e = da + db
            add_binary_tile(1, H, 1);        // lo += e
            const uint32_t t = H; H = S; S = t;
        }
        cb_wait_front(cb_e, 1);
        cp(cb_e, 0, 4);
        cb_pop_front(cb_e, 1);
        mul_binary_tile_init();
        mul_binary_tile(H, 4, H);
        mul_binary_tile(1, 4, 1);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_hi, 1);
        cb_reserve_back(cb_lo, 1);
        pack_tile(H, cb_hi);
        pack_tile(1, cb_lo);
        cb_push_back(cb_hi, 1);
        cb_push_back(cb_lo, 1);
        tile_regs_release();
    }
}
