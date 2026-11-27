// oz_cut compute: r' = X 2^(1 - ea) (exact), then per digit i: q = round(r') -> bf16 digit plane i (|q| <= 2),
// r' = 4 (r' - q) (both exact). The digit decomposition is exact: X 2^-ea = sum_i q_i 2^-(2i+1) + residual.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rounding.h"
#include "api/compute/tile_move_copy.h"

namespace {
constexpr uint32_t cb_x = 0, cb_e = 1, cb_o = 16;
inline void cp(uint32_t cb, uint32_t t, uint32_t d) {
    copy_tile_to_dst_init_short(cb);
    copy_tile(cb, t, d);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t G = get_compile_time_arg_val(1);   // digits per dest acquire (slots: r', 0, G digits)
    constexpr uint32_t FOUR = 0x40800000u;
    const uint32_t nu = get_arg_val<uint32_t>(1);
    binary_op_init_common(cb_x, cb_e, cb_o);
    for (uint32_t u = 0; u < nu; ++u) {
        for (uint32_t i0 = 0; i0 < N; i0 += G) {
            const uint32_t g = (N - i0) < G ? (N - i0) : G;
            tile_regs_acquire();
            // r' is recomputed from the X tile each group (X and E stay in their CBs until the last group): an fp32
            // dest -> CB -> dest spill of r' lost low bits from digit 8 on (runs/ozx1), recomputing is exact.
            if (i0 == 0) {
                cb_wait_front(cb_x, 1);
                cb_wait_front(cb_e, 1);
            }
            cp(cb_x, 0, 0);
            cp(cb_e, 0, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            sub_binary_tile_init();
            sub_binary_tile(0, 0, 1);              // zero tile (r' finite)
            for (uint32_t j = 0; j < i0 + g; ++j) {
                const uint32_t d = j < i0 ? 2 : 2 + (j - i0);
                add_binary_tile_init();
                add_binary_tile(0, 1, d);          // q = r'
                rounding_op_tile_init();
                round_tile(d, 0);
                sub_binary_tile_init();
                sub_binary_tile(0, d, 0);          // r' - q (exact)
                binop_with_scalar_tile_init();
                mul_unary_tile(0, FOUR);           // next digit's scale (exact)
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_o);
            cb_reserve_back(cb_o, g);
            for (uint32_t j = 0; j < g; ++j) pack_tile(2 + j, cb_o);
            cb_push_back(cb_o, g);
            if (i0 + G >= N) {
                cb_pop_front(cb_x, 1);
                cb_pop_front(cb_e, 1);
            }
            tile_regs_release();
        }
    }
}
