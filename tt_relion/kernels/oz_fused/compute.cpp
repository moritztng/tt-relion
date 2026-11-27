// oz_fused compute: S_k block = sum_i sum_runs A_i[run] B_{k-i}[run]. Each run's matmul starts from a fresh (zero)
// dest, so every FPU partial is an exact integer below 2^11 (480 x 4); it is then added to the fp32 accumulator tile on
// the SFPU (exact: integers < 2^24) and packed back in place. The FPU never adds onto a large value.
#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t RB = get_compile_time_arg_val(0), NB = get_compile_time_arg_val(1);
    constexpr uint32_t KR = get_compile_time_arg_val(2), NRUN = get_compile_time_arg_val(3);
    constexpr uint32_t NRB = get_compile_time_arg_val(4), NCB = get_compile_time_arg_val(5);
    constexpr bool BT = get_compile_time_arg_val(6);
    constexpr uint32_t NB_K = get_compile_time_arg_val(7);   // the unpacker transposes each B tile
    constexpr uint32_t OG = 2;   // output tiles per dest pass: OG matmul slots + OG accumulator slots <= 4 fp32
    constexpr uint32_t cb_a = 0, cb_b = 1, cb_acc = 24, cb_done = 25;
    const uint32_t u0 = get_arg_val<uint32_t>(0), nu = get_arg_val<uint32_t>(1);
    if (nu == 0) return;
    mm_block_init(cb_a, cb_b, cb_acc, BT, OG, 1, KR);
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t k = (u / (NRB * NCB)) % NB_K;
        cb_wait_front(cb_acc, RB * NB);
        for (uint32_t i = 0; i <= k; ++i) {
            for (uint32_t run = 0; run < NRUN; ++run) {
                cb_wait_front(cb_b, KR * NB);
                for (uint32_t r = 0; r < RB; ++r) {
                    cb_wait_front(cb_a, KR);
                    for (uint32_t g = 0; g < NB; g += OG) {
                        // odd NB: the last group is one tile wide (a 2-wide group there wrote the next row's tile 0)
                        const uint32_t og = (g + OG <= NB) ? OG : 1;
                        tile_regs_acquire();
                        mm_block_init_short(cb_a, cb_b, BT, og, 1, KR);
                        reconfig_data_format_srca(cb_acc, cb_b);
                        for (uint32_t kk = 0; kk < KR; ++kk) matmul_block(cb_a, cb_b, kk, kk * NB + g, 0, BT, og, 1, KR);
                        reconfig_data_format_srca(cb_b, cb_acc);
                        copy_tile_to_dst_init_short(cb_acc);
                        for (uint32_t t = 0; t < og; ++t) copy_tile(cb_acc, r * NB + g + t, OG + t);
                        add_binary_tile_init();
                        for (uint32_t t = 0; t < og; ++t) add_binary_tile(t, OG + t, t);
                        tile_regs_commit();
                        tile_regs_wait();
                        for (uint32_t t = 0; t < og; ++t) pack_tile<true>(t, cb_acc, r * NB + g + t);
                        tile_regs_release();
                    }
                    cb_pop_front(cb_a, KR);
                }
                cb_pop_front(cb_b, KR * NB);
            }
        }
        cb_reserve_back(cb_done, 1);
        cb_push_back(cb_done, 1);
        cb_pop_front(cb_acc, RB * NB);
    }
}
