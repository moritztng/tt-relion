// proj_rot compute. Per orientation, cb_c holds 6 broadcast tiles: PF*(e0, e1, e3, e4, e6, e7). Per chunk:
//   p_a = c_2a * x + c_2a+1 * y            (RELION's (e0 x + e1 y) * PF; PF = 2 is exact either way)
//   l = (p_x < 0); q_a = p_a - 2 l p_a     (Friedel flip, exact)
//   floor(q_a) -> cb_i, q_a - floor(q_a) -> cb_f;  raw p_x and r2 = p_x^2 + p_y^2 + p_z^2 -> cb_i (reader masks, flips im)
// then the same 14 fp32 SFPU lerps as proj_lerp on the reader's corner tiles. All arithmetic fp32 on the SFPU.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_unary/rounding.h"
#include "api/compute/tile_move_copy.h"

namespace {
constexpr uint32_t cb_v = 0, cb_f = 1, cb_x = 4, cb_y = 5, cb_c = 6, cb_xy = 7, cb_i = 8, cb_p = 10, cb_q = 11,
                   cb_out = 16;

inline void cp(uint32_t cb, uint32_t t, uint32_t d) {
    copy_tile_to_dst_init_short(cb);
    copy_tile(cb, t, d);
}

inline void lerp(uint32_t cb_ab, uint32_t ia, uint32_t ib, uint32_t jf, uint32_t cb_o) {
    tile_regs_acquire();
    cp(cb_ab, ia, 0);
    cp(cb_ab, ib, 1);
    cp(cb_f, jf, 2);
    sub_binary_tile_init();
    sub_binary_tile(1, 0, 1);
    mul_binary_tile_init();
    mul_binary_tile(1, 2, 1);
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_o);
    tile_regs_release();
}

inline void coords() {
    cb_wait_front(cb_xy, 2);
    cb_reserve_back(cb_p, 3);
    for (uint32_t a = 0; a < 3; ++a) {
        tile_regs_acquire();
        cp(cb_xy, 0, 0);
        cp(cb_c, 2 * a, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        cp(cb_xy, 1, 1);
        cp(cb_c, 2 * a + 1, 2);
        mul_binary_tile_init();
        mul_binary_tile(1, 2, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_p);
        tile_regs_release();
    }
    cb_push_back(cb_p, 3);
    cb_pop_front(cb_xy, 2);
    cb_wait_front(cb_p, 3);
    cb_reserve_back(cb_q, 3);
    for (uint32_t a = 0; a < 3; ++a) {
        tile_regs_acquire();
        cp(cb_p, 0, 0);
        ltz_tile_init();
        ltz_tile(0);
        cp(cb_p, a, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 2);
        add_binary_tile_init();
        add_binary_tile(2, 2, 2);
        sub_binary_tile_init();
        sub_binary_tile(1, 2, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(1, cb_q);
        tile_regs_release();
    }
    cb_push_back(cb_q, 3);
    cb_wait_front(cb_q, 3);
    cb_reserve_back(cb_i, 5);
    cb_reserve_back(cb_f, 3);
    for (uint32_t a = 0; a < 3; ++a) {
        tile_regs_acquire();
        cp(cb_q, a, 0);
        cp(cb_q, a, 1);
        rounding_op_tile_init();
        floor_tile(1);
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_f);
        pack_tile(1, cb_i);
        tile_regs_release();
    }
    cb_pop_front(cb_q, 3);
    tile_regs_acquire();
    cp(cb_p, 0, 0);
    cp(cb_p, 0, 1);
    cp(cb_p, 0, 2);
    mul_binary_tile_init();
    mul_binary_tile(1, 2, 1);
    cp(cb_p, 1, 2);
    cp(cb_p, 1, 3);
    mul_binary_tile_init();
    mul_binary_tile(2, 3, 2);
    add_binary_tile_init();
    add_binary_tile(1, 2, 1);
    cp(cb_p, 2, 2);
    cp(cb_p, 2, 3);
    mul_binary_tile_init();
    mul_binary_tile(2, 3, 2);
    add_binary_tile_init();
    add_binary_tile(1, 2, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_i);
    pack_tile(1, cb_i);
    tile_regs_release();
    cb_pop_front(cb_p, 3);
    cb_push_back(cb_f, 3);
    cb_push_back(cb_i, 5);
}

inline void lerps() {
    cb_wait_front(cb_v, 16);
    cb_wait_front(cb_f, 3);
    cb_reserve_back(cb_x, 8);
    for (uint32_t zy = 0; zy < 4; ++zy)
        for (uint32_t p = 0; p < 2; ++p) lerp(cb_v, 2 * (2 * zy) + p, 2 * (2 * zy + 1) + p, 0, cb_x);
    cb_push_back(cb_x, 8);
    cb_pop_front(cb_v, 16);
    cb_wait_front(cb_x, 8);
    cb_reserve_back(cb_y, 4);
    for (uint32_t z = 0; z < 2; ++z)
        for (uint32_t p = 0; p < 2; ++p) lerp(cb_x, (2 * z) * 2 + p, (2 * z + 1) * 2 + p, 1, cb_y);
    cb_push_back(cb_y, 4);
    cb_pop_front(cb_x, 8);
    cb_wait_front(cb_y, 4);
    cb_reserve_back(cb_out, 2);
    for (uint32_t p = 0; p < 2; ++p) lerp(cb_y, p, 2 + p, 2, cb_out);
    cb_push_back(cb_out, 2);
    cb_pop_front(cb_y, 4);
    cb_pop_front(cb_f, 3);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t NC = get_compile_time_arg_val(0);
    const uint32_t no = get_arg_val<uint32_t>(1);
    binary_op_init_common(cb_xy, cb_c, cb_out);
    for (uint32_t o = 0; o < no; ++o) {
        cb_wait_front(cb_c, 6);
        for (uint32_t ch = 0; ch < NC; ++ch) {
            coords();
            lerps();
        }
        cb_pop_front(cb_c, 6);
    }
}
