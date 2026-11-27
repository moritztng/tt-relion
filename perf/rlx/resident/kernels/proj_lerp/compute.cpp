// proj_lerp compute: RELION's trilinear order (x, then y, then z), each step a + f * (b - a) in fp32 on the SFPU.
// cb_v tile j = corner c value part p, j = 2 c + p, c = 4 dz + 2 dy + dx. cb_f: fx, fy, fz. Out: re, im.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"

namespace {
// dst0 = a + f * (b - a) with a, b from cb_ab tiles ia, ib and f from cb_f tile jf; packs into cb_o.
inline void lerp(uint32_t cb_ab, uint32_t ia, uint32_t ib, uint32_t jf, uint32_t cb_o) {
    constexpr uint32_t cb_f = 1;
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_ab);
    copy_tile(cb_ab, ia, 0);
    copy_tile(cb_ab, ib, 1);
    copy_tile_to_dst_init_short(cb_f);
    copy_tile(cb_f, jf, 2);
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
}  // namespace

void kernel_main() {
    constexpr uint32_t cb_v = 0, cb_f = 1, cb_x = 4, cb_y = 5, cb_out = 16;
    const uint32_t nunits = get_arg_val<uint32_t>(1);
    binary_op_init_common(cb_v, cb_f, cb_out);
    for (uint32_t u = 0; u < nunits; ++u) {
        cb_wait_front(cb_v, 16);
        cb_wait_front(cb_f, 3);
        cb_reserve_back(cb_x, 8);
        for (uint32_t zy = 0; zy < 4; ++zy)          // zy = 2 dz + dy
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
}
