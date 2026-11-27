// proj_rot reader. Per orientation: read its 6 PF-scaled rotation coefficients and fill 6 broadcast tiles for compute.
// Per 1024-pixel chunk: read the chunk's pixel x, y tiles; wait for compute's floor(x, y, z) / raw xp / r2 tiles; build
// each pixel's base-voxel index on the RISC-V from the float bits (floors are small integers, r2 >= 0 compares as uint);
// gather the 64 B corner brick per pixel; scatter it into 16 corner tiles, zeroing pixels outside the radius and negating
// the imaginary corners of Friedel-flipped pixels (xp < 0): exact, since lerp(-a, -b, f) == -lerp(a, b, f) in IEEE.
#include "api/dataflow/dataflow_api.h"

namespace {
inline int32_t f2i(uint32_t b) {  // b = bits of an integer-valued float with |v| < 2^23
    const uint32_t a = b & 0x7fffffffu;
    if (a < 0x3f800000u) return 0;
    const int32_t e = int32_t(a >> 23) - 127;
    const int32_t m = int32_t((a & 0x7fffffu) | 0x800000u) >> (23 - e);
    return (b >> 31) ? -m : m;
}
}  // namespace

void kernel_main() {
    constexpr auto tab_args = TensorAccessorArgs<0>();
    constexpr auto pix_args = TensorAccessorArgs<tab_args.next_compile_time_args_offset()>();
    constexpr auto rot_args = TensorAccessorArgs<pix_args.next_compile_time_args_offset()>();
    constexpr uint32_t base = rot_args.next_compile_time_args_offset();
    constexpr uint32_t NC = get_compile_time_arg_val(base);
    constexpr uint32_t LIM = get_compile_time_arg_val(base + 1);   // bits of float(maxR2 + 1)
    constexpr int32_t XS = get_compile_time_arg_val(base + 2);
    constexpr int32_t YS = get_compile_time_arg_val(base + 3);
    constexpr int32_t ZS = get_compile_time_arg_val(base + 4);
    constexpr int32_t IY = get_compile_time_arg_val(base + 5);     // -initY
    constexpr int32_t IZ = get_compile_time_arg_val(base + 6);
    constexpr uint32_t cb_v = 0, cb_c = 6, cb_xy = 7, cb_i = 8, cb_stage = 3, cb_rot = 9, cb_flag = 12;
    constexpr uint32_t PAGE = 4096, BRICK = 64;

    const auto s_tab = TensorAccessor(tab_args, get_common_arg_val<uint32_t>(0), BRICK);
    const auto s_pix = TensorAccessor(pix_args, get_common_arg_val<uint32_t>(1), PAGE);
    const auto s_rot = TensorAccessor(rot_args, get_common_arg_val<uint32_t>(2), 32);
    const uint32_t o0 = get_arg_val<uint32_t>(0);
    const uint32_t no = get_arg_val<uint32_t>(1);

    const uint32_t stg_l1 = get_write_ptr(cb_stage);
    volatile tt_l1_ptr uint32_t* stg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stg_l1);
    const uint32_t rot_l1 = get_write_ptr(cb_rot);
    volatile tt_l1_ptr uint32_t* rot = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rot_l1);
    // per pixel: 0 = normal, 1 = Friedel-flipped, 2 = outside the radius (L1 scratch CB, not the small RISC-V stack)
    volatile tt_l1_ptr uint8_t* flag = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(get_write_ptr(cb_flag));

    for (uint32_t o = o0; o < o0 + no; ++o) {
        noc_async_read_page(o, s_rot, rot_l1);
        noc_async_read_barrier();
        cb_reserve_back(cb_c, 6);
        volatile tt_l1_ptr uint32_t* c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_c));
        for (uint32_t j = 0; j < 6; ++j) {
            const uint32_t v = rot[j];
            for (uint32_t k = 0; k < 1024; ++k) c[j * 1024 + k] = v;
        }
        cb_push_back(cb_c, 6);
        for (uint32_t ch = 0; ch < NC; ++ch) {
            cb_reserve_back(cb_xy, 2);
            const uint32_t xy = get_write_ptr(cb_xy);
            noc_async_read_page(2 * ch, s_pix, xy);
            noc_async_read_page(2 * ch + 1, s_pix, xy + PAGE);
            noc_async_read_barrier();
            cb_push_back(cb_xy, 2);

            cb_wait_front(cb_i, 5);
            volatile tt_l1_ptr uint32_t* in = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_i));
            uint32_t sp = stg_l1;
            for (uint32_t k = 0; k < 1024; ++k, sp += BRICK) {
                if (in[4 * 1024 + k] >= LIM) {
                    flag[k] = 2;
                    continue;
                }
                const uint32_t xb = in[3 * 1024 + k];
                flag[k] = ((xb >> 31) && (xb & 0x7fffffffu)) ? 1 : 0;
                int32_t xi = f2i(in[k]), yi = f2i(in[1024 + k]) + IY, zi = f2i(in[2048 + k]) + IZ;
                xi = xi < 0 ? 0 : (xi > XS - 2 ? XS - 2 : xi);
                yi = yi < 0 ? 0 : (yi > YS - 2 ? YS - 2 : yi);
                zi = zi < 0 ? 0 : (zi > ZS - 2 ? ZS - 2 : zi);
                noc_async_read(s_tab.get_noc_addr(uint32_t((zi * YS + yi) * XS + xi)), sp, BRICK);
            }
            noc_async_read_barrier();
            cb_pop_front(cb_i, 5);
            cb_reserve_back(cb_v, 16);
            volatile tt_l1_ptr uint32_t* v = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_v));
            for (uint32_t k = 0; k < 1024; ++k) {
                volatile tt_l1_ptr uint32_t* b = stg + 16 * k;
                const uint8_t f = flag[k];
                if (f == 0) {
                    for (uint32_t j = 0; j < 16; ++j) v[j * 1024 + k] = b[j];
                } else if (f == 1) {
                    for (uint32_t j = 0; j < 16; j += 2) {
                        v[j * 1024 + k] = b[j];
                        v[(j + 1) * 1024 + k] = b[j + 1] ^ 0x80000000u;
                    }
                } else {
                    for (uint32_t j = 0; j < 16; ++j) v[j * 1024 + k] = 0;
                }
            }
            cb_push_back(cb_v, 16);
        }
    }
}
