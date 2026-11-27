// oz_cut reader: input tile u = (rt, kt) of X fp32 TILE [RT*32, KT*32], plus the exact row scale tile
// E[r, c] = 2^(1 - ea[r]) from the int32 row exponents (int32 RM [RT, 32]).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto a_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr uint32_t base = a_args.next_compile_time_args_offset();
    constexpr uint32_t KT = get_compile_time_arg_val(base + 0);
    constexpr uint32_t cb_x = 0, cb_e = 1, cb_s = 2, TB = 4096;
    const auto sx = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0), TB);
    const auto sa = TensorAccessor(a_args, get_common_arg_val<uint32_t>(1), 128);
    const uint32_t u0 = get_arg_val<uint32_t>(0), nu = get_arg_val<uint32_t>(1);
    cb_reserve_back(cb_s, 1);
    const uint32_t xa = get_write_ptr(cb_s);
    volatile tt_l1_ptr int32_t* ea = reinterpret_cast<volatile tt_l1_ptr int32_t*>(xa);
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        cb_reserve_back(cb_x, 1);
        noc_async_read_page(u, sx, get_write_ptr(cb_x));
        noc_async_read_page(u / KT, sa, xa);
        noc_async_read_barrier();
        cb_push_back(cb_x, 1);
        cb_reserve_back(cb_e, 1);
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_e));
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t v = static_cast<uint32_t>(128 - ea[r]) << 23;
            const uint32_t fr = (r >> 4) * 512 + (r & 15) * 16;
            for (uint32_t c = 0; c < 16; ++c) { p[fr + c] = v; p[fr + 256 + c] = v; }
        }
        cb_push_back(cb_e, 1);
    }
}
