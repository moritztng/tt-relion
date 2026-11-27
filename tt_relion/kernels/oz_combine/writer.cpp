// oz_combine writer: hi and lo tiles of output tile u to two fp32 TILE [RT*32, OT*32] tensors.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto h_args = TensorAccessorArgs<0>();
    constexpr auto l_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_hi = 16, cb_lo = 17, TB = 4096;
    const auto sh = TensorAccessor(h_args, get_common_arg_val<uint32_t>(0), TB);
    const auto sl = TensorAccessor(l_args, get_common_arg_val<uint32_t>(1), TB);
    const uint32_t u0 = get_arg_val<uint32_t>(0), nu = get_arg_val<uint32_t>(1);
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        cb_wait_front(cb_hi, 1);
        cb_wait_front(cb_lo, 1);
        noc_async_write_page(u, sh, get_read_ptr(cb_hi));
        noc_async_write_page(u, sl, get_read_ptr(cb_lo));
        noc_async_writes_flushed();
        cb_pop_front(cb_hi, 1);
        cb_pop_front(cb_lo, 1);
    }
    noc_async_write_barrier();
}
