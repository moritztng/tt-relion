// oz_cut writer: digit planes of input tile u -> bf16 TILE [N, RT*32, KT*32], plane i page i*RT*KT + u.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto d_args = TensorAccessorArgs<0>();
    constexpr uint32_t base = d_args.next_compile_time_args_offset();
    constexpr uint32_t N = get_compile_time_arg_val(base + 0), NT = get_compile_time_arg_val(base + 1);
    constexpr uint32_t cb_o = 16, TB = 2048;
    const auto sd = TensorAccessor(d_args, get_common_arg_val<uint32_t>(0), TB);
    const uint32_t u0 = get_arg_val<uint32_t>(0), nu = get_arg_val<uint32_t>(1);
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        for (uint32_t i = 0; i < N; ++i) {
            cb_wait_front(cb_o, 1);
            noc_async_write_page(i * NT + u, sd, get_read_ptr(cb_o));
            noc_async_writes_flushed();
            cb_pop_front(cb_o, 1);
        }
    }
    noc_async_write_barrier();
}
