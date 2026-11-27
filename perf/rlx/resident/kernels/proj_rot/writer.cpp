// proj_rot writer: orientation o, chunk c -> pages 2 (o NC + c) (re) and + 1 (im) of the fp32 [2 O NC, 1024] output.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto o_args = TensorAccessorArgs<0>();
    constexpr uint32_t NC = get_compile_time_arg_val(o_args.next_compile_time_args_offset());
    constexpr uint32_t cb_out = 16, PAGE = 4096;
    const auto so = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0), PAGE);
    const uint32_t o0 = get_arg_val<uint32_t>(0);
    const uint32_t no = get_arg_val<uint32_t>(1);
    for (uint32_t u = o0 * NC; u < (o0 + no) * NC; ++u) {
        cb_wait_front(cb_out, 2);
        const uint32_t rd = get_read_ptr(cb_out);
        noc_async_write_page(2 * u, so, rd);
        noc_async_write_page(2 * u + 1, so, rd + PAGE);
        noc_async_writes_flushed();
        cb_pop_front(cb_out, 2);
    }
    noc_async_write_barrier();
}
