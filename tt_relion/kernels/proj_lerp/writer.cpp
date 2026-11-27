// proj_lerp writer: unit u's re, im tiles are pages 2u, 2u + 1 (4 KB each) of the fp32 output.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto o_args = TensorAccessorArgs<0>();
    constexpr uint32_t cb_out = 16, PAGE = 4096;
    const auto so = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0), PAGE);
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);
    for (uint32_t u = u0; u < u0 + nunits; ++u) {
        cb_wait_front(cb_out, 2);
        uint32_t rd = get_read_ptr(cb_out);
        noc_async_write_page(2 * u, so, rd);
        noc_async_write_page(2 * u + 1, so, rd + PAGE);
        noc_async_writes_flushed();
        cb_pop_front(cb_out, 2);
    }
    noc_async_write_barrier();
}
