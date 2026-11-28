// oz_fused writer: when compute signals a unit done, write its RB x NB fp32 accumulator tiles to S[k] (TILE
// [n, RT, OT]) straight from the accumulator's L1, then hand the reader a token to zero it for the next unit.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto s_args = TensorAccessorArgs<0>();
    constexpr uint32_t base = s_args.next_compile_time_args_offset();
    constexpr uint32_t RB = get_compile_time_arg_val(base), NB = get_compile_time_arg_val(base + 1);
    constexpr uint32_t RT = get_compile_time_arg_val(base + 2), OT = get_compile_time_arg_val(base + 3);
    constexpr uint32_t NRB = get_compile_time_arg_val(base + 4), NCB = get_compile_time_arg_val(base + 5);
    constexpr uint32_t NB_K = get_compile_time_arg_val(base + 6);
    constexpr bool ZIG = get_compile_time_arg_val(base + 7);
    constexpr uint32_t cb_acc = 24, cb_done = 25, cb_tok = 26, TF = 4096;
    const auto ss = TensorAccessor(s_args, get_common_arg_val<uint32_t>(0), TF);
    const uint32_t u0 = get_arg_val<uint32_t>(0), nu = get_arg_val<uint32_t>(1);
    const uint32_t acc = get_read_ptr(cb_acc);   // the writer never moves cb_acc's pointers: always its base
    for (uint32_t v = u0; v < u0 + nu; ++v) {
        // the reader's unit order (see reader.cpp)
        const uint32_t zz = v % NB_K, kz = (zz & 1) ? NB_K - 1 - (zz >> 1) : (zz >> 1), blk = v / NB_K;
        const uint32_t u = ZIG ? ((blk / (NRB * NCB)) * NB_K + kz) * (NRB * NCB) + blk % (NRB * NCB) : v;
        const uint32_t qk = u / (NRB * NCB), rb = (u / NCB) % NRB, cbk = u % NCB;   // qk = q * n + k
        cb_wait_front(cb_done, 1);
        uint32_t p = acc;
        for (uint32_t r = 0; r < RB; ++r)
            for (uint32_t c = 0; c < NB; ++c, p += TF)
                noc_async_write_page(qk * RT * OT + (rb * RB + r) * OT + cbk * NB + c, ss, p);
        noc_async_write_barrier();
        cb_pop_front(cb_done, 1);
        if (v + 1 < u0 + nu) {
            cb_reserve_back(cb_tok, 1);
            cb_push_back(cb_tok, 1);
        }
    }
}
