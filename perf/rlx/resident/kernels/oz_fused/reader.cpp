// oz_fused reader. Unit u = (diagonal k, row block rb of RB tiles, column block cb of NB tiles). Zeroes the fp32
// accumulator block in L1 (after the writer has drained the previous unit), then streams, for every digit pair
// (i, k - i) and every 480-wide K run, the B_{k-i} run block [KR x NB tiles] and per row tile the A_i run [KR tiles].
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr uint32_t base = b_args.next_compile_time_args_offset();
    constexpr uint32_t RB = get_compile_time_arg_val(base), NB = get_compile_time_arg_val(base + 1);
    constexpr uint32_t KR = get_compile_time_arg_val(base + 2), NRUN = get_compile_time_arg_val(base + 3);
    constexpr uint32_t RT = get_compile_time_arg_val(base + 4), OT = get_compile_time_arg_val(base + 5);
    constexpr uint32_t NRB = get_compile_time_arg_val(base + 6), NCB = get_compile_time_arg_val(base + 7);
    constexpr uint32_t KT = KR * NRUN;
    constexpr uint32_t cb_a = 0, cb_b = 1, cb_acc = 24, cb_tok = 26;
    constexpr uint32_t TB = 2048, TF = 4096;
    const auto sa = TensorAccessor(a_args, get_common_arg_val<uint32_t>(0), TB);
    const auto sb = TensorAccessor(b_args, get_common_arg_val<uint32_t>(1), TB);
    const uint32_t u0 = get_arg_val<uint32_t>(0), nu = get_arg_val<uint32_t>(1);

    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t k = u / (NRB * NCB), rb = (u / NCB) % NRB, cbk = u % NCB;
        if (u != u0) {
            cb_wait_front(cb_tok, 1);
            cb_pop_front(cb_tok, 1);
        }
        cb_reserve_back(cb_acc, RB * NB);
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_acc));
        for (uint32_t w = 0; w < RB * NB * (TF / 4); ++w) z[w] = 0;
        cb_push_back(cb_acc, RB * NB);
        for (uint32_t i = 0; i <= k; ++i) {
            const uint32_t j = k - i;
            for (uint32_t run = 0; run < NRUN; ++run) {
                cb_reserve_back(cb_b, KR * NB);
                uint32_t p = get_write_ptr(cb_b);
                for (uint32_t kk = 0; kk < KR; ++kk)
                    for (uint32_t c = 0; c < NB; ++c, p += TB)
                        noc_async_read_page(j * KT * OT + (run * KR + kk) * OT + cbk * NB + c, sb, p);
                noc_async_read_barrier();
                cb_push_back(cb_b, KR * NB);
                for (uint32_t r = 0; r < RB; ++r) {
                    cb_reserve_back(cb_a, KR);
                    uint32_t q = get_write_ptr(cb_a);
                    for (uint32_t kk = 0; kk < KR; ++kk, q += TB)
                        noc_async_read_page(i * RT * KT + (rb * RB + r) * KT + run * KR + kk, sa, q);
                    noc_async_read_barrier();
                    cb_push_back(cb_a, KR);
                }
            }
        }
    }
}
