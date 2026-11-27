// oz_combine reader: output tile u = (j, rt, ct) of [Q, RT, OT] (Q independent batches, batch-major as oz_fused writes
// them: S page (j * N + k) * RT * OT + v; row exponents ea page j * RT + rt, column exponents eb page j * OT + ct). Streams the n Ozaki sums S_k (fp32 TILE [n, RT*32, OT*32])
// for that tile, and builds the exact scale tile E[r, c] = 2^(ea[r] + eb[c] - 2n) from the int32 row/column exponents.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto s_args = TensorAccessorArgs<0>();
    constexpr auto a_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr uint32_t base = b_args.next_compile_time_args_offset();
    constexpr uint32_t N = get_compile_time_arg_val(base + 0);
    constexpr uint32_t RT = get_compile_time_arg_val(base + 1);
    constexpr uint32_t OT = get_compile_time_arg_val(base + 2);
    constexpr uint32_t cb_s = 0, cb_e = 2, cb_x = 3, TB = 4096;
    const auto ss = TensorAccessor(s_args, get_common_arg_val<uint32_t>(0), TB);
    const auto sa = TensorAccessor(a_args, get_common_arg_val<uint32_t>(1), 128);
    const auto sb = TensorAccessor(b_args, get_common_arg_val<uint32_t>(2), 128);
    const uint32_t u0 = get_arg_val<uint32_t>(0), nu = get_arg_val<uint32_t>(1);


    cb_reserve_back(cb_x, 2);
    const uint32_t xa = get_write_ptr(cb_x), xb = xa + 128;
    volatile tt_l1_ptr int32_t* ea = reinterpret_cast<volatile tt_l1_ptr int32_t*>(xa);
    volatile tt_l1_ptr int32_t* eb = reinterpret_cast<volatile tt_l1_ptr int32_t*>(xb);
    uint32_t last_rt = 0xffffffffu;
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t j = u / (RT * OT), v = u % (RT * OT);
        const uint32_t rt = j * RT + v / OT, ct = j * OT + v % OT;   // exponent pages of batch j
        for (uint32_t k = 0; k < N; ++k) {
            cb_reserve_back(cb_s, 1);
            noc_async_read_page((j * N + k) * RT * OT + v, ss, get_write_ptr(cb_s));
            noc_async_read_barrier();
            cb_push_back(cb_s, 1);
        }
        if (rt != last_rt) noc_async_read_page(rt, sa, xa);
        noc_async_read_page(ct, sb, xb);
        noc_async_read_barrier();
        last_rt = rt;
        cb_reserve_back(cb_e, 1);
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_e));
        for (uint32_t r = 0; r < 32; ++r) {
            const int32_t er = ea[r] + 127 - 2 * static_cast<int32_t>(N);   // Horner leaves a 4^n factor
            const uint32_t fr = (r >> 4) * 512 + (r & 15) * 16;
            for (uint32_t c = 0; c < 32; ++c)
                p[fr + (c >> 4) * 256 + (c & 15)] = static_cast<uint32_t>(er + eb[c]) << 23;
        }
        cb_push_back(cb_e, 1);
    }
}
