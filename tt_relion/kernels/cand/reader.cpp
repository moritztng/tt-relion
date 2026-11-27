// cand: candidate compaction of d2 on chip. Core j scans tiles u in [u0, u0 + nu) of D (fp32 TILE [RT*32, OT*32])
// and emits (flat index r * OC + c, value bits) for every valid element with D[r, c] < thr[r] (IEEE order, exact,
// via integer keys: the RISC-V cores have no FPU) into its own row of OUT (uint32 RM [ncores, 2 CAP]). CNT[j, 0]
// is the core's total count; a total above CAP means overflow (host falls back to a full readback).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

namespace {
inline uint32_t key(uint32_t x) { return (x & 0x80000000u) ? ~x : (x | 0x80000000u); }
}  // namespace

void kernel_main() {
    constexpr auto d_args = TensorAccessorArgs<0>();
    constexpr auto t_args = TensorAccessorArgs<d_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<t_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr uint32_t base = c_args.next_compile_time_args_offset();
    constexpr uint32_t OT = get_compile_time_arg_val(base + 0), CAP = get_compile_time_arg_val(base + 1);
    constexpr uint32_t OC = get_compile_time_arg_val(base + 2), RV = get_compile_time_arg_val(base + 3);
    constexpr uint32_t cb_d = 0, cb_t = 1, cb_o = 2, cb_c = 3, TB = 4096, SB = 256;   // SB pairs per flush
    const auto sd = TensorAccessor(d_args, get_common_arg_val<uint32_t>(0), TB);
    const auto st = TensorAccessor(t_args, get_common_arg_val<uint32_t>(1), 128);
    const auto so = TensorAccessor(o_args, get_common_arg_val<uint32_t>(2), CAP * 8);
    const auto sc = TensorAccessor(c_args, get_common_arg_val<uint32_t>(3), 64);
    const uint32_t u0 = get_arg_val<uint32_t>(0), nu = get_arg_val<uint32_t>(1), core = get_arg_val<uint32_t>(2);
    cb_reserve_back(cb_d, 1); cb_reserve_back(cb_t, 1); cb_reserve_back(cb_o, 1); cb_reserve_back(cb_c, 1);
    const uint32_t ld = get_write_ptr(cb_d), lt = get_write_ptr(cb_t), lo = get_write_ptr(cb_o), lc = get_write_ptr(cb_c);
    const uint32_t* p = reinterpret_cast<const uint32_t*>(ld);
    const uint32_t* th = reinterpret_cast<const uint32_t*>(lt);
    uint32_t* stg = reinterpret_cast<uint32_t*>(lo);
    uint32_t tk[32];
    uint32_t total = 0, ns = 0, last_rt = 0xffffffffu;
    auto flush = [&](uint32_t n) {
        if (n && total + n <= CAP) {
            noc_async_write(lo, so.get_noc_addr(core, total * 8), ((n * 8 + 63) / 64) * 64);
            noc_async_write_barrier();
        }
        total += n;
    };
    for (uint32_t u = u0; u < u0 + nu; ++u) {
        const uint32_t rt = u / OT, ct = u % OT;
        noc_async_read_page(u, sd, ld);
        if (rt != last_rt) noc_async_read_page(rt, st, lt);
        noc_async_read_barrier();
        if (rt != last_rt) {
            for (uint32_t r = 0; r < 32; ++r) tk[r] = key(th[r]);
            last_rt = rt;
        }
        for (uint32_t f = 0; f < 4; ++f) {
            const uint32_t r0 = rt * 32 + (f >> 1) * 16, c0 = ct * 32 + (f & 1) * 16;
            for (uint32_t i = 0; i < 256; ++i) {
                const uint32_t v = p[f * 256 + i];
                const uint32_t rr = (f >> 1) * 16 + (i >> 4);
                if (key(v) < tk[rr]) {
                    const uint32_t gr = r0 + (i >> 4), gc = c0 + (i & 15);
                    if (gr < RV && gc < OC) {
                        stg[2 * ns] = gr * OC + gc;
                        stg[2 * ns + 1] = v;
                        if (++ns == SB) { flush(ns); ns = 0; }
                    }
                }
            }
        }
    }
    flush(ns);
    uint32_t* cw = reinterpret_cast<uint32_t*>(lc);
    for (uint32_t i = 0; i < 16; ++i) cw[i] = 0;
    cw[0] = total;
    noc_async_write(lc, sc.get_noc_addr(core, 0), 64);
    noc_async_write_barrier();
}
