// proj_lerp reader: per unit of 1024 projected pixels, one 64-byte read per pixel of the 2x2x2 corner brick
// (8 corners x re, im in fp32) into a staging buffer, then a scalar scatter so corner value j of pixel k sits at
// element k of tile j (16 tiles). Pixel k <-> physical element k of every tile: the order is the caller's.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr auto tab_args = TensorAccessorArgs<0>();
    constexpr auto idx_args = TensorAccessorArgs<tab_args.next_compile_time_args_offset()>();
    constexpr auto frc_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_v = 0, cb_f = 1, cb_idx = 2, cb_stage = 3;
    constexpr uint32_t PAGE = 4096, BRICK = 64;

    const auto s_tab = TensorAccessor(tab_args, get_common_arg_val<uint32_t>(0), BRICK);
    const auto s_idx = TensorAccessor(idx_args, get_common_arg_val<uint32_t>(1), PAGE);
    const auto s_frc = TensorAccessor(frc_args, get_common_arg_val<uint32_t>(2), PAGE);
    const uint32_t u0 = get_arg_val<uint32_t>(0);
    const uint32_t nunits = get_arg_val<uint32_t>(1);

    const uint32_t idx_l1 = get_write_ptr(cb_idx);
    const uint32_t stg_l1 = get_write_ptr(cb_stage);
    volatile tt_l1_ptr uint32_t* idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);
    volatile tt_l1_ptr uint32_t* stg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stg_l1);

    for (uint32_t u = u0; u < u0 + nunits; ++u) {
        noc_async_read_page(u, s_idx, idx_l1);
        cb_reserve_back(cb_f, 3);
        uint32_t fp = get_write_ptr(cb_f);
        for (uint32_t t = 0; t < 3; ++t, fp += PAGE) noc_async_read_page(3 * u + t, s_frc, fp);
        noc_async_read_barrier();
        uint32_t sp = stg_l1;
        for (uint32_t k = 0; k < 1024; ++k, sp += BRICK) noc_async_read(s_tab.get_noc_addr(idx[k]), sp, BRICK);
        noc_async_read_barrier();
        cb_push_back(cb_f, 3);
        cb_reserve_back(cb_v, 16);
        volatile tt_l1_ptr uint32_t* v = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_v));
        for (uint32_t k = 0; k < 1024; ++k) {
            volatile tt_l1_ptr uint32_t* b = stg + 16 * k;
            for (uint32_t j = 0; j < 16; ++j) v[j * 1024 + k] = b[j];
        }
        cb_push_back(cb_v, 16);
    }
}
