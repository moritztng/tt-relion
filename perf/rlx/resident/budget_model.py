"""Full-job E-step chip budget for the resident design, from MEASURED one-chip rates (chip 16, AICLK 1350) applied to
RELION's own per-iteration work (results/work_full_s0.tsv, 105,247 particles, 28 E-steps). A model, not a wall:
every rate below is a measured file; shapes are padded the way the kernels pad them.
  python budget_model.py [--chips 32]
"""
import json
import sys

CH = int(sys.argv[sys.argv.index("--chips") + 1]) if "--chips" in sys.argv else 32
NP = 105247
pad = lambda x, m: -(-x // m) * m  # noqa: E731
# measured (results/global_pass_cand.jsonl, runs/gp20): prod batch B256 T13 O4096 K2400 (Pc 1021) n13
PROD = dict(R=3328, O=4096, K=2400, oz=0.1721, proj=0.0124, op=0.0051, rb=0.0087, sig=0.0197)
# measured (results/oz_local_shape_bh_chip16.jsonl, runs/loc1): per particle [64, K] x [160, K] n13, q128
LOC_MS_PER_K2400 = 0.236
PROJ_PX_S = 390e6       # proj_lerp, results/proj_lerp_bh_chip16.jsonl (389-392M px/s)
GATHER_US_K8160 = 168.9  # bf16 digit-table row gather per particle (n13 x 160 rows), results/table_gather_digits_bh_chip16.jsonl
TABLE = "--table" in sys.argv   # local passes from a shared per-half digit table (final stays on proj_lerp: 788 GB/half)
FINE_S = 1030.0         # fine pass: 0.4e12 px of child projections at PROJ_PX_S + 1.9e13 FLOP (state doc), chip-s
rows = [l.split("\t") for l in open("results/work_full_s0.tsv") if l[0].isdigit()]
tot = {}
for r in rows:
    it, ph, Oc, Tc, Pc = int(r[0]), r[1], int(r[3]), int(r[4]), int(r[6])
    K = pad(2 * Pc, 480)
    if ph in ("gc", "gf"):   # global: B256 batches, shared projections, all-in per-batch time scaled by padded work
        R = 256 * Tc
        w = R * Oc * K / (PROD["R"] * PROD["O"] * PROD["K"])
        per = PROD["oz"] * w + (PROD["proj"] + PROD["rb"] + PROD["sig"]) * Oc / PROD["O"] + PROD["op"] * R * K / (PROD["R"] * PROD["K"])
        s = per * NP / 256
        tot[ph] = tot.get(ph, 0) + s
    else:                    # local / final: per-particle product (T padded to 64) + own-orientation projections
        s = LOC_MS_PER_K2400 * 1e-3 * K / 2400 * max(1, pad(Tc, 64) // 64) * NP
        tot[ph + "_mm"] = tot.get(ph + "_mm", 0) + s
        if TABLE and ph == "loc":   # gather per particle + building both halves' tables once per iteration
            s = GATHER_US_K8160 * 1e-6 * K / 8160 * NP + 2 * 294912 * Pc / PROJ_PX_S
        else:
            s = Oc * Pc * NP / PROJ_PX_S
        tot[ph + "_proj"] = tot.get(ph + "_proj", 0) + s
tot["fine"] = FINE_S
chip_s = sum(tot.values())
print(json.dumps(dict(table=TABLE, chips=CH, chip_s={k: round(v) for k, v in tot.items()}, total_chip_s=round(chip_s),
                      per_chip_s=round(chip_s / CH, 1),
                      per_chip_by_phase={k: round(v / CH, 1) for k, v in tot.items()})))
