#!/usr/bin/env python3
"""split.py <run dir or prof files...>: where a RELION rank's CPU per particle goes.

Reads the stage tables the profiler writes (TT_RELION_PROFILE_FILE, one per rank), sums them over ranks
and prints CPU milliseconds per particle by stage. The leader's table has no particles and is skipped.
`--json` prints one line for a results file.
"""
import glob, json, os, sys

# stage -> regions; a region nested in another is listed under its parent and not added to the stage total
STAGES = [
    ("fft_ctf", "getFourierTransformsAndCtfs"),
    ("diff2_coarse", "getAllSquaredDifferencesCoarse"),
    ("weights_coarse", "convertAllSquaredDifferencesToWeightsCoarse"),
    ("diff2_fine", "getAllSquaredDifferencesFine"),
    ("weights_fine", "convertAllSquaredDifferencesToWeightsFine"),
    ("sums", "storeWeightedSums"),
]
# weightPass also holds work RELION does not bracket (the dense weight array is allocated and filled per particle)
IN_WEIGHTPASS = ["getAllSquaredDifferencesCoarse", "convertAllSquaredDifferencesToWeightsCoarse", "generateProjectionSetup",
                 "getAllSquaredDifferencesFine", "convertAllSquaredDifferencesToWeightsFine"]
DETAIL = {
    "fft_ctf": ["nonZeroProb", "readData", "makeNoiseMask", "TranslateAndNormCorrect", "normalizeAndTransform_recImg",
                "applyMask", "normalizeAndTransform", "powerClass", "CTFRead2D", "selfApplyBeamTilt"],
    "diff2_coarse": ["generateProjectionSetupCoarse", "translation_1"],
    "diff2_fine": ["precalculateShiftedImagesCtfsAndInvSigma2s", "translation_2", "pair_list_1", "generateEulerMatrices", "Diff2CALL"],
    "sums": ["store_init", "collect_data_2", "translation_3", "generateEulerMatricesProjector", "pre_wavg_map",
             "wavg", "backproject", "store_post_gpu"],
}


# work done for the particles outside oneParticle, added to its stage and to the total: the last batch of the
# deferred splat (relion5-sums-defer) is applied when the volume is read, after the last particle
OUTSIDE = {"sums": ["splat_flush_tail"]}


def read(paths):
    cpu, wall, calls, ranks = {}, {}, {}, 0
    for p in paths:
        rows = {}
        for line in open(p, errors="replace"):
            f = line.split()
            if len(f) == 6 and f[0] == "TTPROF" and f[1] != "region":
                try:
                    rows[f[1]] = (float(f[2]), float(f[3]), int(f[4]))
                except ValueError:
                    pass
        if rows.get("oneParticle", (0, 0, 0))[2] == 0:
            continue
        ranks += 1
        for k, (w, c, n) in rows.items():
            wall[k] = wall.get(k, 0.0) + w
            cpu[k] = cpu.get(k, 0.0) + c
            calls[k] = calls.get(k, 0) + n
    return cpu, wall, calls, ranks


def main():
    args = [a for a in sys.argv[1:] if a != "--json"]
    paths = []
    for a in args:
        paths += sorted(glob.glob(os.path.join(a, "prof.*"))) if os.path.isdir(a) else [a]
    paths = [p for p in paths if not p.endswith(".tmp")]
    cpu, wall, calls, ranks = read(paths)
    n = calls.get("oneParticle", 0)
    if not n:
        sys.exit("no particles in " + " ".join(paths))
    ms = lambda k: 1e3 * cpu.get(k, 0.0) / n
    outside = sum(ms(r) for rs in OUTSIDE.values() for r in rs)
    out = {"ranks": ranks, "particles": n, "one_particle_cpu_ms": round(ms("oneParticle") + outside, 3),
           "one_particle_wall_ms": round(1e3 * wall["oneParticle"] / n, 3), "stages": {}, "detail": {}}
    for stage, region in STAGES:
        out["stages"][stage] = round(ms(region) + sum(ms(r) for r in OUTSIDE.get(stage, [])), 3)
        out["detail"][stage] = {r: round(ms(r), 3) for r in DETAIL.get(stage, []) + OUTSIDE.get(stage, []) if calls.get(r)}
    out["stages"]["weightpass_rest"] = round(ms("weightPass") - sum(ms(r) for r in IN_WEIGHTPASS), 3)
    out["stages"]["other"] = round(ms("oneParticle") + outside - sum(out["stages"].values()), 3)
    # what stays on the rank when both diff2 kernels run on the card
    out["own_cpu_ms"] = round(sum(v for k, v in out["stages"].items() if not k.startswith("diff2")), 3)
    if "--json" in sys.argv:
        print(json.dumps(out))
        return
    print(f"{ranks} ranks, {n} particles, {out['one_particle_cpu_ms']} ms CPU per particle "
          f"({out['one_particle_wall_ms']} ms thread wall); without the two diff2 stages {out['own_cpu_ms']} ms")
    for stage, v in out["stages"].items():
        print(f"  {stage:<16}{v:>10.3f} ms  {100 * v / out['one_particle_cpu_ms']:5.1f} %")
        for r, d in out["detail"].get(stage, {}).items():
            print(f"      {r:<44}{d:>9.3f}")


if __name__ == "__main__":
    main()
