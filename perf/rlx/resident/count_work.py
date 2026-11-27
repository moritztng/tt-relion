"""Count RELION E-step diff2 work per iteration from a finished auto-refine run (CPU reference).

Reads RELION's own log lines (NrOrientations/NrTranslations per adaptive pass), the model star current size,
data.star rlnNrOfSignificantSamples, and RELION's coarse-size formula (ml_optimiser.cpp updateImageSizeAndResolutionPointers).
Optional --local counts.tsv: per-iteration mean coarse orientations per particle for local searches (count_local).
Prints one TSV row per E-step: it, local, Oc, Tc, coarse_size, Pc, Of, Tf, cur_size, Pf, nsig, and FLOP for the
coarse cross term (2 * Oc*Tc * 2Pc per particle, real-matmul form) and the fine cross term over the significant set.
"""
import math, re, sys, glob, os
import numpy as np

run = sys.argv[1]; npart_override = None
local = {}
if "--local" in sys.argv:
    for l in open(sys.argv[sys.argv.index("--local") + 1]):
        it, oc = l.split()[:2]; local[int(it)] = float(oc)

def star_val(fn, key):
    for l in open(fn):
        if l.startswith(key + " ") or l.startswith(key + "\t"):
            return l.split()[1]
    return None

def star_col(fn, key):
    lines = open(fn).read().split("\n"); cols = []; out = []; in_loop = False; blk = None
    for l in lines:
        s = l.strip()
        if s.startswith("data_"): blk = s; cols = []; in_loop = False; continue
        if blk != "data_particles": continue
        if s == "loop_": in_loop = True; cols = []; continue
        if in_loop and s.startswith("_rln"): cols.append(s.split()[0]); continue
        if in_loop and s and cols and not s.startswith("#"):
            out.append(s.split()[cols.index(key)])
    return np.array(out, dtype=float)

def half_pixels(size):
    # RELION Mresol: elements of the FFTW half transform with ires < size/2+1, minus the redundant jp==0 && ip<0 half
    r = size // 2 + 1; n = 0
    for ip in range(-(size // 2) + (1 - size % 2) * 0, size - size // 2):
        pass
    cnt = 0
    for i in range(size):
        ip = i if i < (size + 1) // 2 else i - size
        for jp in range(size // 2 + 1):
            if round(math.sqrt(ip * ip + jp * jp)) < r and not (jp == 0 and ip < 0):
                cnt += 1
    return cnt

log = open(os.path.join(run, "refine.log"), errors="replace").read().replace("\r", "\n")
blocks = re.split(r"Expectation iteration (\d+)", log)
# blocks: [pre, it1, body1, it2, body2, ...]; the sampling lines for E-step k are at the END of the text before it
diam, apix, ori = 360.0, 1.34, 360
pix_cache = {}
rows = []
pre = blocks[0]
for k in range(1, len(blocks), 2):
    it = int(blocks[k]); head = pre
    pre = blocks[k + 1]
    o0 = re.findall(r"Oversampling= 0 NrHiddenVariableSamplingPoints= \d+\s+OrientationalSampling= ([\d.]+) NrOrientations= (\d+)\s+TranslationalSampling= ([\d.]+) NrTranslations= (\d+)", head)
    o1 = re.findall(r"Oversampling= 1 NrHiddenVariableSamplingPoints= \d+\s+OrientationalSampling= ([\d.]+) NrOrientations= (\d+)\s+TranslationalSampling= ([\d.]+) NrTranslations= (\d+)", head)
    loc = re.findall(r"local searches= (\w+)", head)
    ang0, Oc, _, Tc = o0[-1]; _, Of, _, Tf = o1[-1]
    ang0 = float(ang0); Oc = int(Oc); Of = int(Of); Tc = int(Tc); Tf = int(Tf)
    is_local = bool(loc) and loc[-1] == "true"
    prev = it - 1
    mfn = os.path.join(run, f"run_it{prev:03d}_half1_model.star")
    cur = int(star_val(mfn, "_rlnCurrentImageSize"))
    dfn = os.path.join(run, f"run_it{it:03d}_data.star")
    if not os.path.exists(dfn):
        dfn = os.path.join(run, "run_data.star")  # final iteration writes no itNNN data
    nsig = star_col(dfn, "_rlnNrOfSignificantSamples")
    npart = len(nsig)
    coarse_res = (ang0 / 360.0) * math.pi * diam / 1.2
    cs = min(ori, 2 * math.ceil(apix * ori / coarse_res)); cs = min(cs, cur)
    for s in (cs, cur):
        if s not in pix_cache: pix_cache[s] = half_pixels(s)
    Pc, Pf = pix_cache[cs], pix_cache[cur]
    oc_pp = local.get(it, Oc) if is_local else Oc
    fac = (Of // Oc) * (Tf // Tc)
    coarse = 2.0 * oc_pp * Tc * 2 * Pc * npart
    fine = 2.0 * nsig.mean() * fac * 2 * Pf * npart
    rows.append((it, int(is_local), oc_pp, Tc, cs, Pc, fac, cur, Pf, nsig.mean(), npart, coarse, fine))
print("it\tlocal\tOc_pp\tTc\tcoarse_size\tPc\tfine_fac\tcur_size\tPf\tnsig_mean\tnpart\tcoarse_flop\tfine_flop")
for r in rows:
    print("\t".join(f"{x:.4g}" if isinstance(x, float) else str(x) for x in r))
tc = sum(r[11] for r in rows); tf = sum(r[12] for r in rows)
print(f"# total coarse {tc:.4g} fine {tf:.4g} all {tc+tf:.4g} FLOP (dev cell, {rows[0][10]} particles, {len(rows)} E-steps)")
