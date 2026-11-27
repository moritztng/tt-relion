"""Full-job E-step diff2 work from RELION's own log (no per-iteration stars kept on the GPU side).

Per E-step: NrOrientations/NrTranslations of both adaptive passes and 'local searches' from the log; current size
from the logged CurrentResolution with RELION's rule size = 2*(ROUND(ori*apix/res) + incr_size 10) (the +25 %-box step
only applies while the FSC is high at the limit, early, where the fine pass is ~0 of the work), all data at ori_size in
the final pass; coarse size by RELION's formula. Significant samples per particle per phase from the dev cell's
data.star means (passed in as --nsig global_coarse,global_fine,local,final).
"""
import math, re, sys
ori, apix, diam, incr = 360, 1.34, 360.0, 10
log = open(sys.argv[1], errors="replace").read().replace("\r", "\n")
npart = int(sys.argv[2])
nsig = dict(zip(("gc", "gf", "loc", "fin"), map(float, sys.argv[3].split(","))))

def half_pixels(size):
    r = size // 2 + 1; cnt = 0
    for i in range(size):
        ip = i if i < (size + 1) // 2 else i - size
        for jp in range(size // 2 + 1):
            if round(math.sqrt(ip * ip + jp * jp)) < r and not (jp == 0 and ip < 0): cnt += 1
    return cnt
pc = {}
blocks = re.split(r"Expectation iteration (\d+)", log)
pre = blocks[0]; rows = []
nblk = (len(blocks) - 1) // 2
for bi, k in enumerate(range(1, len(blocks), 2)):
    it = int(blocks[k]); head = pre; pre = blocks[k + 1]
    res = float(re.findall(r"CurrentResolution= ([\d.]+)", head)[-1])
    o0 = re.findall(r"Oversampling= 0 NrHiddenVariableSamplingPoints= \d+\s+OrientationalSampling= ([\d.]+) NrOrientations= (\d+)\s+TranslationalSampling= ([\d.]+) NrTranslations= (\d+)", head)[-1]
    o1 = re.findall(r"Oversampling= 1 NrHiddenVariableSamplingPoints= \d+\s+OrientationalSampling= ([\d.]+) NrOrientations= (\d+)\s+TranslationalSampling= ([\d.]+) NrTranslations= (\d+)", head)[-1]
    loc = re.findall(r"local searches= (\w+)", head); is_local = bool(loc) and loc[-1] == "true"
    final = bi == nblk - 1
    cur = ori if final else min(ori, 2 * (round(ori * apix / res) + incr))
    ang0 = float(o0[0]); Oc, Tc, Of, Tf = int(o0[1]), int(o0[3]), int(o1[1]), int(o1[3])
    cs = min(cur, 2 * math.ceil(apix * ori / ((ang0 / 360) * math.pi * diam / 1.2)))
    for s in (cs, cur):
        if s not in pc: pc[s] = half_pixels(s)
    ph = "fin" if final else ("loc" if is_local else ("gf" if Oc > 4608 else "gc"))
    fac = (Of // Oc) * (Tf // Tc)
    coarse = 2.0 * Oc * Tc * 2 * pc[cs] * npart
    fine = 2.0 * nsig[ph] * fac * 2 * pc[cur] * npart
    rows.append((it, ph, res, Oc, Tc, cs, pc[cs], cur, pc[cur], coarse, fine))
print("it\tphase\tres_A\tOc\tTc\tcoarse_size\tPc\tcur_size\tPf\tcoarse_flop\tfine_flop")
for r in rows: print("\t".join(f"{x:.4g}" if isinstance(x, float) else str(x) for x in r))
tc = sum(r[9] for r in rows); tf = sum(r[10] for r in rows)
print(f"# total coarse {tc:.4g} fine {tf:.4g} all {tc+tf:.4g} FLOP ({npart} particles, {len(rows)} E-steps)")
