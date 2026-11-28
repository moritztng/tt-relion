"""A recorded call stream, read back, gives the calls their own answers again (torch backend, CPU)."""
import glob
import os
import subprocess
import sys

import numpy as np

os.environ.setdefault("TT_RELION_BACKEND", "torch")
from tt_relion import record, relion  # noqa: E402
from test_batch import _case  # noqa: E402


def _calls(tmp):
    mdl, eul, tx, ty, ir, ii, corr, g, P = _case(seed=3, n=6)
    n, T, O = tx.shape[0], tx.shape[1], eul.shape[0]
    sc = (g["mdlX"], g["mdlY"], g["mdlZ"], g["mdlInitY"], g["mdlInitZ"], g["maxR"], g["maxR2_padded"],
          g["padding_factor"], g["imgX"], g["imgY"])
    coarse = record.wrap("C", relion.diff2_coarse, tmp)
    fine = record.wrap("F", relion.diff2_fine, tmp)
    rot = np.arange(0, O, 3, dtype=np.uint64).repeat(2)
    trn = np.tile(np.array([1, 4], np.uint64), len(rot) // 2)
    outs = []
    for p in range(n):
        mv = [memoryview(v) for v in (mdl, eul.ravel(), tx[p], ty[p], ir[p], ii[p], corr[p])]
        out = np.zeros(O * T, np.float32)
        assert coarse(*mv, memoryview(out), *sc, O, T, P)
        outs.append(out)
        out = np.zeros(len(rot), np.float32)
        assert fine(*mv, memoryview(rot), memoryview(trn), memoryview(out), *sc, 0.25, O, T, len(rot), P, len(rot))
        outs.append(out)
    return outs


def test_record_roundtrip(tmp_path):
    outs = _calls(str(tmp_path))
    models, euls, calls = record.load(glob.glob(str(tmp_path / "rec.*.pkl"))[0])
    assert len(models) == 1 and len(euls) == 1 and [r["k"] for r in calls] == ["C", "F"] * 6
    for r, ref in zip(calls, outs):
        args, out = record.args(models, euls, r)
        assert (relion.diff2_coarse if r["k"] == "C" else relion.diff2_fine)(*args)
        assert np.array_equal(out, ref)


def test_replay_matches_itself(tmp_path):
    _calls(str(tmp_path))
    rec = glob.glob(str(tmp_path / "rec.*.pkl"))[0]
    here = os.path.dirname(os.path.abspath(__file__))
    run = [sys.executable, os.path.join(here, "..", "perf", "rlx", "bridge", "replay.py"), "--rec", rec, "--think", "0",
           "--out", str(tmp_path / "r.jsonl")]
    env = dict(os.environ, PYTHONPATH=os.path.join(here, ".."), TT_RELION_BACKEND="torch")
    subprocess.run(run + ["--digest", str(tmp_path / "a.npz")], env=env, check=True, capture_output=True)
    env.update(TT_RELION_BATCH="4", TT_RELION_BATCH_WAIT_MS="50")
    p = subprocess.run(run + ["--threads", "3", "--digest", str(tmp_path / "b.npz"), "--ref", str(tmp_path / "a.npz")],
                       env=env, check=True, capture_output=True, text=True)
    import json
    res = json.loads(p.stdout.strip().splitlines()[-1])
    assert res["calls"] == 12 and res["handled"] == 12 and res["threads"] == 3
    assert res["same_argmin"] == 12 and res["same_set"] == 12 and res["batches"] >= 1
