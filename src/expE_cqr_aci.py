"""Experiment E: does the projection still help ACI when the base construction is CQR (adaptive and asymmetric)?

Does PCCP keep a meaningful gain when the base interval is already adaptive, asymmetric and reasonably tight?
Experiment D answers this for static intervals (Table 9). Here the question is asked in the setting of the paper's
central result (composition with ACI under shift), with the deployment protocol of Experiment C (Section 4.7):

  * source model: FD001-trained single-cycle MLP with two quantile outputs (5% and 95%, pinball loss; Section 4.3 architecture),
    85% of the FD001 engines for training and the remaining 15% for calibration (as in Experiment C);
  * streams A of FD004 (severe shift) and FD003 (mild shift); the quantile outputs are NOT clipped: unlike a point prediction, a clipped
    quantile band with a negative conformal correction would put the upper endpoint of every cap-valued row below Rmax and miss the
    label Y = Rmax (an artefact of the clipping, not of CQR), and clipping is what the projection does, on the final interval;
  * CQR scores s = max(q_lo - y, y - q_hi); ACI with the level clipped to the range of the trailing window of 500 revealed scores
    (exactly as in Experiment C), gamma = 0.03, target 0.90; the miss indicator is that of the projected interval;
  * arms: fixed CQR, fixed CQR projected onto [0, inf) and onto [0, Rmax], CQR+ACI, and the two projected CQR+ACI variants;
  * feedback protocols: oracle, fixed delay of 50 steps, end-of-life release.

usage: python expE_cqr_aci.py OUTDIR WORKERS N_SEEDS
"""
import json
import os
import sys
import time
from bisect import bisect_left, insort
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from expC_shift import (ALPHA, GAMMA, KEEP, WINDOW, _high_priority, arm_metrics, proj, read, target_stream)

CAP = 125.0
DELAY = 50


def source_quantile_model(seed):
    from expD_backbones import predict_np, train_quantile
    fd1 = read("train_FD001.txt")
    life = fd1.groupby("unit")["cycle"].transform("max")
    y = np.minimum((life - fd1["cycle"]).to_numpy(float), CAP)
    mn, mx = fd1[KEEP].min(), fd1[KEEP].max()
    X = ((fd1[KEEP] - mn) / (mx - mn).replace(0, 1)).to_numpy(np.float64)
    units = np.unique(fd1["unit"].to_numpy())
    rng = np.random.RandomState(seed)
    rng.shuffle(units)
    fit = np.isin(fd1["unit"].to_numpy(), units[:int(0.85 * len(units))])
    net = train_quantile(X[fit], y[fit], seed + 1000)
    q = predict_np(net, X[~fit])
    lo, hi = np.minimum(q[:, 0], q[:, 1]), np.maximum(q[:, 0], q[:, 1])
    scores = np.maximum(lo - y[~fit], y[~fit] - hi)
    n = len(scores)
    Q0 = float(np.sort(scores)[min(int(np.ceil((n + 1) * (1 - ALPHA))), n) - 1])
    return net, scores, Q0


def run_cqr_aci(lo, hi, y, unit, seed_scores, gamma=GAMMA, delay=0, eol=False, alpha=ALPHA):
    """ACI around the quantile band [lo, hi] with signed CQR scores; feedback as in expC_shift.run_aci."""
    n = len(lo)
    win = deque(seed_scores[-WINDOW:])
    srt = sorted(win)
    a = alpha
    L, U, A = np.empty(n), np.empty(n), np.empty(n)
    pending, buf, n_sat, mismatch = deque(), {}, 0, 0

    def push(s):
        win.append(s)
        insort(srt, s)
        if len(win) > WINDOW:
            srt.pop(bisect_left(srt, win.popleft()))

    for t in range(n):
        m = len(srt)
        k = int(np.ceil((m + 1) * (1 - a)))
        n_sat += k > m
        q = srt[min(max(k, 1), m) - 1]
        L[t], U[t], A[t] = lo[t] - q, hi[t] + q, a
        s_t = max(lo[t] - y[t], y[t] - hi[t])
        miss = not (max(L[t], 0.0) <= y[t] <= min(U[t], CAP))
        mismatch += int(miss != (not (L[t] <= y[t] <= U[t])))          # Theorem 2(i): must stay 0
        if eol:
            buf.setdefault(unit[t], []).append((s_t, miss))
            if t == n - 1 or unit[t + 1] != unit[t]:
                for s_, m_ in buf.pop(unit[t]):
                    a += gamma * (alpha - float(m_))
                    push(s_)
        elif delay == 0:
            a += gamma * (alpha - float(miss))
            push(s_t)
        else:
            pending.append((s_t, miss))
            if len(pending) > delay:
                s_, m_ = pending.popleft()
                a += gamma * (alpha - float(m_))
                push(s_)
    return L, U, A, {"frac_level_saturated": n_sat / n, "alpha_min": float(A.min()), "alpha_max": float(A.max()),
                     "indicator_mismatches": int(mismatch)}


def run_seed(args):
    seed, outdir = args
    _high_priority()
    path = f"{outdir}/expE_seed{seed}.json"
    if os.path.exists(path):
        return path
    import torch
    torch.set_num_threads(1)
    from expD_backbones import predict_np
    t0 = time.time()
    net, cal_scores, Q0 = source_quantile_model(seed)
    res = {"seed": seed, "Q0": Q0}
    for fd in ("FD004", "FD003"):
        SA = target_stream(fd, "test")
        q = predict_np(net, SA["X"].astype(np.float64))
        lo, hi = np.minimum(q[:, 0], q[:, 1]), np.maximum(q[:, 0], q[:, 1])      # raw quantile band, not clipped
        y = np.minimum(SA["true_rul"], CAP)
        out = {"n": int(len(y)), "share_at_cap": float((y >= CAP).mean())}
        Lf, Uf = lo - Q0, hi + Q0
        out["fixed"] = {"CQR": arm_metrics(Lf, Uf, y, CAP)}
        for nm, hi_inf in (("PCCP-inf", True), ("PCCP-R", False)):
            out["fixed"][nm] = arm_metrics(*proj(Lf, Uf, CAP, hi_inf=hi_inf), y, CAP)
        for prot, kw in (("oracle", {}), (f"delay{DELAY}", {"delay": DELAY}), ("eol", {"eol": True})):
            L, U, A, dg = run_cqr_aci(lo, hi, y, SA["unit"], cal_scores, **kw)
            arms = {"ACI": arm_metrics(L, U, y, CAP)}
            for nm, hi_inf in (("PCCP-inf+ACI", True), ("PCCP-R+ACI", False)):
                Lp, Up = proj(L, U, CAP, hi_inf=hi_inf)
                arms[nm] = arm_metrics(Lp, Up, y, CAP)
                # width statistics restricted to the steps at which the projected interval is not empty: an empty intersection is a
                # miss of zero width and its removed base width would otherwise be counted as a width reduction (Section 5.9, "Empty intersections")
                ne = Lp <= Up
                arms[nm]["nonempty"] = {"share": float(ne.mean()), "mpiw_base": float(np.maximum(U - L, 0)[ne].mean()),
                                        "mpiw_proj": float(np.maximum(Up - Lp, 0)[ne].mean())}
            arms["diag"] = dg
            out[prot] = arms
        res[fd] = out
    res["seconds"] = time.time() - t0
    with open(path, "w") as fh:
        json.dump(res, fh)
    return path


if __name__ == "__main__":
    outdir, workers, n_seeds = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    os.makedirs(outdir, exist_ok=True)
    tasks = [(s, outdir) for s in range(n_seeds)]
    t0 = time.time()
    if workers <= 1:
        for t in tasks:
            print(run_seed(t), f"{time.time() - t0:.0f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for p in ex.map(run_seed, tasks):
                print(p, f"{time.time() - t0:.0f}s", flush=True)
    print("ALL DONE", flush=True)
