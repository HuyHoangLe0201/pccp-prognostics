"""Critical-region behaviour of split CP versus CQR (paired seeds of Experiment D), quoted in Sections 5.3 and 5.7.

Subgroups by the PREDICTED remaining life tau, as in Table 4 of the manuscript: the point prediction of split CP, and the midpoint
of the quantile band for CQR. For every subset and threshold the script reports the mean over seeds of the subgroup PICP and MPIW
before and after projection onto [0, Rmax] (an empty intersection is a miss of zero width), and the share of intervals with L < 0.
On FD002/FD004 the midpoint of the quantile band never falls below 50 cycles, so the CQR subgroups are empty there.

usage: python crit_cqr.py OUT_EXPA OUT_EXPD GENDIR            (default: out/expA out/expD .)
writes GENDIR/numbers_crit.json
"""
import glob
import json
import os
import sys

import numpy as np

from common import ALPHA, conformal_quantile

OUT_A = sys.argv[1] if len(sys.argv) > 1 else "out/expA"
OUT_D = sys.argv[2] if len(sys.argv) > 2 else "out/expD"
GEN = sys.argv[3] if len(sys.argv) > 3 else "."
os.makedirs(GEN, exist_ok=True)
CAP = 125.0
TAUS = (50, 30, 15)


def proj_cov_w(L, U, y):
    Lp, Up = np.maximum(L, 0.0), np.minimum(U, CAP)
    return (y >= Lp) & (y <= Up), np.maximum(Up - Lp, 0.0)


result = {}
for sub in ("FD001", "FD003", "FD002", "FD004"):
    rows = {t: {"cp": [], "cqr": []} for t in TAUS}
    for f in sorted(glob.glob(f"{OUT_D}/expD_{sub}_seed*_arrays.npz")):
        seed = int(f.split("seed")[1].split("_")[0])
        d = np.load(f)
        a = np.load(f"{OUT_A}/expA_{sub}_seed{seed}_arrays.npz")
        assert np.allclose(a["y_te"], d["y"]), "row order differs"
        y = d["y"].astype(float)
        fte, fca, yca = a["f_te"].astype(float), a["f_ca"].astype(float), a["y_ca"].astype(float)
        q = conformal_quantile(np.abs(yca - fca), ALPHA)
        Lcp, Ucp = fte - q, fte + q
        Lq, Uq = d["L_cqr"].astype(float), d["U_cqr"].astype(float)
        mid = 0.5 * (Lq + Uq)
        for t in TAUS:
            for name, (L, U, pred) in {"cp": (Lcp, Ucp, fte), "cqr": (Lq, Uq, mid)}.items():
                m = pred <= t
                if m.sum() < 5:
                    continue
                covp, wp = proj_cov_w(L, U, y)
                rows[t][name].append((m.sum(), ((y >= L) & (y <= U))[m].mean(), np.maximum(U - L, 0)[m].mean(), covp[m].mean(), wp[m].mean(), (L[m] < 0).mean()))
    result[sub] = {}
    print(sub)
    for t in TAUS:
        for name in ("cp", "cqr"):
            r = np.array(rows[t][name])
            if len(r) == 0:
                continue
            result[sub][f"tau{t}_{name}"] = {"n_mean": float(r[:, 0].mean()), "picp": float(r[:, 1].mean()), "picp_proj": float(r[:, 3].mean()), "mpiw": float(r[:, 2].mean()),
                                              "mpiw_proj": float(r[:, 4].mean()), "share_L_neg": float(r[:, 5].mean()), "n_seeds": int(len(r))}
            x = result[sub][f"tau{t}_{name}"]
            print(f"  tau<={t:3d} {name:3s}  n~{x['n_mean']:6.0f}  PICP {x['picp']:.3f} -> proj {x['picp_proj']:.3f}   MPIW {x['mpiw']:6.1f} -> {x['mpiw_proj']:6.1f} "
                  f"({100 * (1 - x['mpiw_proj'] / x['mpiw']):4.1f}%)   share L<0 {100 * x['share_L_neg']:4.1f}%")
json.dump(result, open(f"{GEN}/numbers_crit.json", "w"), indent=1)
