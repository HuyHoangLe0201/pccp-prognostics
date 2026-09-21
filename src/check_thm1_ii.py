"""Computed checks of Theorem 1 (i) and (ii) on FD001 (30 seeds of Experiment A).

(i)  mean PICP of PCCP and FCP over seeds and its standard error (the nominal level is 0.90);
(ii) PCCP = FCP when every calibration and test prediction is feasible: calibration and test sets are restricted to the points with
     fhat in K = [0, 125] (so that ftilde = fhat), both intervals are rebuilt from scratch and compared endpoint by endpoint.
     Also reported: the same comparison on the full samples (where about 5% of the predictions are infeasible), which is Table 7's regime.

usage: python check_thm1_ii.py OUT_EXPA [GENDIR]        writes GENDIR/numbers_thm1.json
"""
import glob
import json
import os
import sys

import numpy as np

from common import ALPHA, conformal_quantile

OUT_A = sys.argv[1] if len(sys.argv) > 1 else "out/expA"
GEN = sys.argv[2] if len(sys.argv) > 2 else "."
os.makedirs(GEN, exist_ok=True)
LO, HI = 0.0, 125.0


def pccp(f_ca, y_ca, f_te):
    q = conformal_quantile(np.abs(y_ca - f_ca), ALPHA)
    return np.maximum(f_te - q, LO), np.minimum(f_te + q, HI), q


def fcp(f_ca, y_ca, f_te):
    ft_ca, ft_te = np.clip(f_ca, LO, HI), np.clip(f_te, LO, HI)
    q = conformal_quantile(np.abs(y_ca - ft_ca), ALPHA)
    return np.maximum(ft_te - q, LO), np.minimum(ft_te + q, HI), q


picp_p, picp_f, diff_feas, diff_full, n_feas_ca, n_feas_te, n_seeds = [], [], [], [], [], [], 0
for f in sorted(glob.glob(f"{OUT_A}/expA_FD001_seed*_arrays.npz")):
    z = np.load(f)
    f_ca, y_ca, f_te, y_te = (z[k].astype(np.float64) for k in ("f_ca", "y_ca", "f_te", "y_te"))
    Lp, Up, qp = pccp(f_ca, y_ca, f_te)
    Lf, Uf, qf = fcp(f_ca, y_ca, f_te)
    picp_p.append(np.mean((y_te >= Lp) & (y_te <= Up)))
    picp_f.append(np.mean((y_te >= Lf) & (y_te <= Uf)))
    diff_full.append(max(np.abs(Lp - Lf).max(), np.abs(Up - Uf).max()))
    mca, mte = (f_ca >= LO) & (f_ca <= HI), (f_te >= LO) & (f_te <= HI)
    n_feas_ca.append(int(mca.sum()))
    n_feas_te.append(int(mte.sum()))
    Lp2, Up2, qp2 = pccp(f_ca[mca], y_ca[mca], f_te[mte])
    Lf2, Uf2, qf2 = fcp(f_ca[mca], y_ca[mca], f_te[mte])
    diff_feas.append(max(np.abs(Lp2 - Lf2).max(), np.abs(Up2 - Uf2).max(), abs(qp2 - qf2)))
    n_seeds += 1

picp_p, picp_f = np.array(picp_p), np.array(picp_f)
out = {"n_seeds": n_seeds, "picp_pccp_mean": float(picp_p.mean()), "picp_fcp_mean": float(picp_f.mean()),
       "picp_pccp_sd": float(picp_p.std(ddof=1)), "picp_se": float(picp_p.std(ddof=1) / np.sqrt(n_seeds)),
       "max_diff_feasible_subset": float(max(diff_feas)), "max_diff_full_sample": float(max(diff_full)),
       "mean_feasible_cal_points": float(np.mean(n_feas_ca)), "mean_feasible_test_points": float(np.mean(n_feas_te))}
json.dump(out, open(f"{GEN}/numbers_thm1.json", "w"), indent=1)
print(json.dumps(out, indent=1))
