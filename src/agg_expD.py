r"""Aggregate Experiment D (PCCP on other backbones, single-cycle protocol) into a LaTeX table and a numbers file.

The split-CP row is recomputed from the arrays of Experiment A for exactly the same seeds/splits.

usage: python agg_expD.py OUT_EXPA OUT_EXPD GENDIR
writes GENDIR/tab_backbones.tex and GENDIR/numbers_D.json
"""
import glob
import json
import os
import sys

import numpy as np

from common import ALPHA, conformal_quantile
from expD_backbones import rows_for

OUT_A, OUT_D, GEN = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(GEN, exist_ok=True)
SUBS = ["FD001", "FD002", "FD003", "FD004"]
METHODS = [("cp", "Split CP"), ("lw", "Locally weighted CP"), ("cqr", "CQR"), ("qr", "QR (raw)"), ("mcd", "MC dropout (raw)")]


def cp_row(subset, seed):
    z = np.load(f"{OUT_A}/expA_{subset}_seed{seed}_arrays.npz")
    f_ca, y_ca, f_te, y_te = (z[k].astype(np.float64) for k in ("f_ca", "y_ca", "f_te", "y_te"))
    q = conformal_quantile(np.abs(y_ca - f_ca), ALPHA)
    return rows_for(f_te - q, f_te + q, y_te, z["ute"])


data = {s: [] for s in SUBS}
for s in SUBS:
    for f in sorted(glob.glob(f"{OUT_D}/expD_{s}_seed*.json")):
        r = json.load(open(f))
        r["cp"] = cp_row(s, r["seed"])
        data[s].append(r)
n_seeds = {s: len(v) for s, v in data.items()}
print("seeds:", n_seeds)


def sd(v):
    return float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")


def summ(rows):
    base = np.array([r["mpiw"] for r in rows])
    cap = np.array([r["Kcap"]["mpiw"] for r in rows])
    zero = np.array([r["K0inf"]["mpiw"] for r in rows])
    picp = np.array([r["picp"] for r in rows])
    red_cap, red_zero = 100 * (1 - cap / base), 100 * (1 - zero / base)
    loss = np.array([[r["picp"] - r["Kcap"]["picp"], r["picp"] - r["K0inf"]["picp"]] for r in rows])
    return {"n": len(rows), "picp": float(picp.mean()), "picp_sd": sd(picp), "mpiw": float(base.mean()),
            "mpiw_cap": float(cap.mean()), "mpiw_zero": float(zero.mean()),
            "red_cap": float(red_cap.mean()), "red_cap_sd": sd(red_cap), "red_cap_min": float(red_cap.min()), "red_cap_max": float(red_cap.max()),
            "red_zero": float(red_zero.mean()), "red_zero_sd": sd(red_zero),
            "psi_ok": float(100 * np.mean([r["psi_ok"] for r in rows])),
            "psi_plus": float(100 * np.mean([r["psi_plus"] for r in rows])),
            "psi_minus": float(100 * np.mean([r["psi_minus"] for r in rows])),
            "max_picp_loss": float(np.abs(loss).max()),
            "eng_p10": float(np.mean([r["engine"]["p10"] for r in rows])),
            "eng_frac_below_0.8": float(100 * np.mean([r["engine"]["frac_below_0.8"] for r in rows]))}


S = {s: {m: summ([r[m] for r in data[s]]) for m, _ in METHODS} for s in SUBS if data[s]}
for s in S:                                     # Lemma 1(i): the projected intervals cover exactly the same points
    for m, _ in METHODS:
        assert S[s][m]["max_picp_loss"] == 0.0, (s, m, S[s][m]["max_picp_loss"])
print("PICP(projected) == PICP(base) exactly for every backbone, subset and seed")
extra = {}
for s in S:
    extra[s] = {"lw_rmse": float(np.mean([r["lw"]["rmse"] for r in data[s]])), "mcd_rmse": float(np.mean([r["mcd"]["rmse"] for r in data[s]])),
                "cqr_Q": float(np.mean([r["cqr"]["Q"] for r in data[s]])), "lw_qhat": float(np.mean([r["lw"]["qhat"] for r in data[s]]))}
# pooled over subsets (mean of subset means)
POOL = {m: {k: float(np.mean([S[s][m][k] for s in S])) for k in ("red_cap", "red_zero", "picp", "mpiw", "mpiw_cap", "psi_ok")} for m, _ in METHODS}
json.dump({"n_seeds": n_seeds, "by_subset": S, "pooled": POOL, "extra": extra}, open(f"{GEN}/numbers_D.json", "w"), indent=1)

f2 = lambda v: f"{v:.2f}"
f1 = lambda v: f"{v:.1f}"
lines = [r"\footnotesize\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}llccccccc@{}}", r"\toprule",
         r"Subset & Backbone & PICP & MPIW & MPIW (PCCP) & \multicolumn{2}{c}{Reduction (\%) onto} & $\psi_{\mathrm{ok}}$ (\%) \\",
         r"\cmidrule(lr){6-7}", r" & & (SD) & & & $[0,\Rmax]$ (SD) & $[0,\infty)$ & \\", r"\midrule"]
for i, s in enumerate(S):
    for j, (m, name) in enumerate(METHODS):
        x = S[s][m]
        first = f"{s} ({x['n']} seeds)" if j == 0 else ""
        lines.append(f"{first} & {name} & {f2(x['picp'])} ({f2(x['picp_sd'])}) & {f1(x['mpiw'])} & {f1(x['mpiw_cap'])} & "
                     f"{f1(x['red_cap'])} ({f1(x['red_cap_sd'])}) & {f1(x['red_zero'])} & {f1(x['psi_ok'])}" + r" \\")
    if i < len(S) - 1:
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{GEN}/tab_backbones.tex", "w", encoding="utf-8").write("\n".join(lines) + "\n")

print("pooled over subsets (reduction onto [0,Rmax] / onto [0,inf) / PICP / MPIW -> MPIW(PCCP) / psi_ok):")
for m, name in METHODS:
    p = POOL[m]
    print(f"  {name:22s} {p['red_cap']:5.1f} / {p['red_zero']:5.1f} / {p['picp']:.3f} / {p['mpiw']:6.1f} -> {p['mpiw_cap']:6.1f} / {p['psi_ok']:5.1f}")
for s in S:
    print(s, {m: (round(S[s][m]["red_cap"], 1), round(S[s][m]["picp"], 3), round(S[s][m]["mpiw"], 1)) for m, _ in METHODS},
          "max |PICP loss|:", max(S[s][m]["max_picp_loss"] for m, _ in METHODS))
