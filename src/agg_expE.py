r"""Aggregate Experiment E (CQR-based ACI under the FD001 -> FD004 / FD003 shifts) into a LaTeX table and a numbers file.

usage: python agg_expE.py OUT_EXPE GENDIR
writes GENDIR/tab_cqraci.tex and GENDIR/numbers_E.json
"""
import glob
import json
import os
import sys

import numpy as np

OUT, GEN = sys.argv[1], sys.argv[2]
os.makedirs(GEN, exist_ok=True)
runs = [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/expE_seed*.json"))]
print("seeds:", [r["seed"] for r in runs])
TARGETS = ["FD004", "FD003"]
PROTOS = [("oracle", "Oracle"), ("delay50", "Delay $d=50$"), ("eol", "End of life")]
ARMS = [("ACI", "CQR$+$ACI"), ("PCCP-inf+ACI", "\\quad projected onto $[0,\\infty)$"), ("PCCP-R+ACI", "\\quad projected onto $[0,\\Rmax]$")]


def sd(v):
    return float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")


def stat(fd, proto, arm, key):
    v = [r[fd][proto][arm][key] for r in runs]
    return float(np.mean(v)), sd(v)


N = {"n_seeds": len(runs)}
for fd in TARGETS:
    N[fd] = {"fixed": {a: {k: stat(fd, "fixed", a, k)[0] for k in ("picp", "mpiw", "feas", "iscore")} for a in ("CQR", "PCCP-inf", "PCCP-R")}}
    for proto, _ in PROTOS:
        N[fd][proto] = {a: {k: stat(fd, proto, a, k)[0] for k in ("picp", "mpiw", "feas", "iscore")} for a, _ in ARMS}
        base = np.array([r[fd][proto]["ACI"]["mpiw"] for r in runs])
        for a, _ in ARMS[1:]:
            proj = np.array([r[fd][proto][a]["mpiw"] for r in runs])
            N[fd][proto][a]["reduction_vs_ACI_pct"] = float(np.mean(100 * (1 - proj / base)))
            N[fd][proto][a]["reduction_sd"] = sd(100 * (1 - proj / base))
            # the same reduction on the steps at which the projected interval is not empty (an empty intersection is a miss of zero width)
            ne = [r[fd][proto][a]["nonempty"] for r in runs]
            N[fd][proto][a]["reduction_nonempty_pct"] = float(np.mean([100 * (1 - x["mpiw_proj"] / x["mpiw_base"]) for x in ne]))
            N[fd][proto][a]["empty_share_pct"] = float(100 * (1 - np.mean([x["share"] for x in ne])))
        N[fd][proto]["diag"] = {k: float(np.mean([r[fd][proto]["diag"][k] for r in runs])) for k in ("frac_level_saturated", "alpha_min", "alpha_max")}
        N[fd][proto]["diag"]["indicator_mismatches_total"] = int(sum(r[fd][proto]["diag"]["indicator_mismatches"] for r in runs))
    N[fd]["n_steps"] = runs[0][fd]["n"]
json.dump(N, open(f"{GEN}/numbers_E.json", "w"), indent=1)

f3, f2, f1 = (lambda v: f"{v:.3f}"), (lambda v: f"{v:.2f}"), (lambda v: f"{v:.1f}")
lines = [r"\footnotesize\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
         r" & \multicolumn{4}{c}{FD001$\to$FD004} & \multicolumn{4}{c}{FD001$\to$FD003} \\", r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r" & PICP & MPIW & Red.\ (\%) & $\psi_{\mathrm{ok}}$ & PICP & MPIW & Red.\ (\%) & $\psi_{\mathrm{ok}}$ \\", r"\midrule"]
row = lambda name, vals: name + " & " + " & ".join(vals) + r" \\"
fx = []
for fd in TARGETS:
    m = N[fd]["fixed"]["CQR"]
    fx += [f3(m["picp"]), f1(m["mpiw"]), "--", f2(m["feas"])]
lines.append(row("Fixed CQR (no adaptation)", fx))
for i, (proto, pname) in enumerate(PROTOS):
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{9}{@{}l}{\emph{" + pname + r"}} \\")
    for arm, aname in ARMS:
        vals = []
        for fd in TARGETS:
            m = N[fd][proto][arm]
            vals += [f3(m["picp"]), f1(m["mpiw"]), "--" if arm == "ACI" else f1(m["reduction_vs_ACI_pct"]), f2(m["feas"])]
        lines.append(row(aname, vals))
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{GEN}/tab_cqraci.tex", "w", encoding="utf-8").write("\n".join(lines) + "\n")

for fd in TARGETS:
    print(fd, "fixed CQR:", {k: round(v, 3) for k, v in N[fd]["fixed"]["CQR"].items()})
    for proto, _ in PROTOS:
        a, r0, rR = N[fd][proto]["ACI"], N[fd][proto]["PCCP-inf+ACI"], N[fd][proto]["PCCP-R+ACI"]
        print(f"  {proto:8s} ACI picp {a['picp']:.3f} mpiw {a['mpiw']:.1f} feas {a['feas']:.2f} | inf: {r0['mpiw']:.1f} ({r0['reduction_vs_ACI_pct']:.1f}%) | R: {rR['mpiw']:.1f} "
              f"({rR['reduction_vs_ACI_pct']:.1f}%, sd {rR['reduction_sd']:.1f}) feas {rR['feas']:.2f} | sat {N[fd][proto]['diag']['frac_level_saturated']:.2f} "
              f"| mismatches {N[fd][proto]['diag']['indicator_mismatches_total']}")
