r"""Generate the LaTeX table bodies and the numbers quoted in the text for Sections 5.2-5.4 from Experiment A.

writes  work/gen/tab_prov.tex, tab_cap.tex, tab_unit.tex, tab_hier.tex  and  work/gen/numbers_A.json
"""
import glob
import json
import os
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "out/expA"
GEN = sys.argv[2] if len(sys.argv) > 2 else "gen"
os.makedirs(GEN, exist_ok=True)
SUBS = ["FD001", "FD002", "FD003", "FD004"]
main, sens = {s: [] for s in SUBS}, {s: [] for s in SUBS}
for f in sorted(glob.glob(f"{OUT}/expA_FD00*_seed*_*.json")):
    r = json.load(open(f))
    (main if f.endswith("_main.json") else sens)[r["subset"]].append(r)
nm = {s: len(v) for s, v in main.items()}
ns = {s: len(v) for s, v in sens.items()}
print("seeds main", nm, "sens", ns)
N = {"seeds_main": nm, "seeds_sens": ns}


def rows_of(s, name):
    return [r[name] for r in (main[s] if name == "cap125" else sens[s]) if name in r]


def red(rows, key):
    if not rows or key not in rows[0]:
        return None
    base = np.mean([x["none"]["mpiw"] for x in rows])
    return 100 * (base - np.mean([x[key]["mpiw"] for x in rows])) / base


def viol(rows, key):
    return 100 * np.mean([x[key]["viol"] for x in rows]) if rows and key in rows[0] else None


def loss(rows, key):
    return 100 * np.mean([x["none"]["picp"] - x[key]["picp"] for x in rows]) if rows and key in rows[0] else None


def f1(v, d=1):
    return "--" if v is None else f"{v:.{d}f}"


# --------------------------------------------------------------------- tab_prov
lines = []
lines.append(r"\begin{tabular}{@{}lcccc@{}}")
lines.append(r"\toprule")
lines.append(" & " + " & ".join(SUBS) + r" \\")
lines.append(r"\midrule")
lines.append(r"\multicolumn{5}{@{}l}{\emph{Labels capped at $\Rmax=125$}} \\")
R125 = {s: rows_of(s, "cap125") for s in SUBS}
lines.append("MPIW of the base interval & " + " & ".join(f1(np.mean([x["none"]["mpiw"] for x in R125[s]]), 2) for s in SUBS) + r" \\")
spec = [("D: $\\Kc\\equiv[0,\\infty)$", "K0inf", False),
        ("D$+$L: $\\Kc\\equiv[0,125]$", "Kcap", False),
        ("D$+$E: $\\Kc(x)=[0,T_{1}-t]$", "Kage", True),
        ("D$+$E: $\\Kc(x)=[0,T_{0.95}-t]$", "Kage_q95", True),
        ("D$+$L$+$E: $[0,125]\\cap[0,T_{0.95}-t]$", "Kagecap_q95", False)]
for lab, key, show_v in spec:
    lines.append(lab + " & " + " & ".join(f1(red(R125[s], key)) for s in SUBS) + r" \\")
    if show_v:
        lines.append(r"\quad violation $\hat\delta$ (\%) & " + " & ".join(f1(viol(R125[s], key), 1) for s in SUBS) + r" \\")
        lines.append(r"\quad coverage loss (points) & " + " & ".join(f1(loss(R125[s], key), 2) for s in SUBS) + r" \\")
lines.append(r"\midrule")
lines.append(r"\multicolumn{5}{@{}l}{\emph{Uncapped labels $Y=T-t$}} \\")
RU = {s: rows_of(s, "uncapped") for s in SUBS}
lines.append("MPIW of the base interval & " + " & ".join(f1(np.mean([x["none"]["mpiw"] for x in RU[s]]), 1) for s in SUBS) + r" \\")
for lab, key, show_v in [("D: $\\Kc\\equiv[0,\\infty)$", "K0inf", False), ("D$+$E: $\\Kc(x)=[0,T_{1}-t]$", "Kage", True),
                         ("D$+$E: $\\Kc(x)=[0,T_{0.99}-t]$", "Kage_q99", True), ("D$+$E: $\\Kc(x)=[0,T_{0.95}-t]$", "Kage_q95", True),
                         ("D$+$E: $\\Kc(x)=[0,T_{0.90}-t]$", "Kage_q90", True)]:
    lines.append(lab + " & " + " & ".join(f1(red(RU[s], key)) for s in SUBS) + r" \\")
    if show_v:
        lines.append(r"\quad violation $\hat\delta$ (\%) & " + " & ".join(f1(viol(RU[s], key), 1) for s in SUBS) + r" \\")
        lines.append(r"\quad coverage loss (points) & " + " & ".join(f1(loss(RU[s], key), 2) for s in SUBS) + r" \\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
open(f"{GEN}/tab_prov.tex", "w", encoding="utf-8").write("\n".join(lines) + "\n")

# --------------------------------------------------------------------- tab_cap
lines = [r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
         " & " + " & ".join(r"\multicolumn{2}{c}{%s}" % s for s in SUBS) + r" \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
         r"$\Rmax$ & " + " & ".join("D & D$+$L" for _ in SUBS) + r" \\", r"\midrule"]
capnums = {}
for name, lab in (("cap100", "100"), ("cap125", "125"), ("cap150", "150"), ("uncapped", r"$\infty$ (uncapped)")):
    cells = []
    for s in SUBS:
        rows = rows_of(s, name)
        d, dl = red(rows, "K0inf"), red(rows, "Kcap")
        cells += [f1(d), f1(dl)]
        capnums[f"{name}_{s}"] = {"D": d, "DL": dl, "share_at_cap": (float(np.mean([x["share_at_cap"] for x in rows])) if rows else None),
                                   "rmse": (float(np.mean([x["rmse"] for x in rows])) if rows else None),
                                   "mpiw": (float(np.mean([x["none"]["mpiw"] for x in rows])) if rows else None), "n": len(rows)}
    lines.append(lab + " & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{GEN}/tab_cap.tex", "w", encoding="utf-8").write("\n".join(lines) + "\n")
N["cap"] = capnums

# --------------------------------------------------------------------- unit-level: table + bootstrap
rng = np.random.RandomState(2026)
unit_rows_a, unit_rows_b, boot = [], [], {}
for s in SUBS:
    rows = [r["cap125"] for r in main[s]]
    per_seed = [(np.array(x["engine"]["cp"]["n"], float), np.array(x["engine"]["cp"]["cov"]), np.array(x["engine"]["cp"]["w"]),
                 np.array(x["engine"]["pccp"]["w"])) for x in rows]
    pic, mwc, mwp, rdc = [], [], [], []
    for _ in range(1000):
        pc, wc, wp = [], [], []
        for n, cv, w1, w2 in per_seed:
            ix = rng.randint(0, len(n), len(n))          # the same engines are resampled for CP and PCCP
            pc.append(cv[ix].sum() / n[ix].sum())
            wc.append(w1[ix].sum() / n[ix].sum())
            wp.append(w2[ix].sum() / n[ix].sum())
        pic.append(np.mean(pc))
        mwc.append(np.mean(wc))
        mwp.append(np.mean(wp))
        rdc.append(100 * (1 - mwp[-1] / mwc[-1]))
    pc0 = float(np.mean([cv.sum() / n.sum() for n, cv, _, _ in per_seed]))
    ci = np.percentile(pic, [2.5, 97.5])
    ci_c, ci_w, ci_r = (np.percentile(v, [2.5, 97.5]) for v in (mwc, mwp, rdc))
    ec = np.concatenate([np.array(x["engine"]["cp"]["cov"]) / np.array(x["engine"]["cp"]["n"]) for x in rows])
    w_cp = float(np.mean([w1.sum() / n.sum() for n, _, w1, _ in per_seed]))
    w_p = float(np.mean([w2.sum() / n.sum() for n, _, _, w2 in per_seed]))
    boot[s] = {"picp": pc0, "picp_ci": ci.tolist(), "mpiw_cp": w_cp, "mpiw_pccp": w_p, "mpiw_pccp_ci": np.percentile(mwp, [2.5, 97.5]).tolist(),
               "engine_min": float(ec.min()), "engine_p10": float(np.percentile(ec, 10)), "engine_median": float(np.median(ec)),
               "frac_below_0.8": float((ec < 0.8).mean()), "n_engine_seeds": int(len(ec)),
               "reduction_pct": 100 * (w_cp - w_p) / w_cp}
    # CP width is constant within a seed, so an engine-level resampling interval exists only for PCCP; MPIW and the
    # reduction are therefore reported with the standard deviation over seeds
    wcs = np.array([w1.sum() / n.sum() for n, _, w1, _ in per_seed])
    wps = np.array([w2.sum() / n.sum() for n, _, _, w2 in per_seed])
    rds = 100 * (1 - wps / wcs)
    boot[s].update({"mpiw_cp_sd": float(wcs.std(ddof=1)), "mpiw_pccp_sd": float(wps.std(ddof=1)), "reduction_sd": float(rds.std(ddof=1)),
                    "reduction_ci_boot": ci_r.tolist()})
    pcs = np.array([cv.sum() / n.sum() for n, cv, _, _ in per_seed])
    boot[s]["picp_sd"] = float(pcs.std(ddof=1))
    unit_rows_a.append(f"{s} & {len(rows)} & {pc0:.3f} ({pcs.std(ddof=1):.3f}) [{ci[0]:.3f}, {ci[1]:.3f}] & {w_cp:.1f} ({wcs.std(ddof=1):.1f}) & "
                       f"{w_p:.1f} ({wps.std(ddof=1):.1f}) & {boot[s]['reduction_pct']:.1f} ({rds.std(ddof=1):.1f}) \\\\")
    unit_rows_b.append(f"{s} & {len(ec)} & {ec.min():.2f} & {np.percentile(ec, 10):.2f} & {np.median(ec):.2f} & {100 * (ec < 0.8).mean():.0f}\\% \\\\")
lines = [r"\footnotesize\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r"Subset & Seeds & PICP & MPIW (CP) & MPIW (PCCP) & Reduction (\%) \\", r"\midrule"] + unit_rows_a + [r"\bottomrule", r"\end{tabular}", "",
         r"\medskip", r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
         r"Subset & Engines & \multicolumn{4}{c}{Engine-level PICP} \\", r"\cmidrule(lr){3-6}",
         r" & & min & 10th pct. & median & $<0.8$ \\", r"\midrule"] + unit_rows_b + [r"\bottomrule", r"\end{tabular}"]
open(f"{GEN}/tab_unit.tex", "w", encoding="utf-8").write("\n".join(lines) + "\n")
N["unit"] = boot

# --------------------------------------------------------------------- stage-wise
stage = {}
for s in SUBS:
    rows = [r["cap125"] for r in main[s]]
    tr = np.sum([np.array(x["stage_rul"], float) for x in rows], axis=0)
    tl = np.sum([np.array(x["stage_life"], float) for x in rows], axis=0)
    stage[s] = {"rul_band_picp": [float(tr[i][1] / tr[i][0]) if tr[i][0] else None for i in range(7)],
                "rul_band_n": [int(tr[i][0]) for i in range(7)],
                "rul_band_w_cp": [float(tr[i][3] / tr[i][0]) if tr[i][0] else None for i in range(7)],
                "rul_band_w_pccp": [float(tr[i][4] / tr[i][0]) if tr[i][0] else None for i in range(7)],
                "life_picp": [float(tl[i][1] / tl[i][0]) for i in range(5)]}
N["stage"] = stage

# --------------------------------------------------------------------- engine-level calibration table
lines = [r"\footnotesize\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}llccccccc@{}}", r"\toprule",
         r"Subset & Calibration & $\hat q$ & PICP & Engine 10th pct. & Engine $<0.8$ & MPIW (CP) & MPIW (PCCP) & Recovered \\", r"\midrule"]
hier = {}
for s in SUBS:
    hs = [r["cap125"]["hier"] for r in main[s]]
    for k, lab in (("cycle_level", "Cycle level"), ("pooled_cdf", "CDF pooling"), ("subsample", "Repeated subsampling")):
        g = lambda key: float(np.mean([x[k][key] for x in hs]))
        rec = 100 * (g("mpiw") - g("mpiw_pccp")) / g("mpiw")
        hier[f"{s}_{k}"] = {key: g(key) for key in ("q", "picp", "picp_engine_mean", "engine_min", "engine_p10", "frac_below_0.8", "mpiw", "mpiw_pccp")}
        hier[f"{s}_{k}"]["recovered_pct"] = rec
        lines.append(f"{s if k == 'cycle_level' else ''} & {lab} & {g('q'):.1f} & {g('picp'):.3f} & {g('engine_p10'):.2f} & "
                     f"{100 * g('frac_below_0.8'):.0f}\\% & {g('mpiw'):.1f} & {g('mpiw_pccp'):.1f} & {rec:.1f}\\% \\\\")
    if s != SUBS[-1]:
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
open(f"{GEN}/tab_hier.tex", "w", encoding="utf-8").write("\n".join(lines) + "\n")
N["hier"] = hier
json.dump(N, open(f"{GEN}/numbers_A.json", "w"), indent=1)
print("written to", GEN)
