"""Aggregate Experiment A into the numbers quoted in Sections 5.1-5.4 (constraint variants, label convention,
engine-level coverage, stages, engine-level calibration)."""
import glob
import json
import os
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "out/expA"
SUBS = ["FD001", "FD002", "FD003", "FD004"]
main, sens = {s: [] for s in SUBS}, {s: [] for s in SUBS}
for f in sorted(glob.glob(f"{OUT}/expA_FD00*_seed*_*.json")):
    r = json.load(open(f))
    (main if f.endswith("_main.json") else sens)[r["subset"]].append(r)
print("seeds available (main):", {s: len(v) for s, v in main.items()}, "(sens):", {s: len(v) for s, v in sens.items()})


def mean_sd(v):
    v = np.asarray(v, float)
    return v.mean(), v.std()


# ------------------------------------------------------------------ Table: constraint variants, cap 125
print("\n=== K variants, Rmax = 125 (mean over seeds)")
print("%-6s %4s %8s | %-28s" % ("subset", "n", "MPIW(CP)", "reduction % (K=[0,inf) | [0,R] | age_max | age_max&R | age_q99&R | q95&R | q90&R)"))
for s in SUBS:
    rows = [r["cap125"] for r in main[s]]
    if not rows:
        continue
    base = np.mean([x["none"]["mpiw"] for x in rows])
    red = lambda k: 100 * (base - np.mean([x[k]["mpiw"] for x in rows])) / base
    v = lambda k: 100 * np.mean([x[k]["viol"] for x in rows])
    print("%-6s %4d %8.2f | %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f %5.1f | viol%% q99 %.2f q95 %.2f q90 %.2f | picp %.3f rmse %.1f share@cap %.2f" % (
        s, len(rows), base, red("K0inf"), red("Kcap"), red("Kage"), red("Kage_cap"), red("Kagecap_q99"), red("Kagecap_q95"), red("Kagecap_q90"),
        v("Kagecap_q99"), v("Kagecap_q95"), v("Kagecap_q90"),
        np.mean([x["none"]["picp"] for x in rows]), np.mean([x["rmse"] for x in rows]), np.mean([x["share_at_cap"] for x in rows])))

# ------------------------------------------------------------------ sensitivity
print("\n=== Sensitivity to the label convention (mean over seeds available in 'sens' + main for cap125)")
for s in SUBS:
    print(s)
    for name in ("cap100", "cap125", "cap150", "uncapped"):
        rows = [r[name] for r in (main[s] if name == "cap125" else sens[s]) if name in r]
        if not rows:
            continue
        base = np.mean([x["none"]["mpiw"] for x in rows])
        red = lambda k: (100 * (base - np.mean([x[k]["mpiw"] for x in rows])) / base) if k in rows[0] else float("nan")
        vio = lambda k: (100 * np.mean([x[k]["viol"] for x in rows])) if k in rows[0] else float("nan")
        print("  %-9s n=%2d MPIW(CP) %7.2f | red%% K0inf %5.1f  Kcap %5.1f  Kage(max) %5.1f  Kage_q99 %5.1f (viol %.2f%%) q95 %5.1f (viol %.2f%%) q90 %5.1f (viol %.2f%%) | picp %.3f | rmse %.1f | share@cap %.2f" % (
            name, len(rows), base, red("K0inf"), red("Kcap"), red("Kage"), red("Kage_q99"), vio("Kage_q99"), red("Kage_q95"), vio("Kage_q95"),
            red("Kage_q90"), vio("Kage_q90"), np.mean([x["none"]["picp"] for x in rows]), np.mean([x["rmse"] for x in rows]),
            np.mean([x["share_at_cap"] for x in rows])))

# ------------------------------------------------------------------ engine-level coverage
print("\n=== Engine-level coverage (cap 125): per-engine PICP pooled over seeds")
for s in SUBS:
    rows = [r["cap125"] for r in main[s]]
    if not rows:
        continue
    ec = np.concatenate([np.array(x["engine"]["cp"]["cov"]) / np.array(x["engine"]["cp"]["n"]) for x in rows])
    ecp = np.concatenate([np.array(x["engine"]["pccp"]["cov"]) / np.array(x["engine"]["pccp"]["n"]) for x in rows])
    print("%s: %d engine-seeds | PICP(cycle) %.3f | engine PICP: min %.2f p05 %.2f p10 %.2f median %.2f mean %.3f | frac<0.8 %.3f frac<0.7 %.3f | max|CP-PCCP| per engine %.2e" % (
        s, len(ec), np.mean([x["none"]["picp"] for x in rows]), ec.min(), np.percentile(ec, 5), np.percentile(ec, 10), np.median(ec), ec.mean(),
        (ec < 0.8).mean(), (ec < 0.7).mean(), np.abs(ec - ecp).max()))

print("\n=== Coverage by RUL band (pooled): band, n, PICP (CP), PICP (PCCP), MPIW CP, MPIW PCCP")
labels = ["[0,15)", "[15,30)", "[30,50)", "[50,75)", "[75,100)", "[100,125)", "=125"]
for s in SUBS:
    rows = [r["cap125"] for r in main[s]]
    if not rows:
        continue
    tot = np.sum([np.array(x["stage_rul"], float) for x in rows], axis=0)
    print(s, "  ".join(f"{labels[i]}: n={int(tot[i][0])} {tot[i][1] / tot[i][0]:.3f}/{tot[i][2] / tot[i][0]:.3f} w {tot[i][3] / tot[i][0]:.1f}/{tot[i][4] / tot[i][0]:.1f}" for i in range(7) if tot[i][0] > 0))

print("\n=== Coverage by fraction of life elapsed (pooled): quintiles")
for s in SUBS:
    rows = [r["cap125"] for r in main[s]]
    if not rows:
        continue
    tot = np.sum([np.array(x["stage_life"], float) for x in rows], axis=0)
    print(s, "  ".join(f"Q{i + 1}: {tot[i][1] / tot[i][0]:.3f}" for i in range(5)))

# ------------------------------------------------------------------ engine-level bootstrap CIs
print("\n=== Engine-level cluster bootstrap (95% CI of seed-averaged PICP and MPIW)")
rng = np.random.RandomState(2026)
for s in SUBS:
    rows = [r["cap125"] for r in main[s]]
    if not rows:
        continue
    B = 1000
    pics, mws_cp, mws_p = [], [], []
    per_seed = [(np.array(x["engine"]["cp"]["n"], float), np.array(x["engine"]["cp"]["cov"]), np.array(x["engine"]["cp"]["w"]),
                 np.array(x["engine"]["pccp"]["w"])) for x in rows]
    for _ in range(B):
        pc, wc, wp = [], [], []
        for n, cv, w1, w2 in per_seed:
            ix = rng.randint(0, len(n), len(n))
            pc.append(cv[ix].sum() / n[ix].sum())
            wc.append(w1[ix].sum() / n[ix].sum())
            wp.append(w2[ix].sum() / n[ix].sum())
        pics.append(np.mean(pc))
        mws_cp.append(np.mean(wc))
        mws_p.append(np.mean(wp))
    pc0 = np.mean([cv.sum() / n.sum() for n, cv, _, _ in per_seed])
    print("%s: PICP %.3f [%.3f, %.3f] | MPIW CP %.2f [%.2f, %.2f] | MPIW PCCP %.2f [%.2f, %.2f]" % (
        s, pc0, *np.percentile(pics, [2.5, 97.5]), np.mean([w1.sum() / n.sum() for n, _, w1, _ in per_seed]), *np.percentile(mws_cp, [2.5, 97.5]),
        np.mean([w2.sum() / n.sum() for n, _, _, w2 in per_seed]), *np.percentile(mws_p, [2.5, 97.5])))

# ------------------------------------------------------------------ engine-level calibration
print("\n=== Engine-level (hierarchical) calibration, cap 125: mean over seeds")
for s in SUBS:
    rows = [r["cap125"]["hier"] for r in main[s]]
    if not rows:
        continue
    print(s, "(m_cal engines ~ %d)" % np.mean([r["n_cal_engines"] for r in main[s]]))
    for k in ("cycle_level", "pooled_cdf", "subsample"):
        g = lambda key: np.mean([x[k][key] for x in rows])
        print("   %-12s q %.1f | PICP %.3f engine-mean %.3f | engine min %.2f p10 %.2f frac<0.8 %.3f | MPIW %.1f -> PCCP %.1f (%.1f%% recovered)" % (
            k, g("q"), g("picp"), g("picp_engine_mean"), g("engine_min"), g("engine_p10"), g("frac_below_0.8"), g("mpiw"), g("mpiw_pccp"),
            100 * (g("mpiw") - g("mpiw_pccp")) / g("mpiw")))
