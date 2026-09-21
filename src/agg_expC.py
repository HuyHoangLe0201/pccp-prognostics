r"""Aggregate Experiment C (shift + decisions) over seeds; print a readable digest and write
gen/numbers_C.json  (all means and SDs used in Sections 5.9-5.10)."""
import glob
import json
import os
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "out/expC"
GEN = sys.argv[2] if len(sys.argv) > 2 else "gen"
os.makedirs(GEN, exist_ok=True)
R = [json.load(open(f)) for f in sorted(glob.glob(f"{OUT}/expC_seed*.json"))]
print("seeds:", [r["seed"] for r in R])
TARGETS = ["FD004", "FD003"]
ARMS = ["CP", "PCCP-inf", "PCCP-R", "ACI", "PCCP-inf+ACI", "PCCP-R+ACI"]


def ms(vals):
    """mean and sample standard deviation (n-1 denominator) over seeds"""
    v = np.asarray(vals, float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


N = {"seeds": [r["seed"] for r in R]}
for fd in TARGETS:
    A = [r["targets"][fd]["caps"]["125"]["A"] for r in R]
    B = {cap: [r["targets"][fd]["caps"][cap]["B"] for r in R] for cap in ("125", "100", "150")}
    q = ms([r["targets"][fd]["caps"]["125"]["q"] for r in R])
    n = {"q": q, "rmse_source": ms([r["targets"][fd]["caps"]["125"]["rmse_source"] for r in R]), "rmse_A": ms([a["rmse"] for a in A]),
         "rmse_B": ms([b["rmse"] for b in B["125"]]), "share_at_cap_A": ms([a["share_at_cap"] for a in A]),
         "share_pred_clipped": ms([a["share_pred_clipped"] for a in A]), "empty_raw_predictor": ms([a["empty_raw_predictor"] for a in A]),
         "invariance_mismatch_total": int(sum(a["invariance_miss_mismatch"] for a in A)), "pearson": ms([a["pearson_mass_vs_shift"] for a in A]),
         "mass_range": [ms([a["mass_range"][0] for a in A])[0], ms([a["mass_range"][1] for a in A])[0]],
         "share_capped_upper_CP": ms([a["share_capped_upper_CP"] for a in A]), "n_A": R[0]["targets"][fd]["n_A"], "n_B": R[0]["targets"][fd]["n_B"],
         "n_units_A": R[0]["targets"][fd]["n_units_A"], "n_units_B": R[0]["targets"][fd]["n_units_B"]}
    # coverage table for all protocols
    tab = {}
    for key in A[0]["table"]:
        tab[key] = {m: ms([a["table"][key][m] for a in A]) for m in ("picp", "mpiw", "feas", "iscore", "empty")}
    n["table"] = tab
    n["diag"] = {p: {k: ms([a["diag"][p][k] for a in A]) for k in ("frac_level_saturated", "alpha_min", "alpha_max")} for p in A[0]["diag"]}
    print(f"\n===== {fd}  (n_A={n['n_A']}, q={q[0]:.1f}, RMSE src {n['rmse_source'][0]:.1f} -> A {n['rmse_A'][0]:.1f}, share@cap {n['share_at_cap_A'][0]:.3f}, "
          f"clipped preds {n['share_pred_clipped'][0]:.3f}, empty(raw) {n['empty_raw_predictor'][0]:.4f}, invariance mismatches {n['invariance_mismatch_total']}, Pearson {n['pearson'][0]:.2f})")
    print("%-26s %14s %14s %8s %14s" % ("arm [oracle]", "PICP", "MPIW", "feas", "IScore"))
    for arm in ARMS:
        key = arm if arm in ("CP", "PCCP-inf", "PCCP-R") else arm.replace("ACI", "ACI[oracle]")
        k = key if arm in ("CP", "PCCP-inf", "PCCP-R") else ("ACI[oracle]" if arm == "ACI" else arm + "[oracle]")
        t = tab[k]
        print("%-26s %6.3f(%.3f) %7.1f(%4.1f) %8.2f %7.1f(%4.1f)" % (arm, *t["picp"], *t["mpiw"], t["feas"][0], *t["iscore"]))
    print(" -- feedback protocols (PICP / MPIW): ACI | PCCP-R+ACI | frac level saturated")
    for p in ("oracle", "delay10", "delay50", "delay125", "eol", "gamma0.005", "gamma0.01", "gamma0.1"):
        a, b = tab[f"ACI[{p}]"], tab[f"PCCP-R+ACI[{p}]"]
        print("  %-10s ACI %.3f/%.1f | PCCP-R+ACI %.3f/%.1f | sat %.3f alpha_min %.2f" % (p, a["picp"][0], a["mpiw"][0], b["picp"][0], b["mpiw"][0],
                                                                                    n["diag"][p]["frac_level_saturated"][0], n["diag"][p]["alpha_min"][0]))
    # per-moment
    pm = {}
    for cf in ("20", "50", "100"):
        pm[cf] = {arm: {t: ms([a["per_moment"][cf][arm][t] for a in A]) for t in ("capped", "uncapped")} for arm in ARMS}
    n["per_moment"] = pm
    print(" -- per-moment cost-regret (capped | uncapped truth) at cf=20 / 100")
    for arm in ARMS:
        print("  %-14s cf20 %6.1f | %6.1f   cf100 %6.1f | %6.1f" % (arm, pm["20"][arm]["capped"][0], pm["20"][arm]["uncapped"][0], pm["100"][arm]["capped"][0], pm["100"][arm]["uncapped"][0]))
    # sequential
    seq = {}
    for cap in ("125", "100", "150"):
        seq[cap] = {}
        for prot in ("oracle", "eol"):
            seq[cap][prot] = {}
            for arm in ARMS:
                seq[cap][prot][arm] = {}
                for delta in ("1", "5", "10", "25"):
                    seq[cap][prot][arm][delta] = {}
                    for cf in ("5", "10", "20", "50", "100"):
                        rows = [b["sequential"][prot][arm][delta][cf] for b in B[cap]]
                        seq[cap][prot][arm][delta][cf] = {k: ms([r_[k] for r_ in rows]) for k in ("rate", "fail_rate", "waste", "rate_beta0", "beta")}
    n["sequential"] = seq
    n["seq_diag"] = {cap: {prot: {k: ms([b["sequential"][prot]["_diag"][k] for b in B[cap]]) for k in ("frac_level_saturated", "alpha_min")} for prot in ("oracle", "eol")} for cap in B}
    print(" -- sequential (cap 125): cost rate x1e3 [fail %] , Delta=10")
    for prot in ("oracle", "eol"):
        for cf in ("20", "100"):
            print("   %-6s cf=%-3s " % (prot, cf) + "  ".join("%s %.1f[%.0f]" % (arm, 1e3 * seq["125"][prot][arm]["10"][cf]["rate"][0], 100 * seq["125"][prot][arm]["10"][cf]["fail_rate"][0]) for arm in ARMS))
    N[fd] = n
json.dump(N, open(f"{GEN}/numbers_C.json", "w"), indent=1)
print("\nwritten", f"{GEN}/numbers_C.json")
