"""Aggregate Experiment B (battery, leakage-free Rmax) into the numbers used in Section 5.8 (Table 11)."""
import json
import sys

import numpy as np

d = json.load(open(sys.argv[1]))
R = d["results"]
cells = d["cells"]
print("folds:", len(R), "cells:", cells)


def stat(rows, mode, method, key):
    v = np.array([r[mode][method][key] for r in rows])
    return v.mean(), v.std()


def table(mode):
    print(f"\n=== Table 11 rows, mode = {mode}")
    for c in cells + ["Average"]:
        rows = R if c == "Average" else [r for r in R if r["cell"] == c]
        for m in ("cp", "pccp", "fcp"):
            pm, ps = stat(rows, mode, m, "picp")
            mw, _ = stat(rows, mode, m, "mpiw")
            neg, _ = stat(rows, mode, m, "psi_neg")
            pos, _ = stat(rows, mode, m, "psi_pos")
            ok, _ = stat(rows, mode, m, "psi_ok")
            print(f"{c:8s} {m.upper():5s} PICP {pm:.3f} ({ps:.3f})  MPIW {mw:6.2f}  psi- {neg:5.1f} psi+ {pos:5.1f} psi_ok {ok:6.1f}")


for mode in ("legacy", "loo", "noceil"):
    table(mode)

print("\n=== Constraint violation P(Y not in K) and coverage cost (mode = loo)")
for c in cells:
    rows = [r for r in R if r["cell"] == c]
    viol = np.mean([r["loo"]["viol"] for r in rows])
    cost = np.mean([r["loo"]["cp"]["picp"] - r["loo"]["pccp"]["picp"] for r in rows])
    rm = rows[0]["rmax_loo"]
    print(f"{c}: Rmax_train={rm}  delta={viol:.4f}  mean PICP(CP)-PICP(PCCP)={cost:.4f}  max over folds={max(r['loo']['cp']['picp'] - r['loo']['pccp']['picp'] for r in rows):.4f}")

print("\n=== MPIW reduction (%) per cell: K=[0,Rmax_loo] vs K=[0,inf) vs legacy K=[0,124]")
for c in cells + ["Average"]:
    rows = R if c == "Average" else [r for r in R if r["cell"] == c]
    out = []
    for mode in ("noceil", "loo", "legacy"):
        cp = np.mean([r[mode]["cp"]["mpiw"] for r in rows])
        pc = np.mean([r[mode]["pccp"]["mpiw"] for r in rows])
        out.append(f"{mode}: {100 * (cp - pc) / cp:5.1f}")
    print(c, "  ".join(out))

print("\n=== Lemma 1(ii) / wasted budget identity (mode loo): max |MPIW(C)-MPIW(Cproj)-W_general|")
disc = [abs(r["loo"]["cp"]["mpiw"] - r["loo"]["pccp"]["mpiw"] - r["loo"]["cp"]["wasted"]) for r in R]
print("max %.3e  mean %.3e" % (max(disc), np.mean(disc)))
gen_needed = sum(1 for r in R if abs(r["loo"]["cp"]["wasted"] - r["loo"]["cp"]["wasted_simple"]) > 1e-9)
print("folds where the general form differs from the simplified form:", gen_needed, "/", len(R))

print("\n=== coverage identity PICP(Cproj)=PICP(C) (mode loo), by cell: folds equal / total")
for c in cells:
    rows = [r for r in R if r["cell"] == c]
    eq = sum(1 for r in rows if abs(r["loo"]["cp"]["picp"] - r["loo"]["pccp"]["picp"]) < 1e-12)
    print(c, eq, "/", len(rows))
for tau in ("30", "15"):
    for c in cells:
        rows = [r for r in R if r["cell"] == c and r["loo"]["sub"][tau]["cp"] is not None]
        eq = sum(1 for r in rows if abs(r["loo"]["sub"][tau]["cp"]["picp"] - r["loo"]["sub"][tau]["pccp"]["picp"]) < 1e-12)
        red = np.mean([(r["loo"]["sub"][tau]["cp"]["mpiw"] - r["loo"]["sub"][tau]["pccp"]["mpiw"]) / r["loo"]["sub"][tau]["cp"]["mpiw"] for r in rows]) * 100
        print(f"subgroup RUL<{tau} {c}: equal in {eq}/{len(rows)}; mean MPIW reduction {red:.1f}%")

print("\n=== FCP vs PCCP (mode loo)")
holds_full = holds_simple = 0
for r in R:
    m = r["loo"]
    gap = abs(m["pccp"]["mpiw"] - m["fcp"]["mpiw"])
    simple = 2 * (r["q_cp"] - m["q_fcp"])
    if gap <= simple + m["delta_pred"] + 1e-9:
        holds_full += 1
    if gap <= simple + 1e-9:
        holds_simple += 1
print("width-gap bound with Delta_pred holds:", holds_full, "/", len(R), "; simple bound holds:", holds_simple, "/", len(R))
for c in cells:
    rows = [r for r in R if r["cell"] == c]
    print(c, "PICP PCCP %.3f FCP %.3f | MPIW PCCP %.2f FCP %.2f | Delta_pred %.3f" % (
        np.mean([r["loo"]["pccp"]["picp"] for r in rows]), np.mean([r["loo"]["fcp"]["picp"] for r in rows]),
        np.mean([r["loo"]["pccp"]["mpiw"] for r in rows]), np.mean([r["loo"]["fcp"]["mpiw"] for r in rows]),
        np.mean([r["loo"]["delta_pred"] for r in rows])))

print("\n=== KS distance and coverage gap (mode loo, CP)")
for c in cells + ["All"]:
    rows = R if c == "All" else [r for r in R if r["cell"] == c]
    ks = np.array([r["ks"] for r in rows])
    gap = np.array([max(0.0, 0.9 - r["loo"]["cp"]["picp"]) for r in rows])
    gap_p = np.array([max(0.0, 0.9 - r["loo"]["pccp"]["picp"]) for r in rows])
    print(f"{c}: KS {ks.mean():.3f} ({ks.std():.3f}); gap CP {gap.mean():.3f}; gap PCCP {gap_p.mean():.3f}; KS>=gap in {int((ks >= gap).sum())}/{len(rows)}; ratio {np.mean(ks / np.maximum(gap, 1e-9)):.2f}")
