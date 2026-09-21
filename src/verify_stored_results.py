r"""Independent checks of the stored raw results (no data files and no training needed).

usage: python verify_stored_results.py [RESULTS_DIR]        (default ../results)

1. ACI: an independent re-implementation of the level/window bookkeeping (oracle, fixed delay, end-of-life feedback) must
   reproduce the stored intervals of Experiment C (seed 0) exactly; the exact telescoping identity
   PICP = 1 - alpha + (alpha_{T+1} - alpha_1) / (gamma * T) is checked on the same paths.
2. Aggregation: the manuscript numbers (numbers/numbers_C.json) must equal the means recomputed from the raw per-seed files.
3. Battery: every entry of Table 11 and of Section 5.8 (coverage preservation, wasted-budget identity, Delta_pred bound,
   folds needing the general form) is recomputed from the stored per-fold arrays.
4. Experiments D and E (Table 9, Table 14): the projections of every construction have exactly the PICP of their base interval, and
   the CQR-based ACI and its projections share every miss indicator.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results")
ALPHA, GAMMA, W = 0.10, 0.03, 500
failed = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (" | " + detail if detail else ""))
    if not ok:
        failed.append(name)


# ------------------------------------------------------------------------------------------------ 1. ACI
def window_q(scores, a):
    m = len(scores)
    k = int(np.ceil((m + 1) * (1 - a)))
    return np.sort(scores)[min(max(k, 1), m) - 1]


for fd in ("FD004", "FD003"):
    z = np.load(f"{RES}/expC/expC_seed0_{fd}_arrays.npz")
    y, f, u, cap = z["y"], z["f"], z["unit"], float(z["cap"])
    n, s = len(y), np.abs(z["y"] - z["f"])
    first = {uu: int(np.where(u == uu)[0][0]) for uu in np.unique(u)}
    for name, (L, U), applied in (("oracle", (z["L_aci"], z["U_aci"]), lambda t: np.arange(0, t)),
                                  ("delay50", (z["L_d50"], z["U_d50"]), lambda t: np.arange(0, max(t - 50, 0))),
                                  ("end-of-life", (z["L_eol"], z["U_eol"]), lambda t: np.arange(0, first[u[t]]))):
        miss = ~((np.maximum(L, 0) <= y) & (y <= np.minimum(U, cap)))
        upd = ALPHA - miss.astype(float)
        err, alphas = 0.0, []
        for t in range(n):
            ap = applied(t)
            a_t = ALPHA + GAMMA * upd[ap].sum()
            alphas.append(a_t)
            if len(ap) >= W:
                q_t = window_q(s[ap][-W:], a_t)
                err = max(err, abs((f[t] - q_t) - L[t]), abs((f[t] + q_t) - U[t]))
        ap_all = applied(n - 1)
        a_end = ALPHA + GAMMA * upd[ap_all].sum()
        ident = 1 - ALPHA + (a_end - ALPHA) / (GAMMA * len(ap_all))
        check(f"ACI {fd} {name}: intervals reproduced", err < 1e-9, f"max error {err:.1e}, level range [{min(alphas):.2f}, {max(alphas):.2f}]")
        check(f"ACI {fd} {name}: telescoping identity", abs((1 - miss[ap_all].mean()) - ident) < 1e-9, f"coverage {1 - miss[ap_all].mean():.4f}")

# ------------------------------------------------------------------------------------------------ 2. aggregation
R = [json.load(open(p)) for p in sorted(glob.glob(f"{RES}/expC/expC_seed*.json"))]
N = json.load(open(f"{RES}/numbers/numbers_C.json"))
worst = 0.0
for fd in ("FD004", "FD003"):
    A = [r["targets"][fd]["caps"]["125"]["A"] for r in R]
    for key in A[0]["table"]:
        for m in ("picp", "mpiw", "feas", "iscore"):
            worst = max(worst, abs(np.mean([a["table"][key][m] for a in A]) - N[fd]["table"][key][m][0]))
    for cf in ("20", "50", "100"):
        for arm in ("CP", "PCCP-R", "ACI", "PCCP-inf+ACI", "PCCP-R+ACI"):
            for t in ("capped", "uncapped"):
                worst = max(worst, abs(np.mean([a["per_moment"][cf][arm][t] for a in A]) - N[fd]["per_moment"][cf][arm][t][0]))
    for prot in ("oracle", "eol"):
        for arm in ("CP", "ACI", "PCCP-inf+ACI", "PCCP-R+ACI"):
            for delta in ("1", "5", "10", "25"):
                for cf in ("5", "20", "100"):
                    v = np.mean([r["targets"][fd]["caps"]["125"]["B"]["sequential"][prot][arm][delta][cf]["rate"] for r in R])
                    worst = max(worst, abs(v - N[fd]["sequential"]["125"][prot][arm][delta][cf]["rate"][0]))
    check(f"projection invariance {fd}: zero miss-indicator mismatches", sum(a["invariance_miss_mismatch"] for a in A) == 0)
check("Experiment C aggregation equals the recomputed means", worst < 1e-9, f"max difference {worst:.1e}")
# beta = 0 decision neutrality (Proposition 2): the projected and the unprojected constructor coincide exactly
mx, cnt = 0.0, 0
pairs = (("CP", "PCCP-inf"), ("CP", "PCCP-R"), ("ACI", "PCCP-inf+ACI"), ("ACI", "PCCP-R+ACI"))
for r in R:
    for fd in ("FD004", "FD003"):
        for cap in ("100", "125", "150"):
            seq = r["targets"][fd]["caps"][cap]["B"]["sequential"]
            for prot in ("oracle", "eol"):
                for a, b in pairs:
                    for delta in ("1", "5", "10", "25"):
                        for cf in ("5", "10", "20", "50", "100"):
                            mx = max(mx, abs(seq[prot][a][delta][cf]["rate_beta0"] - seq[prot][b][delta][cf]["rate_beta0"]))
                            cnt += 1
check("Proposition 2: cost rates at beta = 0 identical for projected/unprojected", mx == 0.0, f"{cnt} comparisons over {len(R)} seeds, max difference {mx}")

# ------------------------------------------------------------------------------------------------ 3. battery
B = json.load(open(f"{RES}/battery.json"))["results"]
bad, gen_needed, full_ok, simple_ok, picp_eq, loss_max = 0.0, 0, 0, 0, 0, 0.0
loss5 = []
for r in B:
    y, f, L, U = (np.array(r["arrays"][k], float) for k in ("y", "f", "L", "U"))
    rm, st, q = r["loo"]["rmax"], r["loo"], r["q_cp"]
    Lp, Up = np.maximum(L, 0), np.minimum(U, rm)
    cov = lambda a, b: ((y >= a) & (y <= b)).mean()
    w = lambda a, b: np.maximum(b - a, 0).mean()
    ft = np.clip(f, 0, rm)
    Lf, Uf = np.maximum(ft - st["q_fcp"], 0), np.minimum(ft + st["q_fcp"], rm)
    bad = max(bad, abs(cov(L, U) - st["cp"]["picp"]), abs(w(L, U) - st["cp"]["mpiw"]), abs(cov(Lp, Up) - st["pccp"]["picp"]),
              abs(w(Lp, Up) - st["pccp"]["mpiw"]), abs(cov(Lf, Uf) - st["fcp"]["picp"]), abs(w(Lf, Uf) - st["fcp"]["mpiw"]))
    Wg = (np.maximum(0, np.minimum(U, 0) - L) + np.maximum(0, U - np.maximum(L, rm))).mean()
    Ws = (np.maximum(0, -L) + np.maximum(0, U - rm)).mean()
    bad = max(bad, abs((w(L, U) - w(Lp, Up)) - Wg))
    gen_needed += abs(Wg - Ws) > 1e-9
    gap, dp = abs(w(Lp, Up) - w(Lf, Uf)), np.mean(np.abs(f - ft))
    full_ok += gap <= dp + 2 * (q - st["q_fcp"]) + 1e-9
    simple_ok += gap <= 2 * (q - st["q_fcp"]) + 1e-9
    picp_eq += abs(cov(L, U) - cov(Lp, Up)) < 1e-12
    loss_max = max(loss_max, cov(L, U) - cov(Lp, Up))
    if r["cell"] == "B0005":
        loss5.append(cov(L, U) - cov(Lp, Up))
check("battery: stored metrics and the wasted-budget identity (general form) recomputed", bad < 1e-12, f"max difference {bad:.1e}")
check("battery: general form needed in 43 of 150 folds", gen_needed == 43, f"{gen_needed}/150")
check("battery: Delta_pred bound holds in 150 folds, the simple bound in 93", (full_ok, simple_ok) == (150, 93), f"{full_ok}, {simple_ok}")
check("battery: PICP(PCCP) = PICP(CP) in 145 of 150 folds, loss <= 0.012", picp_eq == 145 and loss_max <= 0.012, f"{picp_eq}/150, max loss {loss_max:.4f}, B0005 mean loss {np.mean(loss5):.5f}")

# ------------------------------------------------------------------------------------------------ 4. Experiments D and E
D = [json.load(open(f)) for f in sorted(glob.glob(f"{RES}/expD/expD_*.json"))]
worst = max((abs(r[m]["picp"] - r[m][k]["picp"]) for r in D for m in ("lw", "cqr", "qr", "mcd") for k in ("Kcap", "K0inf")), default=1.0)
check("Experiment D: PICP of every projected construction equals that of its base interval (empty set = miss)", len(D) == 36 and worst == 0.0,
      f"{len(D)} runs, max difference {worst}")
E = [json.load(open(f)) for f in sorted(glob.glob(f"{RES}/expE/expE_seed*.json"))]
cells = [(r, fd, p) for r in E for fd in ("FD004", "FD003") for p in ("oracle", "delay50", "eol")]
mism = sum(r[fd][p]["diag"]["indicator_mismatches"] for r, fd, p in cells)
worst = max((abs(r[fd][p]["ACI"]["picp"] - r[fd][p][a]["picp"]) for r, fd, p in cells for a in ("PCCP-inf+ACI", "PCCP-R+ACI")), default=1.0)
check("Experiment E: CQR+ACI and its projections share miss indicators and PICP", len(E) == 5 and mism == 0 and worst == 0.0,
      f"{len(E)} seeds, {mism} mismatching indicators, max PICP difference {worst}")

# ------------------------------------------------------------------------------------------------ 5. Theorem 1 (ii) on the feasible subset (FD001)
def _cq(s, alpha=ALPHA):
    n = len(s)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    return float(np.max(s)) if level >= 1.0 else float(np.quantile(s, level, method="higher"))


worst, n_seeds = 0.0, 0
for f in sorted(glob.glob(f"{RES}/expA/expA_FD001_seed*_arrays.npz")):
    z = np.load(f)
    fca, yca, fte = (z[k].astype(float) for k in ("f_ca", "y_ca", "f_te"))
    mca, mte = (fca >= 0) & (fca <= 125), (fte >= 0) & (fte <= 125)
    q_hat, q_til = _cq(np.abs(yca[mca] - fca[mca])), _cq(np.abs(yca[mca] - np.clip(fca[mca], 0, 125)))
    ft = fte[mte]
    Lp, Up = np.maximum(ft - q_hat, 0), np.minimum(ft + q_hat, 125)
    Lf, Uf = np.maximum(ft - q_til, 0), np.minimum(ft + q_til, 125)
    worst = max(worst, np.abs(Lp - Lf).max(), np.abs(Up - Uf).max(), abs(q_hat - q_til))
    n_seeds += 1
check("Theorem 1(ii): PCCP and FCP coincide when every prediction is feasible (FD001, feasible subset)", n_seeds == 30 and worst == 0.0,
      f"{n_seeds} seeds, max difference {worst}")
print()
print("ALL CHECKS PASSED" if not failed else "FAILED: " + "; ".join(failed))
sys.exit(1 if failed else 0)
