r"""Independent recomputation of Experiment A results from the stored predictions (needs the C-MAPSS training files, no training).

usage: python verify_with_data.py [RESULTS_DIR]        (default ../results; C-MAPSS folder from CMAPSS_DIR, see common.py)

* constraint variants (Tables 2-3 capped block): PICP, MPIW and violation rate recomputed for every constraint and every run;
  the soft-constraint inequality  PICP(C) - PICP(C n K) <= violation rate  is checked with the convention that an empty
  intersection is a miss of zero width (the code's guard L <= U changes nothing here, which is checked too);
* unit-level statistics, RUL-band coverage (Table 5, Fig. 5) and the engine-level calibration quantiles (Table 6) are recomputed
  with a different code path and compared with the stored values.
"""
import glob
import json
import os
import sys

import numpy as np

from common import conformal_quantile, load_cmapss, project, split_by_engine

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results")
ALPHA, CAP = 0.1, 125.0
failed = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (" | " + detail if detail else ""))
    if not ok:
        failed.append(name)


mism = {"variants": 0.0, "guard": 0.0, "engine_cov": 0.0, "bands": 0.0, "q_pool": 0.0, "q_sub": 0.0}
worst_ineq, n_runs = -1.0, 0
for sub in ("FD001", "FD002", "FD003", "FD004"):
    D = load_cmapss(sub)
    for f in sorted(glob.glob(f"{RES}/expA/expA_{sub}_seed*_main.json")):
        r = json.load(open(f))
        seed = r["seed"]
        z = np.load(f"{RES}/expA/expA_{sub}_seed{seed}_arrays.npz")
        fca, yca, fte, yte = (z[k].astype(np.float64) for k in ("f_ca", "y_ca", "f_te", "y_te"))
        ute, age, life = z["ute"], z["age_te"].astype(float), z["life_te"].astype(float)
        q = r["cap125"]["q"]
        L, U = fte - q, fte + q
        tr, ca, te = split_by_engine(D["eid"], seed)
        lives_tr = np.array([D["life"][tr][D["eid"][tr] == u][0] for u in np.unique(D["eid"][tr])])
        his = {"K0inf": None, "Kcap": np.full_like(age, CAP), "Kage": np.maximum(0, lives_tr.max() - age)}
        for qq in (0.99, 0.95, 0.90):
            his[f"Kage_q{int(qq * 100)}"] = np.maximum(0, np.quantile(lives_tr, qq) - age)
            his[f"Kagecap_q{int(qq * 100)}"] = np.minimum(CAP, his[f"Kage_q{int(qq * 100)}"])
        cov0 = (yte >= L) & (yte <= U)
        for name, hi in his.items():
            hi_ = np.inf if hi is None else hi
            Lg, Ug = project(L, U, 0.0, hi)                       # guarded projection used by the code
            cg, wg = (yte >= Lg) & (yte <= Ug), np.maximum(Ug - Lg, 0)
            Lp, Up = np.maximum(L, 0), np.minimum(U, hi_)
            empty = Lp > Up
            cm, wm = (yte >= Lp) & (yte <= Up) & ~empty, np.where(empty, 0.0, Up - Lp)
            viol = ((yte < 0) | (yte > hi_)).mean()
            j = r["cap125"][name]
            mism["variants"] = max(mism["variants"], abs(j["picp"] - cg.mean()), abs(j["mpiw"] - wg.mean()), abs(j["viol"] - viol))
            mism["guard"] = max(mism["guard"], abs(cg.mean() - cm.mean()), abs(wg.mean() - wm.mean()))
            worst_ineq = max(worst_ineq, (cov0.mean() - cm.mean()) - viol)
        # unit-level statistics and RUL bands
        cov = np.abs(yte - fte) <= q
        eng = np.unique(ute)
        mism["engine_cov"] = max(mism["engine_cov"], float(np.abs(np.array([cov[ute == e].mean() for e in eng]) -
                                                               np.array(r["cap125"]["engine"]["cp"]["cov"]) / np.array(r["cap125"]["engine"]["cp"]["n"])).max()))
        b = np.digitize(yte, [15, 30, 50, 75, 100, 125])
        mism["bands"] = max(mism["bands"], float(np.abs(np.array(r["cap125"]["stage_rul"], float)[:, 1] - np.array([cov[b == i].sum() for i in range(7)])).max()))
        # engine-level calibration quantiles
        uca, s = D["eid"][ca], np.abs(yca - fca)
        engs = np.unique(uca)
        m = len(engs)
        level = min(1.0, (1 - ALPHA) * (m + 1) / m)
        cand = np.sort(s)
        F = np.zeros(len(cand))
        for e in engs:
            se = np.sort(s[uca == e])
            F += np.searchsorted(se, cand, side="right") / len(se)
        F /= m
        q_pool = float(cand[np.argmax(F >= level - 1e-12)])
        mism["q_pool"] = max(mism["q_pool"], abs(q_pool - r["cap125"]["hier"]["pooled_cdf"]["q"]))
        rng = np.random.RandomState(seed + 7)
        qs = [conformal_quantile(np.array([rng.choice(s[uca == e]) for e in engs]), ALPHA) for _ in range(200)]
        mism["q_sub"] = max(mism["q_sub"], abs(float(np.mean(qs)) - r["cap125"]["hier"]["subsample"]["q"]))
        n_runs += 1
check(f"constraint variants recomputed exactly ({n_runs} runs)", mism["variants"] < 1e-12, f"max difference {mism['variants']:.1e}")
check("guarded projection = empty-is-miss convention", mism["guard"] < 1e-12, f"max difference {mism['guard']:.1e}")
check("soft-constraint inequality PICP(C) - PICP(C n K) <= violation rate", worst_ineq <= 1e-12, f"worst margin {worst_ineq:.1e}")
check("per-engine coverage and RUL-band coverage recomputed", mism["engine_cov"] < 1e-12 and mism["bands"] < 1e-9, f"{mism['engine_cov']:.1e}, {mism['bands']:.1e}")
check("engine-level calibration quantiles recomputed", mism["q_pool"] < 1e-9 and mism["q_sub"] < 1e-9, f"{mism['q_pool']:.1e}, {mism['q_sub']:.1e}")
print()
print("ALL CHECKS PASSED" if not failed else "FAILED: " + "; ".join(failed))
sys.exit(1 if failed else 0)
