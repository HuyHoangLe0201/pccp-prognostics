"""Experiment A: C-MAPSS main protocol (Table 1 setting), extended with the constraint, unit-level and label-convention analyses.

Per (subset, seed), with the manuscript's protocol (single-cycle MLP, engine-level
70/15/15 split, alpha = 0.1) this script records, for four label conventions
(piecewise-linear cap Rmax in {125, 100, 150} and the uncapped true RUL T - t):

  * constraint variants: none, K=[0,inf), K=[0,Rmax], the age-based prior
    K(x)=[0, max(0, Tmax_train - t)] and its intersection with [0,Rmax];
    each with PICP, MPIW, violation rate P(Y not in K), empty-intersection rate;
  * for the Rmax=125 convention only: per-engine coverage/width sums, coverage by
    RUL band and by life fraction, and the engine-level (hierarchical)
    calibration variants (pooled-CDF, repeated one-score-per-engine subsampling);
  * calibration/test predictions (npz) so that Figs. 3-4 can be redrawn with seed bands.

usage: python expA_cmapss.py OUTDIR WORKERS N_SEEDS_MAIN N_SEEDS_SENS
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import (ALPHA, conformal_quantile, cov_width, load_cmapss, predict, project,
                    split_by_engine, train_mlp)

SETTINGS_RUN = {"cap125": 125.0, "cap100": 100.0, "cap150": 150.0, "uncapped": None}
RUL_EDGES = [0, 15, 30, 50, 75, 100, 125, np.inf]
N_SUBSAMPLE = 200


def _f(x):
    return float(x)


def variant_row(name, L, U, y, cap, hi):
    """hi: array/scalar upper bound of K (None => +inf)."""
    hi_arr = np.inf if hi is None else hi
    Lp, Up = project(L, U, 0.0, None if hi is None else hi_arr)
    cov, w = cov_width(Lp, Up, y)
    raw_empty = np.maximum(L, 0.0) > (U if hi is None else np.minimum(U, hi_arr))
    viol = (y < 0) | (y > hi_arr)
    return {"picp": _f(cov.mean()), "mpiw": _f(w.mean()), "viol": _f(viol.mean()), "empty": _f(raw_empty.mean())}


def engine_stats(ute, cov, w):
    out = {"n": [], "cov": [], "w": []}
    for u in np.unique(ute):
        m = ute == u
        out["n"].append(int(m.sum()))
        out["cov"].append(_f(cov[m].sum()))
        out["w"].append(_f(w[m].sum()))
    return out


def binned(idx_bins, nb, cov_cp, cov_p, w_cp, w_p):
    rows = []
    for b in range(nb):
        m = idx_bins == b
        rows.append([int(m.sum()), _f(cov_cp[m].sum()), _f(cov_p[m].sum()), _f(w_cp[m].sum()), _f(w_p[m].sum())])
    return rows


def hier_row(qh, f_te, y_te, ute, cap):
    L, U = f_te - qh, f_te + qh
    Lp, Up = project(L, U, 0.0, cap)
    cov, w = cov_width(L, U, y_te)
    covp, wp = cov_width(Lp, Up, y_te)
    ec = np.array([cov[ute == u].mean() for u in np.unique(ute)])
    return {"q": _f(qh), "picp": _f(cov.mean()), "picp_engine_mean": _f(ec.mean()), "engine_min": _f(ec.min()),
            "engine_p10": _f(np.percentile(ec, 10)), "frac_below_0.8": _f((ec < 0.8).mean()),
            "mpiw": _f(w.mean()), "mpiw_pccp": _f(wp.mean()), "picp_pccp": _f(covp.mean())}


def _high_priority():
    """Windows: escape the background/EcoQoS throttling that makes CPU training ~10x slower."""
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x80)
    except Exception:
        pass


def run_seed(args):
    subset, seed, outdir, tag, names = args
    _high_priority()
    path = f"{outdir}/expA_{subset}_seed{seed}_{tag}.json"
    if os.path.exists(path):
        return path
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    D = load_cmapss(subset)
    X = D["X"]
    tr, ca, te = split_by_engine(D["eid"], seed)
    tmax_tr = D["life"][tr].max()
    age_te, ute, life_te = D["age"][te], D["eid"][te], D["life"][te]
    upper_age = np.maximum(0.0, tmax_tr - age_te)
    eids_tr = np.unique(D["eid"][tr])
    lives_tr = np.array([D["life"][tr][D["eid"][tr] == u][0] for u in eids_tr])
    age_q = {q: np.maximum(0.0, np.quantile(lives_tr, q) - age_te) for q in (0.99, 0.95, 0.90)}
    res = {"subset": subset, "seed": seed, "tmax_train": _f(tmax_tr), "n_test": int(te.sum()),
           "n_test_engines": int(len(np.unique(ute))), "n_cal_engines": int(len(np.unique(D["eid"][ca])))}
    for name in names:
        cap = SETTINGS_RUN[name]
        y = D["rul_true"] if cap is None else np.minimum(D["rul_true"], cap)
        model = train_mlp(X[tr], y[tr], seed)
        f_ca, f_te = predict(model, X[ca]), predict(model, X[te])
        y_ca, y_te = y[ca], y[te]
        q = conformal_quantile(np.abs(y_ca - f_ca), ALPHA)
        L, U = f_te - q, f_te + q
        cov0, w0 = cov_width(L, U, y_te)
        row = {"q": _f(q), "rmse": _f(np.sqrt(np.mean((f_te - y_te) ** 2))),
               "share_at_cap": _f((y_te >= cap).mean()) if cap else 0.0,
               "psi_minus": _f((L < 0).mean()), "psi_plus": _f((U > cap).mean()) if cap else 0.0,
               "none": {"picp": _f(cov0.mean()), "mpiw": _f(w0.mean()), "viol": 0.0, "empty": 0.0}}
        row["K0inf"] = variant_row("K0inf", L, U, y_te, cap, None)
        if cap:
            row["Kcap"] = variant_row("Kcap", L, U, y_te, cap, cap)
        row["Kage"] = variant_row("Kage", L, U, y_te, cap, upper_age)
        for q_, hi_ in age_q.items():
            row[f"Kage_q{int(q_ * 100)}"] = variant_row("Kage_q", L, U, y_te, cap, hi_)
        if cap:
            row["Kage_cap"] = variant_row("Kage_cap", L, U, y_te, cap, np.minimum(cap, upper_age))
            for q_, hi_ in age_q.items():
                row[f"Kagecap_q{int(q_ * 100)}"] = variant_row("Kagecap_q", L, U, y_te, cap, np.minimum(cap, hi_))
        if name == "cap125":
            Lp, Up = project(L, U, 0.0, cap)
            covp, wp = cov_width(Lp, Up, y_te)
            row["engine"] = {"cp": engine_stats(ute, cov0, w0), "pccp": engine_stats(ute, covp, wp),
                             "life": [_f(life_te[ute == u][0]) for u in np.unique(ute)]}
            row["stage_rul"] = binned(np.digitize(y_te, RUL_EDGES[1:-1]), len(RUL_EDGES) - 1, cov0, covp, w0, wp)
            frac = np.minimum((age_te / life_te * 5).astype(int), 4)
            row["stage_life"] = binned(frac, 5, cov0, covp, w0, wp)
            # engine-level (hierarchical) calibration, cf. Dunn, Wasserman & Ramdas (JASA 2023)
            sca, uca = np.abs(y_ca - f_ca), D["eid"][ca]
            ucs = np.unique(uca)
            m = len(ucs)
            s_cat = np.concatenate([sca[uca == u] for u in ucs])
            w_cat = np.concatenate([np.full((uca == u).sum(), 1.0 / ((uca == u).sum() * m)) for u in ucs])
            srt = np.argsort(s_cat)
            cw = np.cumsum(w_cat[srt])
            lvl = min(1.0, (1 - ALPHA) * (m + 1) / m)
            q_pool = float(s_cat[srt][min(np.searchsorted(cw, lvl - 1e-12), len(cw) - 1)])
            rng = np.random.RandomState(seed + 7)
            q_sub = float(np.mean([conformal_quantile(np.array([rng.choice(sca[uca == u]) for u in ucs]), ALPHA)
                                   for _ in range(N_SUBSAMPLE)]))
            row["hier"] = {"cycle_level": hier_row(q, f_te, y_te, ute, cap),
                           "pooled_cdf": hier_row(q_pool, f_te, y_te, ute, cap),
                           "subsample": hier_row(q_sub, f_te, y_te, ute, cap)}
            np.savez_compressed(f"{outdir}/expA_{subset}_seed{seed}_arrays.npz", f_ca=f_ca.astype(np.float32),
                                y_ca=y_ca.astype(np.float32), f_te=f_te.astype(np.float32),
                                y_te=y_te.astype(np.float32), ute=ute, age_te=age_te.astype(np.float32),
                                life_te=life_te.astype(np.float32))
        res[name] = row
    res["seconds"] = time.time() - t0
    with open(path, "w") as fh:
        json.dump(res, fh)
    return path


if __name__ == "__main__":
    # usage: python expA_cmapss.py OUTDIR WORKERS MAIN_PLAN SENS_PLAN   (plans like FD001:30,FD002:12,...)
    outdir = sys.argv[1]
    workers = int(sys.argv[2])
    plan = lambda a: {kv.split(":")[0]: int(kv.split(":")[1]) for kv in a.split(",")}
    n_main, n_sens = plan(sys.argv[3]), plan(sys.argv[4])
    os.makedirs(outdir, exist_ok=True)
    order = ["FD004", "FD002", "FD003", "FD001"]      # heaviest first within a seed
    tasks = []
    for s in range(max(list(n_main.values()) + list(n_sens.values()))):
        for sub in order:
            if s < n_main.get(sub, 0):
                tasks.append((sub, s, outdir, "main", ("cap125",)))
            if s < n_sens.get(sub, 0):
                tasks.append((sub, s, outdir, "sens", ("cap100", "cap150", "uncapped")))
    print(len(tasks), "tasks", flush=True)
    t0 = time.time()
    if workers <= 1:
        for t in tasks:
            print(run_seed(t), f"{time.time() - t0:.0f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for p in ex.map(run_seed, tasks):
                print(p, f"{time.time() - t0:.0f}s", flush=True)
    print("ALL DONE", flush=True)
