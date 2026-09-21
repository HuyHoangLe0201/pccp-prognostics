"""Experiment B: NASA Li-ion battery leave-one-cell-out (Table 11 setting), leakage-free Rmax.

The submitted notebook (Battery_RUL_PCCP_Colab_v4) set Rmax = 124 = the largest RUL over *all* valid
cells, i.e. including the held-out test cell.  Here, for every fold the feasible cap is estimated from
the two *training* cells only ("loo" mode); the legacy value is kept ("legacy" mode) so that the effect
of the leakage can be quantified on identical trained models.  One network is trained per
(held-out cell, seed): the predictor and the CP quantile do not depend on Rmax, only the projection
and the FCP predictor-projection do.

usage: python expB_battery.py OUTFILE.json WORKERS [N_SEEDS]
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import ks_2samp

BAT_DIR = os.environ.get("BATTERY_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "NASABattery"))
CELLS = ["B0005", "B0006", "B0007", "B0018"]
EOL_CAPACITY = 1.4
ALPHA = 0.1
LEGACY_RMAX = 124
FEATURES = ["discharge_time", "V_mean", "V_std", "V_min", "V_max", "V_slope", "I_mean", "I_std",
            "T_mean", "T_max", "T_std", "T_slope", "capacity"]


def load_cell(name):
    mat = loadmat(f"{BAT_DIR}/{name}.mat", squeeze_me=True, struct_as_record=False)
    rows, k = [], 0
    for c in mat[name].cycle:
        if c.type != "discharge":
            continue
        d = c.data
        V, I, T = (np.asarray(getattr(d, a)).flatten() for a in ("Voltage_measured", "Current_measured", "Temperature_measured"))
        t = np.asarray(d.Time).flatten()
        cap = float(np.asarray(d.Capacity).flatten()[0])
        if len(V) < 5:
            continue
        slope = lambda y: float(np.polyfit(np.arange(len(y)), y, 1)[0])
        rows.append({"cycle_idx": k, "capacity": cap, "discharge_time": float(t[-1] - t[0]),
                     "V_mean": float(V.mean()), "V_std": float(V.std()), "V_min": float(V.min()), "V_max": float(V.max()),
                     "V_slope": slope(V), "I_mean": float(I.mean()), "I_std": float(I.std()),
                     "T_mean": float(T.mean()), "T_max": float(T.max()), "T_std": float(T.std()), "T_slope": slope(T)})
        k += 1
    df = pd.DataFrame(rows)
    df["cell"] = name
    eol = df["capacity"] <= EOL_CAPACITY
    if not eol.any():
        return df, False
    te = int(df.index[eol][0])
    df["RUL"] = (te - df["cycle_idx"]).clip(lower=0).astype(int)
    return df, True


def _high_priority():
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x80)
    except Exception:
        pass


def conformal_q(scores, alpha=ALPHA):
    n = len(scores)
    k = min(max(int(np.ceil((n + 1) * (1 - alpha))), 1), n)
    return float(np.sort(scores)[k - 1])


def metrics(y, L, U, rmax):
    cov = (y >= L) & (y <= U)
    w = np.maximum(U - L, 0.0)
    hi = np.inf if rmax is None else rmax
    lower = np.maximum(0, np.minimum(U, 0) - L)
    upper = np.maximum(0, U - np.maximum(L, hi)) if rmax is not None else 0.0
    return {"picp": float(cov.mean()), "mpiw": float(w.mean()), "psi_neg": float((L < 0).mean() * 100),
            "psi_pos": float((U > hi).mean() * 100), "psi_ok": float(((L >= 0) & (U <= hi)).mean() * 100),
            "wasted": float((lower + upper).mean()),
            "wasted_simple": float((np.maximum(0, -L) + (np.maximum(0, U - hi) if rmax is not None else 0)).mean())}


def sub_picp(y, L, U, tau):
    m = y < tau
    if m.sum() == 0:
        return None
    return {"n": int(m.sum()), "picp": float(((y[m] >= L[m]) & (y[m] <= U[m])).mean()),
            "mpiw": float(np.maximum(U[m] - L[m], 0).mean())}


def run_fold(args):
    cell, seed, data = args
    _high_priority()
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    cells = {k: pd.DataFrame(v) for k, v in data.items()}
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_cells = [c for c in cells if c != cell]
    df_tr = pd.concat([cells[c] for c in train_cells], ignore_index=True)
    df_te = cells[cell]
    rmax_loo = int(max(cells[c]["RUL"].max() for c in train_cells))
    idx = np.arange(len(df_tr))
    np.random.shuffle(idx)
    ntr = int(0.7 * len(df_tr))
    df_train, df_cal = df_tr.iloc[idx[:ntr]].reset_index(drop=True), df_tr.iloc[idx[ntr:]].reset_index(drop=True)
    Xtr, ytr = df_train[FEATURES].values.astype(np.float32), df_train["RUL"].values.astype(np.float32)
    Xca, yca = df_cal[FEATURES].values.astype(np.float32), df_cal["RUL"].values.astype(np.float32)
    Xte, yte = df_te[FEATURES].values.astype(np.float32), df_te["RUL"].values.astype(np.float32)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xca, Xte = (Xtr - mu) / sd, (Xca - mu) / sd, (Xte - mu) / sd
    nval = max(int(0.1 * len(Xtr)), 5)
    vi = np.random.permutation(len(Xtr))[:nval]
    msk = np.ones(len(Xtr), bool)
    msk[vi] = False
    Xa, ya, Xv, yv = Xtr[msk], ytr[msk], Xtr[~msk], ytr[~msk]

    layers, prev = [], Xa.shape[1]
    for h in (128, 64, 32):
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    net = nn.Sequential(*layers, nn.Linear(prev, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt_, yt_ = torch.from_numpy(Xa), torch.from_numpy(ya)
    Xv_, yv_ = torch.from_numpy(Xv), torch.from_numpy(yv)
    best, best_state = float("inf"), None
    for _ in range(80):
        net.train()
        perm = torch.randperm(len(Xt_))
        for i in range(0, len(Xt_), 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            nn.functional.mse_loss(net(Xt_[b]).squeeze(-1), yt_[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(nn.functional.mse_loss(net(Xv_).squeeze(-1), yv_))
        if v < best:
            best, best_state = v, {k: x.clone() for k, x in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        fca = net(torch.from_numpy(Xca)).squeeze(-1).numpy().astype(float)
        fte = net(torch.from_numpy(Xte)).squeeze(-1).numpy().astype(float)
    s_cal, s_test = np.abs(yca - fca), np.abs(yte - fte)
    q = conformal_q(s_cal)
    L, U = fte - q, fte + q
    out = {"cell": cell, "seed": seed, "n_cal": int(len(Xca)), "n_test": int(len(Xte)), "q_cp": q,
           "rmax_loo": rmax_loo, "rmax_legacy": LEGACY_RMAX,
           "ks": float(ks_2samp(s_cal, s_test).statistic), "y_test_max": float(yte.max())}
    for mode, rmax in (("legacy", LEGACY_RMAX), ("loo", rmax_loo), ("noceil", None)):
        ftil_ca = np.clip(fca, 0, rmax) if rmax is not None else np.maximum(fca, 0)
        ftil_te = np.clip(fte, 0, rmax) if rmax is not None else np.maximum(fte, 0)
        qf = conformal_q(np.abs(yca - ftil_ca))
        Lp = np.maximum(L, 0.0)
        Up = U if rmax is None else np.minimum(U, rmax)
        Lf = np.maximum(ftil_te - qf, 0.0)
        Uf = ftil_te + qf if rmax is None else np.minimum(ftil_te + qf, rmax)
        hi = np.inf if rmax is None else rmax
        out[mode] = {"rmax": rmax, "q_fcp": qf, "delta_pred": float(np.mean(np.abs(fte - ftil_te))),
                     "viol": float((yte > hi).mean()),
                     "cp": metrics(yte, L, U, rmax), "pccp": metrics(yte, Lp, Up, rmax), "fcp": metrics(yte, Lf, Uf, rmax),
                     "sub": {str(tau): {"cp": sub_picp(yte, L, U, tau), "pccp": sub_picp(yte, Lp, Up, tau),
                                        "fcp": sub_picp(yte, Lf, Uf, tau)} for tau in (30, 15)}}
    out["arrays"] = {"y": yte.tolist(), "f": fte.tolist(), "L": L.tolist(), "U": U.tolist()}
    return out


if __name__ == "__main__":
    outfile, workers = sys.argv[1], int(sys.argv[2])
    nseeds = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    data, valid = {}, []
    for c in CELLS:
        df, ok = load_cell(c)
        print(c, len(df), "discharge cycles;", "EOL reached, max RUL %d" % df["RUL"].max() if ok else "censored (excluded)", flush=True)
        if ok:
            data[c] = df.to_dict(orient="list")
            valid.append(c)
    tasks = [(c, s, data) for s in range(nseeds) for c in valid]
    res, t0 = [], time.time()
    if workers <= 1:
        for t in tasks:
            res.append(run_fold(t))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(run_fold, tasks):
                res.append(r)
                if len(res) % 15 == 0:
                    print(len(res), "folds", f"{time.time() - t0:.0f}s", flush=True)
    json.dump({"cells": valid, "results": res}, open(outfile, "w"))
    print("DONE", len(res), "folds", flush=True)
