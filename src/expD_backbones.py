"""Experiment D: PCCP on top of other uncertainty-quantification backbones under the manuscript's own protocol.

Does the projection still help when the base interval is already adaptive and asymmetric (CQR) or comes from a
deep/heteroscedastic construction? Section 5.7 first answers this on a windowed BatchNorm/Dropout architecture and a
single split (Table 10). Here (Table 9) the same constructions are built on the *single-cycle MLP of Section 4.3* (128-64-32, Adam 1e-3, batch 256, 80 epochs, engine-level
70/15/15 split, alpha = 0.1, label cap 125, the splits of Experiment A), over several seeds and all four subsets:

  * lw   : locally weighted split CP  -- heteroscedastic Gaussian-NLL network (mean, log-variance), score |y - mu| / sigma,
           interval mu +- qhat * sigma                                        (heteroscedastic conformal baseline)
  * mcd  : MC dropout (p = 0.2, 50 stochastic passes), raw interval mean +- 1.645 sd, no calibration  (deep UQ baseline)
  * qr   : the raw 5% / 95% outputs of a two-output quantile network trained with the pinball loss
  * cqr  : the same network, conformalized (Romano et al. 2019)              (adaptive, asymmetric baseline)

The plain split-CP row is taken from the arrays of Experiment A (same seeds, same splits), see agg_expD.py.
For every base interval [L, U]: PICP, MPIW, feasibility rates, unit-level coverage, and the same quantities after
projection onto K = [0, inf) and K = [0, Rmax].

usage: python expD_backbones.py OUTDIR WORKERS PLAN        (PLAN like FD001:12,FD002:6,FD003:12,FD004:6)
"""
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import (ALPHA, BATCH, LR, MLP, N_EPOCHS, conformal_quantile, cov_width, load_cmapss, project,
                    split_by_engine)

CAP = 125.0
EPOCHS = int(os.environ.get("EXPD_EPOCHS", N_EPOCHS))
MC_PASSES, MC_DROP = 50, 0.2
Z90 = 1.6448536269514722


def _f(x):
    return float(x)


def _high_priority():
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x80)
    except Exception:
        pass


def _fit(net, X, y, seed, loss_of):
    """Adam, batch 256, EPOCHS epochs, mini-batches from a fresh permutation each epoch (as common.train_mlp)."""
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xt, yt = torch.from_numpy(X).float(), torch.from_numpy(y).float()
    n = Xt.shape[0]
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    net.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss_of(net(Xt[idx]), yt[idx]).backward()
            opt.step()
    net.eval()
    return net


def train_quantile(X, y, seed):
    import torch
    torch.manual_seed(seed)
    net = MLP(X.shape[1], out=2)
    taus = torch.tensor([ALPHA / 2, 1 - ALPHA / 2])

    def pinball(o, yb):
        e = yb[:, None] - o
        return torch.mean(torch.maximum(taus * e, (taus - 1) * e))
    return _fit(net, X, y, seed, pinball)


def train_hetero(X, y, seed):
    import torch
    torch.manual_seed(seed)
    net = MLP(X.shape[1], out=2)
    net.net[-1].bias.data[1] = math.log(30.0 ** 2)          # start from a sensible spread (about 30 cycles)

    def nll(o, yb):
        logvar = torch.clamp(o[:, 1], -4.0, 12.0)
        return torch.mean(0.5 * (logvar + (yb - o[:, 0]) ** 2 * torch.exp(-logvar)))
    return _fit(net, X, y, seed, nll)


def train_dropout(X, y, seed):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    layers, prev = [], X.shape[1]
    for h in (128, 64, 32):
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(MC_DROP)]
        prev = h
    layers.append(nn.Linear(prev, 1))
    net = nn.Sequential(*layers)
    mse = nn.MSELoss()
    return _fit(net, X, y, seed, lambda o, yb: mse(o.squeeze(-1), yb))


def predict_np(net, X):
    import torch
    with torch.no_grad():
        return net(torch.from_numpy(X).float()).numpy().astype(np.float64)


def mc_predict(net, X, passes=MC_PASSES):
    import torch
    net.train()                                             # dropout active, no other stochastic layers
    Xt = torch.from_numpy(X).float()
    with torch.no_grad():
        S = np.stack([net(Xt).squeeze(-1).numpy().astype(np.float64) for _ in range(passes)])
    net.eval()
    return S.mean(0), S.std(0, ddof=1)


def rows_for(L, U, y, ute):
    """base + projected summaries of one interval construction (cap = CAP).

    The projection follows the manuscript's convention: C intersected with K is the empty set when the lower endpoint exceeds
    the upper one, which counts as a miss of zero width (no boundary guard such as L <- min(L, U), which would let a degenerate
    interval at the boundary cover y = 0 or y = Rmax and would break the pointwise identity of Lemma 1)."""
    L, U = np.minimum(L, U), np.maximum(L, U)
    cov, w = cov_width(L, U, y)
    out = {"picp": _f(cov.mean()), "mpiw": _f(w.mean()), "psi_minus": _f((L < 0).mean()), "psi_plus": _f((U > CAP).mean()),
           "psi_ok": _f(((L >= 0) & (U <= CAP)).mean())}
    ec = np.array([cov[ute == u].mean() for u in np.unique(ute)])
    out["engine"] = {"min": _f(ec.min()), "p10": _f(np.percentile(ec, 10)), "median": _f(np.median(ec)),
                     "frac_below_0.8": _f((ec < 0.8).mean())}
    for name, hi in (("K0inf", None), ("Kcap", CAP)):
        Lp = np.maximum(L, 0.0)
        Up = U if hi is None else np.minimum(U, hi)
        covp, wp = cov_width(Lp, Up, y)
        out[name] = {"picp": _f(covp.mean()), "mpiw": _f(wp.mean()), "empty": _f((Lp > Up).mean())}
    return out


def run_seed(args):
    subset, seed, outdir = args
    _high_priority()
    path = f"{outdir}/expD_{subset}_seed{seed}.json"
    if os.path.exists(path):
        return path
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    D = load_cmapss(subset)
    X = D["X"]
    tr, ca, te = split_by_engine(D["eid"], seed)
    y = np.minimum(D["rul_true"], CAP)
    ute = D["eid"][te]
    res = {"subset": subset, "seed": seed, "n_test": int(te.sum()), "epochs": EPOCHS}

    # --- quantile network: QR (raw) and CQR
    qnet = train_quantile(X[tr], y[tr], seed + 1000)
    qca, qte = predict_np(qnet, X[ca]), predict_np(qnet, X[te])
    lo_ca, hi_ca = np.minimum(qca[:, 0], qca[:, 1]), np.maximum(qca[:, 0], qca[:, 1])
    lo_te, hi_te = np.minimum(qte[:, 0], qte[:, 1]), np.maximum(qte[:, 0], qte[:, 1])
    res["qr"] = rows_for(lo_te, hi_te, y[te], ute)
    Q = conformal_quantile(np.maximum(lo_ca - y[ca], y[ca] - hi_ca), ALPHA)
    res["cqr"] = rows_for(lo_te - Q, hi_te + Q, y[te], ute)
    res["cqr"]["Q"] = _f(Q)

    # --- heteroscedastic network: locally weighted split CP
    hnet = train_hetero(X[tr], y[tr], seed + 2000)
    hca, hte = predict_np(hnet, X[ca]), predict_np(hnet, X[te])
    sig_ca = np.exp(0.5 * np.clip(hca[:, 1], -4.0, 12.0))
    sig_te = np.exp(0.5 * np.clip(hte[:, 1], -4.0, 12.0))
    qh = conformal_quantile(np.abs(y[ca] - hca[:, 0]) / sig_ca, ALPHA)
    res["lw"] = rows_for(hte[:, 0] - qh * sig_te, hte[:, 0] + qh * sig_te, y[te], ute)
    res["lw"]["qhat"] = _f(qh)
    res["lw"]["rmse"] = _f(np.sqrt(np.mean((hte[:, 0] - y[te]) ** 2)))
    res["lw"]["mean_sigma"] = _f(sig_te.mean())

    # --- MC dropout, raw interval
    dnet = train_dropout(X[tr], y[tr], seed + 3000)
    m_te, s_te = mc_predict(dnet, X[te])
    res["mcd"] = rows_for(m_te - Z90 * s_te, m_te + Z90 * s_te, y[te], ute)
    res["mcd"]["rmse"] = _f(np.sqrt(np.mean((m_te - y[te]) ** 2)))
    res["mcd"]["mean_sd"] = _f(s_te.mean())

    # the intervals themselves, so that other conventions / summaries can be recomputed without retraining
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    np.savez_compressed(f"{outdir}/expD_{subset}_seed{seed}_arrays.npz", y=f32(y[te]), ute=ute, age=f32(D["age"][te]),
                        L_qr=f32(lo_te), U_qr=f32(hi_te), L_cqr=f32(lo_te - Q), U_cqr=f32(hi_te + Q),
                        L_lw=f32(hte[:, 0] - qh * sig_te), U_lw=f32(hte[:, 0] + qh * sig_te),
                        L_mcd=f32(m_te - Z90 * s_te), U_mcd=f32(m_te + Z90 * s_te))
    res["seconds"] = time.time() - t0
    with open(path, "w") as fh:
        json.dump(res, fh)
    return path


if __name__ == "__main__":
    outdir, workers = sys.argv[1], int(sys.argv[2])
    plan = {kv.split(":")[0]: int(kv.split(":")[1]) for kv in sys.argv[3].split(",")}
    os.makedirs(outdir, exist_ok=True)
    order = ["FD004", "FD002", "FD003", "FD001"]            # heaviest first within a seed
    tasks = [(sub, s, outdir) for s in range(max(plan.values())) for sub in order if s < plan.get(sub, 0)]
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
