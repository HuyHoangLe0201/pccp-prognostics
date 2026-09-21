"""Experiment C: cross-domain deployment under shift (FD001 -> FD004 severe, FD001 -> FD003 mild) with
oracle / delayed / end-of-life feedback, six constructors, per-moment and sequential maintenance
decision analyses.  Fully specified re-implementation used for Sections 5.9-5.10.

Constructors (arms 4-6 share ONE ACI recursion per feedback protocol, see Theorem 2(i)):
  CP        fixed FD001 split-conformal quantile, no projection
  PCCP-inf  ... projected onto K=[0,inf)         (non-negativity only)
  PCCP-R    ... projected onto K=[0,Rmax]        (non-negativity + label ceiling)
  ACI       adaptive level, no projection
  PCCP-inf+ACI, PCCP-R+ACI    the same ACI intervals projected

Streams per target domain
  A  truncated *test* engines, labels from RUL_FDxxx.txt (published protocol; coverage, delay, per-moment cost)
  B  complete run-to-failure engines (the domain's *training* file, labels never used except by the ACI
     feedback protocol and the cost simulation) for the unit-level sequential maintenance simulation

usage: python expC_shift.py OUTDIR WORKERS N_SEEDS
"""
import json
import os
import sys
import time
from bisect import bisect_left, insort
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

DATA = os.environ.get("CMAPSS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "CMAPSSData"))
ALPHA, GAMMA, WINDOW = 0.10, 0.03, 500
KEEP = [f"s{i}" for i in range(1, 22) if f"s{i}" not in ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]]
COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
OP = ["op1", "op2", "op3"]
BETAS = np.linspace(0, 1, 21)
COST_RATIOS = [5, 10, 20, 50, 100]
DELTAS = [1, 5, 10, 25]
DELAYS = [10, 50, 125]
GAMMAS = [0.005, 0.01, 0.03, 0.1]
CAPS = [125.0, 100.0, 150.0]
TARGETS = ["FD004", "FD003"]


def _high_priority():
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x80)
    except Exception:
        pass


# ----------------------------------------------------------------------------- data
def read(name):
    return pd.read_csv(f"{DATA}/{name}", sep=r"\s+", header=None, names=COLS)


def kmeans_norm(train_df, df, k=6):
    """Unsupervised operating-condition normalisation: cluster the 3 operating settings of the deployment
    domain's (unlabelled) training file into its k documented conditions, then min-max the 14 sensors
    within each condition using that file's statistics.  Test rows go to the nearest centroid."""
    from sklearn.cluster import KMeans
    ops_tr, ops = train_df[OP].to_numpy(), df[OP].to_numpy()
    sc = ops_tr.std(0) + 1e-9
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(ops_tr / sc)
    rtr, rte = km.labels_, km.predict(ops / sc)
    S_tr, S = train_df[KEEP].to_numpy(), df[KEEP].to_numpy()
    Xn = np.zeros(S.shape, dtype=float)
    for r in range(k):
        lo, hi = S_tr[rtr == r].min(0), S_tr[rtr == r].max(0)
        Xn[rte == r] = (S[rte == r] - lo) / np.where(hi - lo == 0, 1, hi - lo)
    return Xn.astype(np.float32), k, int(len(np.unique(rte)))


def source_norm(df):
    """No adaptation: the source (FD001) min-max scaling."""
    fd1 = read("train_FD001.txt")
    mn, mx = fd1[KEEP].min(), fd1[KEEP].max()
    return ((df[KEEP] - mn) / (mx - mn).replace(0, 1)).to_numpy(np.float32)


def target_stream(fd, kind):
    """kind='test': truncated test engines (labels from RUL_FDxxx.txt); 'train': complete run-to-failure engines."""
    tr = read(f"train_{fd}.txt")
    if kind == "test":
        df = read(f"test_{fd}.txt")
        rul = pd.read_csv(f"{DATA}/RUL_{fd}.txt", sep=r"\s+", header=None).iloc[:, 0].to_numpy(float)
        last = df.groupby("unit")["cycle"].transform("max").to_numpy()
        true_rul = rul[df["unit"].to_numpy() - 1] + last - df["cycle"].to_numpy()
        life = np.full(len(df), np.nan)
    else:
        df = tr.copy()
        last = df.groupby("unit")["cycle"].transform("max").to_numpy()
        true_rul = last - df["cycle"].to_numpy()
        life = last.astype(float)
    if fd == "FD004":                        # six operating conditions -> per-condition min-max (unsupervised)
        Xn, n_reg, n_used = kmeans_norm(tr, df, k=6)
    else:                                    # FD003 shares FD001's single operating condition: reuse FD001 scaling
        Xn, n_reg, n_used = source_norm(df), 1, 1
    order = np.lexsort((df["cycle"].to_numpy(), df["unit"].to_numpy()))
    return {"X": Xn[order], "unit": df["unit"].to_numpy()[order], "cycle": df["cycle"].to_numpy()[order].astype(float),
            "true_rul": true_rul[order].astype(float), "life": life[order], "n_regimes": n_reg, "n_used": n_used}


# ----------------------------------------------------------------------------- source model
def source_model(seed, cap):
    import torch
    from common import predict, train_mlp
    torch.set_num_threads(1)
    fd1 = read("train_FD001.txt")
    life = fd1.groupby("unit")["cycle"].transform("max")
    y = np.minimum((life - fd1["cycle"]).to_numpy(float), cap)
    mn, mx = fd1[KEEP].min(), fd1[KEEP].max()
    X = ((fd1[KEEP] - mn) / (mx - mn).replace(0, 1)).to_numpy(np.float32)
    units = np.unique(fd1["unit"].to_numpy())
    rng = np.random.RandomState(seed)
    rng.shuffle(units)
    nfit = int(0.85 * len(units))
    fit = np.isin(fd1["unit"].to_numpy(), units[:nfit])
    model = train_mlp(X[fit].astype(np.float64), y[fit], seed)
    f_cal = predict(model, X[~fit].astype(np.float64))
    scores = np.abs(y[~fit] - f_cal)
    n = len(scores)
    q = float(np.sort(scores)[min(int(np.ceil((n + 1) * (1 - ALPHA))), n) - 1])
    return model, q, scores, float(np.sqrt(np.mean((f_cal - y[~fit]) ** 2)))


# ----------------------------------------------------------------------------- ACI
def run_aci(f, y_cap, unit, seed_scores, rmax, gamma=GAMMA, delay=0, eol=False, alpha=ALPHA):
    """Standard ACI (Gibbs & Candes 2021): alpha_{t+1} = alpha_t + gamma (alpha - miss_t) with the
    miss indicator of the *projected* interval and alpha_t unconstrained; the conformal level is clipped
    to the attainable range of the window (q = largest window score when alpha_t <= 1/(m+1), so that
    intervals stay finite).  delay=d: feedback of step t is applied after step t+d; eol: all feedback of
    a unit is applied at the unit's last cycle.  Returns L, U (unprojected), alpha_t and diagnostics."""
    n = len(f)
    win = deque(seed_scores[-WINDOW:])
    srt = sorted(win)
    a = alpha
    L = np.empty(n)
    U = np.empty(n)
    A = np.empty(n)
    pending = deque()
    buf = {}
    n_sat = 0

    def push(s):
        win.append(s)
        insort(srt, s)
        if len(win) > WINDOW:
            old = win.popleft()
            srt.pop(bisect_left(srt, old))

    for t in range(n):
        m = len(srt)
        k = int(np.ceil((m + 1) * (1 - a)))
        if k > m:
            n_sat += 1
        q = srt[min(max(k, 1), m) - 1]
        L[t], U[t], A[t] = f[t] - q, f[t] + q, a
        s_t = abs(y_cap[t] - f[t])
        miss = not (max(L[t], 0.0) <= y_cap[t] <= min(U[t], rmax))
        if eol:
            buf.setdefault(unit[t], []).append((s_t, miss))
            if t == n - 1 or unit[t + 1] != unit[t]:
                for s_, m_ in buf.pop(unit[t]):
                    a += gamma * (alpha - float(m_))
                    push(s_)
        elif delay == 0:
            a += gamma * (alpha - float(miss))
            push(s_t)
        else:
            pending.append((s_t, miss))
            if len(pending) > delay:
                s_, m_ = pending.popleft()
                a += gamma * (alpha - float(m_))
                push(s_)
    return L, U, A, {"frac_level_saturated": n_sat / n, "alpha_min": float(A.min()), "alpha_max": float(A.max())}


# ----------------------------------------------------------------------------- metrics
def proj(L, U, rmax, hi_inf=False):
    return np.maximum(L, 0.0), (U if hi_inf else np.minimum(U, rmax))


def arm_metrics(L, U, y, rmax):
    cov = (y >= L) & (y <= U)
    w = np.maximum(U - L, 0.0)
    feas = float(np.mean((L >= 0) & (U <= rmax)))
    isc = float(np.mean(w + (2 / ALPHA) * (np.maximum(L - y, 0.0) + np.maximum(y - U, 0.0))))
    return {"picp": float(cov.mean()), "mpiw": float(w.mean()), "feas": feas, "iscore": isc, "empty": float(np.mean(L > U))}


def per_moment_cost(L, U, r, cf):
    best = np.inf
    for b in BETAS:
        d = np.maximum(L, 0) + b * (U - L)
        best = min(best, float(np.where(d <= r, r - d, cf).mean()))
    return best


def sequential(L, U, unit, cycle, life, cf_list, delta):
    """Unit-level renewal-reward simulation on complete run-to-failure engines.
    Inspect at ages 1, 1+delta, ...; replace (cost 1) at the first inspection whose planned lead
    d(beta) = max(L,0) + beta (U-L) is <= delta; otherwise the unit fails at its last cycle (cost cf).
    Per cf: the cost rate per operating cycle minimised over beta (with failure rate, mean wasted life)
    and the beta=0 (lower bound only) rate."""
    us, starts = np.unique(unit, return_index=True)
    ends = np.r_[starts[1:], len(unit)]
    nb = len(BETAS)
    n_rep = np.zeros(nb)
    tot_len = np.zeros(nb)
    waste = np.zeros(nb)
    for s, e in zip(starts, ends):
        Lc, W, age, T = np.maximum(L[s:e], 0), np.maximum(U[s:e] - L[s:e], 0), cycle[s:e], life[s]
        ok = ((age - 1) % delta == 0) & (age < T)
        D = Lc[None, :] + BETAS[:, None] * W[None, :]
        hit = (D <= delta) & ok[None, :]
        any_hit = hit.any(axis=1)
        first = hit.argmax(axis=1)
        a_rep = age[first]
        n_rep += any_hit
        tot_len += np.where(any_hit, a_rep, T)
        waste += np.where(any_hit, T - a_rep, 0.0)
    nunits = len(us)
    out = {}
    for cf in cf_list:
        rate = (n_rep * 1.0 + (nunits - n_rep) * cf) / tot_len
        j = int(np.argmin(rate))
        out[str(cf)] = {"rate": float(rate[j]), "beta": float(BETAS[j]), "fail_rate": float((nunits - n_rep[j]) / nunits),
                        "waste": float(waste[j] / max(n_rep[j], 1)), "rate_beta0": float(rate[0]),
                        "fail_rate_beta0": float((nunits - n_rep[0]) / nunits)}
    return out


# ----------------------------------------------------------------------------- analyses
def analysis_B(model, SB, q, cap, seed_scores):
    """Sequential maintenance simulation on complete engines (all six constructors, oracle and end-of-life feedback)."""
    from common import predict
    fB = np.clip(predict(model, SB["X"].astype(np.float64)), 0.0, cap)
    yB = np.minimum(SB["true_rul"], cap)
    out = {"rmse": float(np.sqrt(np.mean((fB - yB) ** 2))), "n": int(len(fB)), "n_units": int(len(np.unique(SB["unit"])))}
    seq = {}
    for prot, kw in (("oracle", {}), ("eol", {"eol": True})):
        L, U, A, dg = run_aci(fB, yB, SB["unit"], seed_scores, cap, **kw)
        P = {"CP": (fB - q, fB + q), "ACI": (L, U)}
        for nm in ("CP", "ACI"):
            Lr, Ur = P[nm]
            P[f"PCCP-inf+{nm}" if nm != "CP" else "PCCP-inf"] = proj(Lr, Ur, cap, hi_inf=True)
            P[f"PCCP-R+{nm}" if nm != "CP" else "PCCP-R"] = proj(Lr, Ur, cap)
        seq[prot] = {nm: {str(d): sequential(Lx, Ux, SB["unit"], SB["cycle"], SB["life"], COST_RATIOS, d) for d in DELTAS}
                     for nm, (Lx, Ux) in P.items()}
        seq[prot]["_diag"] = dg
    out["sequential"] = seq
    return out


def analysis_A(model, SA, q, cap, seed_scores, save_arrays=None):
    """Coverage / delay / gamma / end-of-life / per-moment cost analyses on the truncated test stream (cap 125 only)."""
    from common import predict
    f_raw = predict(model, SA["X"].astype(np.float64))
    f = np.clip(f_raw, 0.0, cap)
    y, y_unc, u = np.minimum(SA["true_rul"], cap), SA["true_rul"], SA["unit"]
    out = {"rmse": float(np.sqrt(np.mean((f - y) ** 2))), "share_at_cap": float(np.mean(y >= cap)), "n": int(len(f)),
           "n_units": int(len(np.unique(u))),
           "empty_raw_predictor": float(np.mean((f_raw + q < 0) | (f_raw - q > cap))),
           "share_pred_clipped": float(np.mean((f_raw < 0) | (f_raw > cap)))}
    protocols = {"oracle": {}, "eol": {"eol": True}}
    for d in DELAYS:
        protocols[f"delay{d}"] = {"delay": d}
    for g in GAMMAS:
        if g != GAMMA:
            protocols[f"gamma{g}"] = {"gamma": g}
    arms, diag = {"CP": (f - q, f + q)}, {}
    for name, kw in protocols.items():
        L, U, A, dg = run_aci(f, y, u, seed_scores, cap, **kw)
        arms[f"ACI[{name}]"] = (L, U)
        diag[name] = dg
        if name == "oracle":
            A_oracle = A
    table = {"CP": arm_metrics(*arms["CP"], y, cap),
             "PCCP-inf": arm_metrics(*proj(*arms["CP"], cap, hi_inf=True), y, cap),
             "PCCP-R": arm_metrics(*proj(*arms["CP"], cap), y, cap)}
    for name in protocols:
        Lr, Ur = arms[f"ACI[{name}]"]
        table[f"ACI[{name}]"] = arm_metrics(Lr, Ur, y, cap)
        table[f"PCCP-inf+ACI[{name}]"] = arm_metrics(*proj(Lr, Ur, cap, hi_inf=True), y, cap)
        table[f"PCCP-R+ACI[{name}]"] = arm_metrics(*proj(Lr, Ur, cap), y, cap)
    out["table"], out["diag"] = table, diag
    Lr, Ur = arms["ACI[oracle]"]
    Lp, Up = proj(Lr, Ur, cap)
    out["invariance_miss_mismatch"] = int(np.sum(((y >= Lr) & (y <= Ur)) != ((y >= Lp) & (y <= Up))))
    base = {"CP": arms["CP"], "PCCP-inf": proj(*arms["CP"], cap, hi_inf=True), "PCCP-R": proj(*arms["CP"], cap),
            "ACI": (Lr, Ur), "PCCP-inf+ACI": proj(Lr, Ur, cap, hi_inf=True), "PCCP-R+ACI": (Lp, Up)}
    out["per_moment"] = {str(cf): {nm: {"capped": per_moment_cost(*lu, y, cf), "uncapped": per_moment_cost(*lu, y_unc, cf)}
                                   for nm, lu in base.items()} for cf in (20, 50, 100)}
    w = 400
    c = np.cumsum(np.insert((~((y >= arms["CP"][0]) & (y <= arms["CP"][1]))).astype(float), 0, 0))
    roll_miss = (c[w:] - c[:-w]) / w
    c2 = np.cumsum(np.insert(np.maximum(0, -Lr) + np.maximum(0, Ur - cap), 0, 0))
    roll_mass = (c2[w:] - c2[:-w]) / w
    out["pearson_mass_vs_shift"] = float(np.corrcoef(roll_miss, roll_mass)[0, 1])
    out["mass_range"] = [float(roll_mass.min()), float(roll_mass.max())]
    out["inspect_theta80"] = {"ACI": float(np.mean((Ur - Lr) > 80)), "PCCP-R+ACI": float(np.mean((Up - Lp) > 80)),
                              "CP": float(np.mean((arms["CP"][1] - arms["CP"][0]) > 80)),
                              "PCCP-R": float(np.mean(np.maximum(np.minimum(arms["CP"][1], cap) - np.maximum(arms["CP"][0], 0), 0) > 80))}
    out["share_capped_upper_CP"] = float(np.mean(arms["CP"][1] > cap))
    if save_arrays:
        np.savez_compressed(save_arrays, y=y, y_unc=y_unc, f=f, unit=u, cycle=SA["cycle"], Lcp=arms["CP"][0], Ucp=arms["CP"][1],
                            L_aci=Lr, U_aci=Ur, A=A_oracle, L_eol=arms["ACI[eol]"][0], U_eol=arms["ACI[eol]"][1],
                            L_d50=arms["ACI[delay50]"][0], U_d50=arms["ACI[delay50]"][1], q=q, cap=cap)
    return out


# ----------------------------------------------------------------------------- one seed
def run_seed(args):
    seed, outdir = args
    _high_priority()
    path = f"{outdir}/expC_seed{seed}.json"
    if os.path.exists(path):
        return path
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    streams = {fd: {"A": target_stream(fd, "test"), "B": target_stream(fd, "train")} for fd in TARGETS}
    res = {"seed": seed, "targets": {fd: {"n_A": int(len(s["A"]["X"])), "n_B": int(len(s["B"]["X"])),
                                          "n_units_A": int(len(np.unique(s["A"]["unit"]))),
                                          "n_units_B": int(len(np.unique(s["B"]["unit"]))),
                                          "n_regimes": s["A"]["n_regimes"], "caps": {}} for fd, s in streams.items()}}
    for cap in CAPS:
        model, q, cal_scores, rmse_src = source_model(seed, cap)
        rng = np.random.RandomState(seed)
        seed_scores = rng.choice(cal_scores, size=min(WINDOW, len(cal_scores)), replace=False)
        for fd, S in streams.items():
            cres = {"q": q, "rmse_source": rmse_src, "n_cal": int(len(cal_scores)),
                    "B": analysis_B(model, S["B"], q, cap, seed_scores)}
            if cap == 125.0:
                cres["A"] = analysis_A(model, S["A"], q, cap, seed_scores,
                                       save_arrays=(f"{outdir}/expC_seed0_{fd}_arrays.npz" if seed == 0 else None))
            res["targets"][fd]["caps"][str(int(cap))] = cres
        print(f"seed {seed} cap {int(cap)} done {time.time() - t0:.0f}s", flush=True)
    res["seconds"] = time.time() - t0
    with open(path, "w") as fh:
        json.dump(res, fh)
    return path


if __name__ == "__main__":
    outdir, workers, nseeds = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    os.makedirs(outdir, exist_ok=True)
    tasks = [(s, outdir) for s in range(nseeds)]
    t0 = time.time()
    if workers <= 1:
        for t in tasks:
            print(run_seed(t), f"{time.time() - t0:.0f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for p in ex.map(run_seed, tasks):
                print(p, f"{time.time() - t0:.0f}s", flush=True)
    print("ALL DONE", flush=True)
