r"""Redraw Fig. 3 (coverage/width/infeasibility vs nominal level) and Fig. 4 (subgroup sweep vs RUL threshold)
for FD001 with mean +/- 1 SD bands over the seeds of Experiment A (same protocol as Table 1), and the
Fig. 5 (a-c): coverage by RUL band and by life fraction, and the distribution of engine-level coverage,
for all four subsets.

usage: python make_fig34.py OUT_EXPA OUTDIR
"""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import conformal_quantile

OUT, OUTDIR = sys.argv[1], sys.argv[2]
os.makedirs(OUTDIR, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12,
                     "axes.titleweight": "bold", "pdf.fonttype": 42})
BLUE, GREEN, RED, GREY = "#2E86AB", "#28A745", "#DC3545", "#7F7F7F"
RMAX = 125.0
NOMINAL = [0.80, 0.85, 0.90, 0.95, 0.98]
TAUS = [125, 100, 75, 50, 40, 30, 25, 20, 15]

files = sorted(glob.glob(f"{OUT}/expA_FD001_seed*_arrays.npz"))
print("FD001 seeds with arrays:", len(files))
S = [np.load(f) for f in files]

# ------------------------------------------------------------------ per-seed metrics vs nominal level
cov = np.zeros((len(S), len(NOMINAL)))
w_cp = np.zeros_like(cov)
w_pc = np.zeros_like(cov)
infe = np.zeros_like(cov)
cov_p, neg_p, ok_cp, ok_pc = (np.zeros_like(cov) for _ in range(4))     # PCCP quantities are MEASURED on the projected intervals
for i, d in enumerate(S):
    fca, yca, fte, yte = d["f_ca"].astype(float), d["y_ca"].astype(float), d["f_te"].astype(float), d["y_te"].astype(float)
    for j, nom in enumerate(NOMINAL):
        q = conformal_quantile(np.abs(yca - fca), 1 - nom)
        L, U = fte - q, fte + q
        Lp, Up = np.maximum(L, 0), np.minimum(U, RMAX)
        Lp = np.minimum(Lp, Up)
        cov[i, j] = np.mean((yte >= L) & (yte <= U))
        cov_p[i, j] = np.mean((yte >= Lp) & (yte <= Up))
        w_cp[i, j] = np.mean(U - L)
        w_pc[i, j] = np.mean(Up - Lp)
        infe[i, j] = np.mean(L < 0)
        neg_p[i, j] = np.mean(Lp < 0)
        ok_cp[i, j] = np.mean((L >= 0) & (U <= RMAX))
        ok_pc[i, j] = np.mean((Lp >= 0) & (Up <= RMAX))
print("Fig. 3: max |PICP(CP) - PICP(PCCP)| over seeds and levels =", float(np.abs(cov - cov_p).max()))
assert np.abs(cov - cov_p).max() == 0.0
x = np.array(NOMINAL) * 100
m, sd = (lambda a: a.mean(0)), (lambda a: a.std(0))


def band(ax, xs, a, color, **kw):
    ax.fill_between(xs, a.mean(0) - a.std(0), a.mean(0) + a.std(0), color=color, alpha=0.18, lw=0)


fig, axs = plt.subplots(2, 2, figsize=(11.7, 8.4))
# (a)
ax = axs[0, 0]
ax.fill_between(x, x - 2, x + 2, color="#DDDDDD", alpha=0.6, label=r"$\pm$2% tolerance band")
ax.plot(x, x, "k--", lw=2, label="Ideal (y=x)")
band(ax, x, 100 * cov, BLUE)
ax.plot(x, 100 * m(cov), "-o", color=BLUE, lw=2.5, ms=9, mec="white", label="Standard CP")
ax.plot(x, 100 * m(cov_p), "-s", color=GREEN, lw=2.0, ms=8, mec="white", label="PCCP")
ax.set_xlabel("Nominal Coverage (%)")
ax.set_ylabel("Empirical Coverage (%)")
ax.set_title("(a) Coverage Calibration")
ax.text(0.04, 0.93, "Corollary 1: CP = PCCP\n(coverage preserved)", transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="#e3f6e3", ec="#999999"))
ax.legend(loc="lower right", fontsize=8.5, frameon=True)
# (b)
ax = axs[0, 1]
band(ax, x, w_cp, BLUE)
band(ax, x, w_pc, GREEN)
ax.fill_between(x, m(w_pc), m(w_cp), color=GREEN, alpha=0.18, label="Width Reduction")
ax.plot(x, m(w_cp), "-o", color=BLUE, lw=2.5, ms=9, mec="white", label="Standard CP")
ax.plot(x, m(w_pc), "-s", color=GREEN, lw=2.5, ms=9, mec="white", label="PCCP")
for xi, a, b in zip(x, m(w_cp), m(w_pc)):
    ax.text(xi, (a + b) / 2, f"-{100 * (a - b) / a:.0f}%", ha="center", color=GREEN, fontweight="bold", fontsize=9)
ax.set_xlabel("Nominal Coverage (%)")
ax.set_ylabel("Mean Prediction Interval Width (cycles)")
ax.set_title("(b) Interval Efficiency")
ax.text(0.97, 0.06, "Corollary 4: PCCP $\\leq$ CP\n(width reduction)", transform=ax.transAxes, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="#e3f6e3", ec="#999999"))
ax.legend(loc="upper left", fontsize=8.5)
# (c)
ax = axs[1, 0]
band(ax, x, 100 * infe, RED)
ax.fill_between(x, 0, 100 * m(infe), color=RED, alpha=0.22)
ax.plot(x, 100 * m(infe), "-o", color=RED, lw=2.5, ms=9, mec="white", label="Standard CP")
ax.plot(x, 100 * m(neg_p), "-s", color=GREEN, lw=2.5, ms=9, mec="white", label="PCCP (always 0%)")
for xi, v in zip(x, 100 * m(infe)):
    ax.text(xi, v + 0.02 * 100 * m(infe).max(), f"{v:.1f}%", ha="center", color=RED, fontweight="bold", fontsize=9)
ax.set_xlabel("Nominal Coverage (%)")
ax.set_ylabel("Intervals with $L<0$ (%)")
ax.set_title("(c) Negative Lower Bounds")
ax.legend(loc="upper left", fontsize=8.5)
# (d)
ax = axs[1, 1]
i90 = NOMINAL.index(0.90)
groups = ["Coverage\n(%)", "MPIW\n(cycles)", "Fully feasible\n$\\psi_{\\mathrm{ok}}$ (%)", "Infeasible\n(%)"]   # feasible = inside [0, Rmax] (both bounds, as in Table 1)
cp_v = [100 * cov[:, i90].mean(), w_cp[:, i90].mean(), 100 * ok_cp[:, i90].mean(), 100 * (1 - ok_cp[:, i90].mean())]
pc_v = [100 * cov_p[:, i90].mean(), w_pc[:, i90].mean(), 100 * ok_pc[:, i90].mean(), 100 * (1 - ok_pc[:, i90].mean())]
scale = [100.0, max(w_cp[:, i90].mean(), 1) * 1.15, 100.0, max(cp_v[3], 1) * 1.15]
xx = np.arange(4)
ax.bar(xx - 0.2, [v / s for v, s in zip(cp_v, scale)], 0.38, color=BLUE, alpha=0.85, label="Standard CP")
ax.bar(xx + 0.2, [v / s for v, s in zip(pc_v, scale)], 0.38, color=GREEN, alpha=0.85, label="PCCP")
for i in range(4):
    ax.text(xx[i] - 0.2, cp_v[i] / scale[i] + 0.02, f"{cp_v[i]:.1f}" + ("%" if i != 1 else ""), ha="center", color=BLUE, fontweight="bold", fontsize=9)
    ax.text(xx[i] + 0.2, pc_v[i] / scale[i] + 0.02, f"{pc_v[i]:.1f}" + ("%" if i != 1 else ""), ha="center", color=GREEN, fontweight="bold", fontsize=9)
ax.set_xticks(xx)
ax.set_xticklabels(groups)
ax.set_ylabel("Normalized Value")
ax.set_ylim(0, 1.3)
ax.set_title("(d) Summary Comparison at 90% Coverage")
ax.legend(loc="upper right", fontsize=8.5)
for a_ in axs.ravel():
    a_.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/Fig_3.pdf")
plt.close(fig)

# ------------------------------------------------------------------ Fig 4: subgroup sweep on {fhat <= tau}
cov_t = np.full((len(S), len(TAUS)), np.nan)
wcp_t, wpc_t, neg_t, n_t, covp_t, negp_t = (np.full_like(cov_t, np.nan) for _ in range(6))
for i, d in enumerate(S):
    fca, yca, fte, yte = d["f_ca"].astype(float), d["y_ca"].astype(float), d["f_te"].astype(float), d["y_te"].astype(float)
    q = conformal_quantile(np.abs(yca - fca), 0.10)
    L, U = fte - q, fte + q
    Lp, Up = np.maximum(L, 0), np.minimum(U, RMAX)
    Lp = np.minimum(Lp, Up)
    for j, tau in enumerate(TAUS):
        msk = fte <= tau
        if msk.sum() >= 5:
            cov_t[i, j] = np.mean((yte[msk] >= L[msk]) & (yte[msk] <= U[msk]))
            covp_t[i, j] = np.mean((yte[msk] >= Lp[msk]) & (yte[msk] <= Up[msk]))        # measured on the projected intervals
            wcp_t[i, j] = np.mean(U[msk] - L[msk])
            wpc_t[i, j] = np.mean(Up[msk] - Lp[msk])
            neg_t[i, j] = np.mean(L[msk] < 0)
            negp_t[i, j] = np.mean(Lp[msk] < 0)
            n_t[i, j] = msk.sum()
print("Fig. 4: max |PICP(CP) - PICP(PCCP)| over seeds and thresholds =", float(np.nanmax(np.abs(cov_t - covp_t))))
assert np.nanmax(np.abs(cov_t - covp_t)) == 0.0
nm_ = lambda a: np.nanmean(a, 0)
ns_ = lambda a: np.nanstd(a, 0)
fig, axs = plt.subplots(1, 2, figsize=(13.7, 5.75))
ax = axs[0]
ax2 = ax.twinx()
ax2.spines["right"].set_visible(True)
xs = np.array(TAUS, float)
ax.fill_between(xs, 100 * (nm_(cov_t) - ns_(cov_t)), 100 * (nm_(cov_t) + ns_(cov_t)), color=GREEN, alpha=0.18, lw=0)
ax.plot(xs, 100 * nm_(cov_t), "-o", color=BLUE, lw=3, ms=8, mec="white", label="CP Coverage")
ax.plot(xs, 100 * nm_(covp_t), "--s", color=GREEN, lw=3, ms=8, mec="white", label="PCCP Coverage")
ax.axhline(90, color=GREY, ls="--", lw=1.5, label="90% Target")
ax2.fill_between(xs, nm_(wpc_t), nm_(wcp_t), color=GREEN, alpha=0.15)
ax2.plot(xs, nm_(wcp_t), "-o", color=BLUE, alpha=0.5, lw=2, ms=6, label="CP MPIW")
ax2.plot(xs, nm_(wpc_t), "-s", color=GREEN, alpha=0.5, lw=2, ms=6, label="PCCP MPIW")
ax.set_xlim(max(TAUS) + 5, min(TAUS) - 3)
ax.set_xlabel("Predicted-RUL Threshold $\\tau$ (cycles)")
ax.set_ylabel("Empirical Coverage (%)")
ax2.set_ylabel("Mean Prediction Interval Width (cycles)", color=GREY)
ax.set_title("(a) Coverage & Interval Width vs RUL Threshold")
ax.text(0.03, 0.05, "Coverage preserved\nacross all thresholds\n(Corollary 2)", transform=ax.transAxes, color=GREEN, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREEN))
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="lower center", fontsize=8, ncol=2)
ax = axs[1]
ax.axhspan(0, 25, color="#e3f2e3", alpha=0.6, label="Low Risk (<25%)")
ax.axhspan(50, 100, color="#fbdcdc", alpha=0.6, label="High Risk (>50%)")
ax.fill_between(xs, 100 * (nm_(neg_t) - ns_(neg_t)), 100 * (nm_(neg_t) + ns_(neg_t)), color=RED, alpha=0.18, lw=0)
ax.plot(xs, 100 * nm_(neg_t), "-o", color=RED, lw=3, ms=8, mec="white", label="CP Negative %")
ax.plot(xs, 100 * nm_(negp_t), "-s", color=GREEN, lw=3, ms=8, mec="white", label="PCCP Negative % (always 0)")
ax.set_xlim(max(TAUS) + 5, min(TAUS) - 3)
ax.set_ylim(-4, 104)
ax.set_xlabel("Predicted-RUL Threshold $\\tau$ (cycles)")
ax.set_ylabel("Intervals with $L<0$ (%)")
ax.set_title("(b) Negative Lower Bounds vs RUL Threshold")
ax.text(0.5, 0.20, "PCCP: 0% at ALL thresholds\n(by construction)", transform=ax.transAxes, ha="center", color=GREEN, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREEN))
ax.legend(loc="upper left", fontsize=8.5)
top = ax.secondary_xaxis("top")
TOPN = [0, 2, 4, 5, 8]      # tau = 125, 75, 40, 30, 15 (the labels of tau = 25, 20 collide with tau = 15)
top.set_xticks(xs[TOPN])
top.set_xticklabels([f"n={np.nanmean(n_t[:, j]):.0f}" for j in TOPN], fontsize=8, color=GREY)
for a_ in (axs[0], axs[1]):
    a_.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/Fig_4.pdf")
plt.close(fig)

# ------------------------------------------------------------------ stage-wise coverage (all subsets)
SUBS = ["FD001", "FD002", "FD003", "FD004"]
cols = {"FD001": "#2E86AB", "FD002": "#E07B39", "FD003": "#28A745", "FD004": "#8E44AD"}
fig, axs = plt.subplots(1, 3, figsize=(14.6, 4.0))
labels = ["<15", "15-30", "30-50", "50-75", "75-100", "100-125", "=125"]
engine_picp = {}
for s in SUBS:
    rows = [json.load(open(f))["cap125"] for f in sorted(glob.glob(f"{OUT}/expA_{s}_seed*_main.json"))]
    tr = np.sum([np.array(x["stage_rul"], float) for x in rows], axis=0)
    tl = np.sum([np.array(x["stage_life"], float) for x in rows], axis=0)
    assert np.array_equal(tr[:, 1], tr[:, 2]) and np.array_equal(tl[:, 1], tl[:, 2]), "CP and PCCP covered counts differ"   # columns: n, covered(CP), covered(PCCP), ...
    axs[0].plot(range(7), 100 * tr[:, 1] / tr[:, 0], "-o", color=cols[s], lw=2, ms=6, label=f"{s} ({len(rows)} seeds)")
    axs[1].plot(range(1, 6), 100 * tl[:, 1] / tl[:, 0], "-o", color=cols[s], lw=2, ms=6, label=s)
    # coverage of every individual test engine (its own PICP), pooled over seeds; CP and PCCP cover exactly the same points
    assert all(x["engine"]["cp"]["cov"] == x["engine"]["pccp"]["cov"] for x in rows)
    pe = np.sort(np.concatenate([np.array(x["engine"]["cp"]["cov"], float) / np.array(x["engine"]["cp"]["n"], float) for x in rows]))
    engine_picp[s] = pe
    axs[2].step(pe, np.arange(1, len(pe) + 1) / len(pe), where="post", color=cols[s], lw=2, label=f"{s} ({len(pe)} engines)")
for ax in axs[:2]:
    ax.axhline(90, color=GREY, ls="--", lw=1.3)
    ax.set_ylim(40, 102)
    ax.grid(alpha=0.25)
axs[2].axvline(0.9, color=GREY, ls="--", lw=1.3)
axs[2].axvline(0.8, color=GREY, ls=":", lw=1.3)
axs[2].set_xlim(0.3, 1.005)
axs[2].set_ylim(0, 1.02)
axs[2].grid(alpha=0.25)
axs[2].set_xlabel("Coverage of an individual test engine, PICP$_e$")
axs[2].set_ylabel("Share of engines with coverage $\\leq$ x")
axs[2].set_title("(c) Distribution of engine-level coverage")
axs[2].legend(fontsize=8.5, loc="upper left")
axs[0].set_xticks(range(7))
axs[0].set_xticklabels(labels)
axs[0].set_xlabel("True RUL band (cycles)")
axs[0].set_ylabel("Coverage (%), CP = PCCP")
axs[0].set_title("(a) Coverage by RUL band")
axs[0].legend(fontsize=8.5, loc="lower right")
axs[1].set_xticks(range(1, 6))
axs[1].set_xticklabels(["Q1\n(early)", "Q2", "Q3", "Q4", "Q5\n(late)"])
axs[1].set_xlabel("Quintile of life elapsed")
axs[1].set_title("(b) Coverage by fraction of life elapsed")
fig.tight_layout()
fig.savefig(f"{OUTDIR}/Fig_5.pdf")
plt.close(fig)
json.dump({"fig3": {"nominal": NOMINAL, "picp": m(cov).tolist(), "picp_pccp": m(cov_p).tolist(), "mpiw_cp": m(w_cp).tolist(), "mpiw_pccp": m(w_pc).tolist(),
                    "infeasible": m(infe).tolist(), "psi_ok_cp": m(ok_cp).tolist(), "psi_ok_pccp": m(ok_pc).tolist(), "n_seeds": len(S)},
           "fig4": {"taus": TAUS, "cov": nm_(cov_t).tolist(), "n": [float(np.nanmean(n_t[:, j])) for j in range(len(TAUS))]}},
          open(f"{OUTDIR}/fig34_numbers.json", "w"), indent=1)
print("figures written to", OUTDIR)
