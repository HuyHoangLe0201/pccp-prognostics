r"""Fig. 2 of the manuscript: qualitative CP vs PCCP comparison on FD001, drawn with the panel design of the authors' figure
notebook (PCCP_Full_Figures*.ipynb) but from seed 0 of the protocol of Section 4.3 (single-cycle MLP, engine-level 70/15/15 split).

* panels (a)-(d): one held-out TEST engine (the one whose life is closest to the median life of the 15 test engines);
* panels (e)-(f): all test samples with RUL < 30 of the 15 test engines.

"Infeasible" means: not contained in [0, Rmax] (the definition of psi_ok in Table 1); the share with L < 0 and with U > Rmax is
reported separately.  usage: python make_fig2.py EXPA_DIR OUTDIR
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

EXPA, OUTDIR = sys.argv[1], sys.argv[2]
os.makedirs(OUTDIR, exist_ok=True)
RMAX = 125.0
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "pdf.fonttype": 42})
CP, PC, INF, NEU = "#2E86AB", "#28A745", "#DC3545", "#6C757D"

z = np.load(f"{EXPA}/expA_FD001_seed0_arrays.npz")
q = json.load(open(f"{EXPA}/expA_FD001_seed0_main.json"))["cap125"]["q"]
ute, age, life, y_all, f_all = z["ute"], z["age_te"].astype(float), z["life_te"].astype(float), z["y_te"].astype(float), z["f_te"].astype(float)
engines = np.unique(ute)
lives = {int(e): float(life[ute == e][0]) for e in engines}
med = float(np.median(list(lives.values())))
eng = min(lives, key=lambda e: (abs(lives[e] - med), e))
m = ute == eng
o = np.argsort(age[m])
t, y_true, pred = age[m][o], y_all[m][o], f_all[m][o]
lo_cp, up_cp = pred - q, pred + q
lo_pc, up_pc = np.maximum(lo_cp, 0), np.minimum(up_cp, RMAX)
crit = y_true < 30
t_c, y_c, pred_c = t[crit], y_true[crit], pred[crit]
lo_cp_c, up_cp_c, lo_pc_c, up_pc_c = lo_cp[crit], up_cp[crit], lo_pc[crit], up_pc[crit]

# all test samples with RUL < 30 (15 test engines)
cm = y_all < 30
y_t, pred_t = y_all[cm], f_all[cm]
lo_t, up_t = pred_t - q, pred_t + q
lo_tp, up_tp = np.maximum(lo_t, 0), np.minimum(up_t, RMAX)

neg_full = lo_cp < 0
pos_full = up_cp > RMAX
inf_full = neg_full | pos_full
neg_crit = lo_cp_c < 0
inf_test = lo_t < 0            # (U < Rmax for all of these samples)

fig = plt.figure(figsize=(11.7, 13.4))
gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0.34, wspace=0.24)
ylim_full = (min(lo_cp.min() - 15, -40), max(up_cp.max() + 10, 150))
ylim_crit = (min(lo_cp_c.min() - 12, -35), max(up_cp_c.max() + 5, 60))


def box(ax, txt, color, x=0.03, y=0.03, ha="left", fs=9):
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=fs, color=color, fontweight="bold", va="bottom", ha=ha,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=color, alpha=0.95))


# (a) ------------------------------------------------------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
ax.fill_between(t, lo_cp, up_cp, alpha=0.25, color=CP, label="90% Prediction Interval")
ax.plot(t, pred, color=CP, lw=2, label="Point Prediction $\\hat{y}$")
ax.plot(t, y_true, color=NEU, lw=2.5, ls="--", label="True RUL")
ax.axhline(0, color=INF, lw=2, ls=":", label="Lower bound ($y=0$)")
ax.axhline(RMAX, color=NEU, lw=1.2, ls=":", alpha=0.8)
ax.fill_between(t, lo_cp, 0, where=neg_full, alpha=0.5, color=INF, hatch="///", label="Infeasible Region")
ax.axvspan(t_c.min(), t_c.max(), alpha=0.08, color="orange")
ax.annotate("Prediction interval\nextends below zero", xy=(t[neg_full][len(t[neg_full]) // 2], lo_cp[neg_full].mean()), xytext=(t[int(len(t) * 0.70)], -30),
            fontsize=8.5, ha="center", arrowprops=dict(arrowstyle="->", color=INF, lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=INF, alpha=0.9))
ax.set_xlabel("Time (cycles)")
ax.set_ylabel("RUL (cycles)")
ax.set_title("(a) Standard CP - Full Degradation Trajectory", fontsize=11, fontweight="bold")
ax.legend(loc="upper right", fontsize=7.5, framealpha=0.95)
ax.set_ylim(*ylim_full)
ax.grid(alpha=0.3)
box(ax, f"Infeasible: {100 * inf_full.mean():.1f}%\n(L<0: {100 * neg_full.mean():.1f}%, U>$R_{{max}}$: {100 * pos_full.mean():.1f}%)\nMin lower bound: {lo_cp.min():.1f}", INF, fs=8.5)

# (b) ------------------------------------------------------------------------------------------------
ax = fig.add_subplot(gs[0, 1])
ax.fill_between(t, lo_pc, up_pc, alpha=0.25, color=PC, label="90% Prediction Interval")
ax.plot(t, np.clip(pred, 0, RMAX), color=PC, lw=2, label="Point Prediction $\\hat{y}$")
ax.plot(t, y_true, color=NEU, lw=2.5, ls="--", label="True RUL")
ax.axhline(0, color=NEU, lw=1.5, alpha=0.5, label="Lower bound ($y=0$)")
ax.axhline(RMAX, color=NEU, lw=1.2, ls=":", alpha=0.8)
ax.axvspan(t_c.min(), t_c.max(), alpha=0.08, color="orange")
ax.annotate("All intervals\nconstrained to $[0, R_{max}]$", xy=(t[-10], lo_pc[-10]), xytext=(t[int(len(t) * 0.65)], -20), fontsize=8.5, ha="center",
            arrowprops=dict(arrowstyle="->", color=PC, lw=1.5), bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=PC, alpha=0.9))
ax.set_xlabel("Time (cycles)")
ax.set_ylabel("RUL (cycles)")
ax.set_title("(b) PCCP - Full Degradation Trajectory", fontsize=11, fontweight="bold")
ax.legend(loc="upper right", fontsize=7.5, framealpha=0.95)
ax.set_ylim(*ylim_full)
ax.grid(alpha=0.3)
box(ax, f"Infeasible: 0.0%\nMin lower bound: {lo_pc.min():.1f}", PC, fs=8.5)

# (c) ------------------------------------------------------------------------------------------------
ax = fig.add_subplot(gs[1, 0])
ax.fill_between(t_c, lo_cp_c, up_cp_c, alpha=0.25, color=CP, label="90% Prediction Interval")
ax.plot(t_c, pred_c, color=CP, lw=2, label="Point Prediction")
ax.plot(t_c, y_c, color=NEU, lw=2.5, ls="--", label="True RUL")
ax.axhline(0, color=INF, lw=2.5, label="Lower bound ($y=0$)")
if neg_crit.any():
    ax.fill_between(t_c, lo_cp_c, 0, where=neg_crit, alpha=0.5, color=INF, hatch="///")
    ax.scatter(t_c[neg_crit], lo_cp_c[neg_crit], color=INF, s=45, zorder=5, marker="x", linewidth=2, label="Infeasible Points")
    w = np.argmin(lo_cp_c)
    ax.annotate(f"Worst violation:\n{lo_cp_c[w]:.1f} cycles", xy=(t_c[w], lo_cp_c[w]), xytext=(t_c[w] - 5, lo_cp_c[w] - 8), fontsize=8.5, ha="center",
                arrowprops=dict(arrowstyle="->", color=INF, lw=1.5), bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=INF))
ax.set_xlabel("Time (cycles)")
ax.set_ylabel("RUL (cycles)")
ax.set_title("(c) Standard CP - Critical Region Detail (RUL < 30)", fontsize=11, fontweight="bold")
ax.legend(loc="upper right", fontsize=7.5, framealpha=0.95)
ax.set_ylim(*ylim_crit)
ax.grid(alpha=0.3)
box(ax, f"Infeasible: {100 * neg_crit.mean():.1f}%\nMin lower: {lo_cp_c.min():.1f}", INF, fs=8.5)

# (d) ------------------------------------------------------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
ax.fill_between(t_c, lo_pc_c, up_pc_c, alpha=0.25, color=PC, label="90% Prediction Interval")
ax.plot(t_c, np.clip(pred_c, 0, RMAX), color=PC, lw=2, label="Point Prediction")
ax.plot(t_c, y_c, color=NEU, lw=2.5, ls="--", label="True RUL")
ax.axhline(0, color=NEU, lw=1.5, alpha=0.5, label="Lower bound ($y=0$)")
ax.fill_between(t_c, 0, lo_pc_c, alpha=0.15, color=PC, label="Feasible Lower Bound")
ax.annotate("Lower bound\nprojected to 0", xy=(t_c[-5], 0), xytext=(t_c[len(t_c) // 2], -18), fontsize=8.5, ha="center",
            arrowprops=dict(arrowstyle="->", color=PC, lw=1.5), bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=PC))
ax.set_xlabel("Time (cycles)")
ax.set_ylabel("RUL (cycles)")
ax.set_title("(d) PCCP - Critical Region Detail (RUL < 30)", fontsize=11, fontweight="bold")
ax.legend(loc="upper right", fontsize=7.5, framealpha=0.95)
ax.set_ylim(*ylim_crit)
ax.grid(alpha=0.3)
box(ax, f"Infeasible: 0.0%\nMin lower: {lo_pc_c.min():.1f}", PC, fs=8.5)

# (e) ------------------------------------------------------------------------------------------------
ax = fig.add_subplot(gs[2, 0])
for i in range(0, len(y_t), 2):
    ax.plot([y_t[i], y_t[i]], [lo_t[i], up_t[i]], color=INF if inf_test[i] else CP, alpha=0.6 if inf_test[i] else 0.25, lw=0.8)
ax.scatter(y_t[~inf_test], pred_t[~inf_test], c=CP, s=18, alpha=0.5, label="Feasible Intervals")
ax.scatter(y_t[inf_test], pred_t[inf_test], c=INF, s=28, alpha=0.8, marker="x", linewidth=1.5, label="Infeasible Intervals (lower < 0)")
ax.plot([0, 30], [0, 30], "k--", lw=2, label="Perfect Prediction")
ax.axhline(0, color=INF, ls=":", lw=1.5, alpha=0.7)
ax.set_xlabel("True RUL (cycles)")
ax.set_ylabel("Predicted RUL (cycles)")
ax.set_title(f"(e) Standard CP - All Test Samples (n={len(y_t)})", fontsize=11, fontweight="bold")
ax.legend(loc="upper left", fontsize=7.5, framealpha=0.95)
ax.set_xlim(-2, 32)
ax.set_ylim(-45, 60)
ax.grid(alpha=0.3)
cov_cp = 100 * ((y_t >= lo_t) & (y_t <= up_t)).mean()
ax.text(5, -37, f"{100 * inf_test.mean():.0f}% of intervals\nhave lower < 0", fontsize=8.5, ha="center", color=INF, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=INF, alpha=0.9))
ax.text(0.97, 0.03, f"Coverage: {cov_cp:.1f}%\nInfeasible: {100 * inf_test.mean():.1f}%\nMin lower: {lo_t.min():.1f}", transform=ax.transAxes, fontsize=8.5, ha="right",
        va="bottom", bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=CP, alpha=0.95))

# (f) ------------------------------------------------------------------------------------------------
ax = fig.add_subplot(gs[2, 1])
for i in range(0, len(y_t), 2):
    ax.plot([y_t[i], y_t[i]], [lo_tp[i], up_tp[i]], color=PC, alpha=0.25, lw=0.8)
ax.scatter(y_t, pred_t, c=PC, s=18, alpha=0.5, label="All Intervals Feasible")
ax.plot([0, 30], [0, 30], "k--", lw=2, label="Perfect Prediction")
ax.axhline(0, color=NEU, lw=1, alpha=0.5)
ax.set_xlabel("True RUL (cycles)")
ax.set_ylabel("Predicted RUL (cycles)")
ax.set_title(f"(f) PCCP - All Test Samples (n={len(y_t)})", fontsize=11, fontweight="bold")
ax.legend(loc="upper left", fontsize=7.5, framealpha=0.95)
ax.set_xlim(-2, 32)
ax.set_ylim(-45, 60)
ax.grid(alpha=0.3)
cov_pc = 100 * ((y_t >= lo_tp) & (y_t <= up_tp)).mean()
ax.text(5, -37, "100% of intervals\nare feasible", fontsize=8.5, ha="center", color=PC, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=PC, alpha=0.9))
ax.text(0.97, 0.03, f"Coverage: {cov_pc:.1f}%\nInfeasible: 0.0%\nMin lower: {lo_tp.min():.1f}", transform=ax.transAxes, fontsize=8.5, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=PC, alpha=0.95))

fig.tight_layout()
fig.savefig(f"{OUTDIR}/Fig_2.pdf")
plt.close(fig)
nums = {"engine": int(eng), "life": lives[eng], "n_test_engines": int(len(engines)), "q": q, "n_traj": int(len(t)),
        "traj_infeasible_pct": 100 * float(inf_full.mean()), "traj_neg_pct": 100 * float(neg_full.mean()), "traj_pos_pct": 100 * float(pos_full.mean()),
        "traj_min_lower": float(lo_cp.min()), "crit_n": int(crit.sum()), "crit_infeasible_pct": 100 * float(neg_crit.mean()),
        "crit_min_lower": float(lo_cp_c.min()), "scatter_n": int(len(y_t)), "scatter_infeasible_pct": 100 * float(inf_test.mean()),
        "scatter_min_lower": float(lo_t.min()), "scatter_cov_cp": cov_cp, "scatter_cov_pccp": cov_pc}
json.dump(nums, open(f"{OUTDIR}/fig2_numbers.json", "w"), indent=1)
print(json.dumps(nums))
