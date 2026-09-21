r"""Figures 6-7 of the manuscript from the seed-0 arrays of Experiment C (FD001 -> FD004, stream A).

Fig_6.pdf      streaming behaviour: rolling coverage and width of the four constructors, oracle feedback
               plus end-of-life feedback for ACI / PCCP+ACI
Fig_7.pdf      summary: (a) width / coverage / feasibility of the four constructors with the feedback
               protocols, (b) discarded infeasible mass vs local shift severity (Corollary on mass growth)

usage: python make_fig_shift.py EXPC_DIR NUMBERS_C.json OUTDIR
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXPC, NUM, OUTDIR = sys.argv[1:4]
os.makedirs(OUTDIR, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11.5,
                     "axes.titleweight": "bold", "pdf.fonttype": 42})
BLUE, GREEN, ORANGE, RED, GREY = "#2E86AB", "#28A745", "#E07B39", "#C0392B", "#7F7F7F"
D = np.load(f"{EXPC}/expC_seed0_FD004_arrays.npz")
y, f, unit = D["y"], D["f"], D["unit"]
cap = float(D["cap"])
N = json.load(open(NUM))["FD004"]


def roll(x, w):
    c = np.cumsum(np.insert(x.astype(float), 0, 0))
    return (c[w:] - c[:-w]) / w


W = 400
Lcp, Ucp = D["Lcp"], D["Ucp"]
Lp_cp, Up_cp = np.maximum(Lcp, 0), np.minimum(Ucp, cap)
Laci, Uaci = D["L_aci"], D["U_aci"]
Lp_aci, Up_aci = np.maximum(Laci, 0), np.minimum(Uaci, cap)
Leol, Ueol = D["L_eol"], D["U_eol"]
Lp_eol, Up_eol = np.maximum(Leol, 0), np.minimum(Ueol, cap)
cov = lambda L, U: (y >= L) & (y <= U)
t = np.arange(W - 1, len(y)) + 1

fig, axs = plt.subplots(2, 1, figsize=(10.2, 6.6), sharex=True)
ax = axs[0]
ax.plot(t, roll(cov(Lcp, Ucp), W), color=BLUE, lw=1.6, label="CP / PCCP (fixed)")
ax.plot(t, roll(cov(Laci, Uaci), W), color=RED, lw=1.6, label="ACI / PCCP$+$ACI (oracle feedback)")
ax.plot(t, roll(cov(Leol, Ueol), W), color=ORANGE, lw=1.3, ls="--", label="ACI / PCCP$+$ACI (end-of-life feedback)")
ax.axhline(0.9, color=GREY, ls=":", lw=1.4)
ax.set_ylabel(f"Rolling coverage (window {W})")
ax.set_ylim(0, 1.02)
ax.set_title("(a) Coverage")
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5, frameon=False)
ax = axs[1]
w = lambda L, U: np.maximum(U - L, 0)
ax.plot(t, roll(w(Lcp, Ucp), W), color=BLUE, lw=1.4, ls="--", label="CP")
ax.plot(t, roll(w(Lp_cp, Up_cp), W), color=BLUE, lw=1.8, label="PCCP")
ax.plot(t, roll(w(Laci, Uaci), W), color=RED, lw=1.4, ls="--", label="ACI (oracle)")
ax.plot(t, roll(w(Lp_aci, Up_aci), W), color=RED, lw=1.8, label="PCCP$+$ACI (oracle)")
ax.plot(t, roll(w(Lp_eol, Up_eol), W), color=ORANGE, lw=1.4, label="PCCP$+$ACI (end-of-life)")
ax.axhline(cap, color=GREY, ls=":", lw=1.4)
ax.text(t[0], cap + 3, "$|\\mathcal{K}|=R_{\\max}$", color="#444444", fontsize=9, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
ax.set_xlabel("Stream position (cycles, engines in file order)")
ax.set_ylabel(f"Rolling mean width (window {W})")
ax.set_title("(b) Interval width")
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5, frameon=False)
for a in axs:
    a.grid(alpha=0.25)
fig.tight_layout(rect=[0, 0, 0.83, 1])
fig.savefig(f"{OUTDIR}/Fig_6.pdf")
plt.close(fig)

# ------------------------------------------------------------------ Fig 6
tab = N["table"]
fig, axs = plt.subplots(1, 2, figsize=(10.2, 4.4))
ax = axs[0]
names = ["CP", "PCCP", "ACI", "PCCP+ACI"]
keys = ["CP", "PCCP-R", "ACI[oracle]", "PCCP-R+ACI[oracle]"]
xs = np.arange(4)
mp = [tab[k]["mpiw"][0] for k in keys]
sd = [tab[k]["mpiw"][1] for k in keys]
bars = ax.bar(xs, mp, 0.6, yerr=sd, color=[BLUE, "#1f6f8b", RED, "#8e2a1f"], alpha=0.9, capsize=3)
for i, k in enumerate(keys):
    ax.text(i, mp[i] + sd[i] + 3, f"PICP {tab[k]['picp'][0]:.2f}\n$\\psi_{{ok}}$ {tab[k]['feas'][0]:.2f}", ha="center", fontsize=8.5)
ax.axhline(cap, color=GREY, ls=":", lw=1.3)
ax.set_xticks(xs)
ax.set_xticklabels(names)
ax.set_ylabel("Mean interval width (cycles)")
ax.set_title("(a) Width, coverage and feasibility")
ax.set_ylim(0, max(mp) * 1.35)
ax = axs[1]
w400 = 400
miss_cp = roll(~cov(Lcp, Ucp), w400)
inf_mass = roll(np.maximum(0, -Laci) + np.maximum(0, Uaci - cap), w400)
ax.scatter(miss_cp[::40], inf_mass[::40], s=6, color=RED, alpha=0.35, rasterized=True)
b = np.polyfit(miss_cp, inf_mass, 1)
xx = np.linspace(miss_cp.min(), miss_cp.max(), 50)
ax.plot(xx, np.polyval(b, xx), color="k", lw=1.6)
r = np.corrcoef(miss_cp, inf_mass)[0, 1]
ax.set_xlabel("Local shift severity: miss rate of fixed CP (window 400)")
ax.set_ylabel("Infeasible mass removed per interval (cycles)")
ax.set_title("(b) Discarded mass grows with shift")
ax.text(0.05, 0.92, f"Pearson $r={r:.2f}$ (seed 0)", transform=ax.transAxes, fontsize=9.5)
for a in axs:
    a.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/Fig_7.pdf")
plt.close(fig)
print("written", OUTDIR)
