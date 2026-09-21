r"""Fig. 8 of the manuscript: unit-level sequential maintenance simulation (Experiment C, stream B).

Grouped bars: cost rate per 1000 operating cycles at the best risk parameter beta, six constructors, oracle and
end-of-life feedback, for the severe (FD004) and mild (FD003) shift; numbers above the bars are the corrective-failure
fractions.  Bars are means over seeds with +/- 1 SD.

usage: python make_fig_decision.py NUMBERS_C.json OUTDIR [DELTA] [CF]
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NUM, OUTDIR = sys.argv[1], sys.argv[2]
DELTA = sys.argv[3] if len(sys.argv) > 3 else "10"
CF = sys.argv[4] if len(sys.argv) > 4 else "20"
os.makedirs(OUTDIR, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11.5,
                     "axes.titleweight": "bold", "pdf.fonttype": 42})
N = json.load(open(NUM))
ARMS = ["CP", "PCCP-inf", "PCCP-R", "ACI", "PCCP-inf+ACI", "PCCP-R+ACI"]
LAB = ["CP", "PCCP\n$[0,\\infty)$", "PCCP\n$[0,R_{\\max}]$", "ACI", "PCCP$[0,\\infty)$\n$+$ACI", "PCCP$[0,R_{\\max}]$\n$+$ACI"]
COL = ["#2E86AB", "#5FA8C8", "#1f6f8b", "#E07B39", "#D9534F", "#8e2a1f"]
fig, axs = plt.subplots(1, 2, figsize=(10.6, 4.7), sharey=False)
for ax, fd, title in zip(axs, ("FD004", "FD003"), ("(a) FD001$\\to$FD004: severe shift", "(b) FD001$\\to$FD003: mild shift")):
    seq = N[fd]["sequential"]["125"]
    top = max(seq[p][a][DELTA][CF]["rate"][0] + seq[p][a][DELTA][CF]["rate"][1] for p in ("oracle", "eol") for a in ARMS) * 1e3
    ax.set_ylim(0, top * 1.14)
    x0 = 0
    ticks, tlabels = [], []
    for prot, plab in (("oracle", "oracle feedback"), ("eol", "end-of-life feedback")):
        for i, arm in enumerate(ARMS):
            v = seq[prot][arm][DELTA][CF]
            mean, sd = 1e3 * v["rate"][0], 1e3 * v["rate"][1]
            ax.bar(x0 + i, mean, 0.8, yerr=sd, color=COL[i], capsize=2, alpha=0.92)
            ax.text(x0 + i, mean + sd + 0.02 * ax.get_ylim()[1], f"{100 * v['fail_rate'][0]:.0f}%", ha="center", fontsize=7.5, color="#333333")
        ticks.append(x0 + 2.5)
        tlabels.append(plab)
        x0 += 7.5
    ax.set_xticks([t for t in ticks])
    ax.set_xticklabels(tlabels)
    ax.set_title(title)
    ax.set_ylabel("Cost rate per 1000 operating cycles")
    ax.grid(axis="y", alpha=0.25)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COL]
fig.legend(handles, [l.replace("\n", " ") for l in LAB], loc="lower center", ncol=6, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.text(0.5, 0.93, "", ha="center")
fig.tight_layout(rect=[0, 0.07, 1, 1])
fig.savefig(f"{OUTDIR}/Fig_8.pdf")
print("written", OUTDIR)
