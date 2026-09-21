r"""LaTeX table bodies of Tables 12, 13, 15, 16 and 17 (Sections 5.9-5.10) from numbers_C.json (Experiment C aggregate)."""
import json
import sys

GEN = sys.argv[1] if len(sys.argv) > 1 else "gen"
N = json.load(open(f"{GEN}/numbers_C.json"))
ARMS = ["CP", "PCCP-inf", "PCCP-R", "ACI", "PCCP-inf+ACI", "PCCP-R+ACI"]
NAME = {"CP": "CP (fixed)", "PCCP-inf": r"PCCP $[0,\infty)$ (fixed)", "PCCP-R": r"PCCP $[0,\Rmax]$ (fixed)", "ACI": "ACI",
        "PCCP-inf+ACI": r"PCCP $[0,\infty)$ $+$ ACI", "PCCP-R+ACI": r"PCCP $[0,\Rmax]$ $+$ ACI"}


def key(arm, prot="oracle"):
    if arm in ("CP", "PCCP-inf", "PCCP-R"):
        return arm
    return "ACI[%s]" % prot if arm == "ACI" else "%s[%s]" % (arm, prot)


def w(path, lines):
    open(f"{GEN}/{path}", "w", encoding="utf-8").write("\n".join(lines) + "\n")


# ------------------------------------------------------------------ Table: coverage recovery (oracle), both targets
L = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule", r"Constructor & PICP & MPIW & $\psi_{\mathrm{ok}}$ & IntScore \\", r"\midrule"]
for fd, lab in (("FD004", r"\emph{FD001$\to$FD004 (severe shift; RMSE %.1f $\to$ %.1f)}"), ("FD003", r"\emph{FD001$\to$FD003 (mild shift; RMSE %.1f $\to$ %.1f)}")):
    n = N[fd]
    L.append(r"\multicolumn{5}{@{}l}{" + lab % (n["rmse_source"][0], n["rmse_A"][0]) + r"} \\")
    best = {m: (min if m in ("mpiw", "iscore") else max) for m in ("mpiw", "iscore")}
    for arm in ARMS:
        t = n["table"][key(arm)]
        bold = arm == "PCCP-R+ACI"
        f = lambda s: (r"\textbf{%s}" % s) if bold else s
        L.append(f"{NAME[arm] if not bold else chr(92) + 'textbf{' + NAME[arm] + '}'} & {f('%.3f' % t['picp'][0])} ({t['picp'][1]:.3f}) & {f('%.1f' % t['mpiw'][0])} & "
                 f"{f('%.2f' % t['feas'][0])} & {f('%.1f' % t['iscore'][0])} \\\\")
    if fd == "FD004":
        L.append(r"\midrule")
L += [r"\bottomrule", r"\end{tabular}"]
w("tab_shift.tex", L)

# ------------------------------------------------------------------ Table: feedback protocols
L = [r"\footnotesize\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
     r" & \multicolumn{4}{c}{FD001$\to$FD004} & \multicolumn{4}{c}{FD001$\to$FD003} \\", r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
     r"Feedback & PICP & MPIW ACI & MPIW PCCP$+$ACI & Sat.\ (\%) & PICP & MPIW ACI & MPIW PCCP$+$ACI & Sat.\ (\%) \\", r"\midrule"]
for p, lab in (("oracle", "Oracle ($d=0$)"), ("delay10", "Delay $d=10$"), ("delay50", "Delay $d=50$"), ("delay125", "Delay $d=125$"), ("eol", "End of life")):
    cells = []
    for fd in ("FD004", "FD003"):
        n = N[fd]
        a, b = n["table"]["ACI[%s]" % p], n["table"]["PCCP-R+ACI[%s]" % p]
        cells += [f"{a['picp'][0]:.3f}", f"{a['mpiw'][0]:.1f}", f"{b['mpiw'][0]:.1f}", f"{100 * n['diag'][p]['frac_level_saturated'][0]:.0f}"]
    L.append(lab + " & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
w("tab_delay.tex", L)

# ------------------------------------------------------------------ Table: per-moment decision cost (FD004, stream A)
L = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule", r" & \multicolumn{3}{c}{Capped truth $r^\star=\min\{\cdot,\Rmax\}$} & \multicolumn{3}{c}{Uncapped truth $r^\star=T-t$} \\",
     r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}", r"Constructor & $20$ & $50$ & $100$ & $20$ & $50$ & $100$ \\", r"\midrule"]
pm = N["FD004"]["per_moment"]
for arm in ARMS:
    vals = [pm[cf][arm]["capped"][0] for cf in ("20", "50", "100")] + [pm[cf][arm]["uncapped"][0] for cf in ("20", "50", "100")]
    L.append(NAME[arm] + " & " + " & ".join(f"{v:.1f}" for v in vals) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
w("tab_decision.tex", L)

# ------------------------------------------------------------------ Table: sequential simulation
L = [r"\footnotesize\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
     r" & \multicolumn{4}{c}{FD001$\to$FD004} & \multicolumn{4}{c}{FD001$\to$FD003} \\", r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
     r" & \multicolumn{2}{c}{Oracle} & \multicolumn{2}{c}{End of life} & \multicolumn{2}{c}{Oracle} & \multicolumn{2}{c}{End of life} \\",
     r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
     r"Constructor & Cost & Fail & Cost & Fail & Cost & Fail & Cost & Fail \\", r"\midrule"]
for arm in ARMS:
    cells = []
    for fd in ("FD004", "FD003"):
        for prot in ("oracle", "eol"):
            v = N[fd]["sequential"]["125"][prot][arm]["10"]["20"]
            cells += [f"{1e3 * v['rate'][0]:.1f}", f"{100 * v['fail_rate'][0]:.0f}\\%"]
    L.append(NAME[arm] + " & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
w("tab_seq.tex", L)

# ------------------------------------------------------------------ Table: sequential sensitivity (FD004, oracle & EOL)
L = [r"\footnotesize\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule", r"Setting & Feedback & CP & PCCP $[0,\Rmax]$ & ACI & PCCP $[0,\infty)$ $+$ ACI & PCCP $[0,\Rmax]$ $+$ ACI \\", r"\midrule"]
seq = N["FD004"]["sequential"]
rows = []
for cap, delta, cf in (("125", "10", "5"), ("125", "10", "20"), ("125", "10", "100"), ("125", "1", "20"), ("125", "5", "20"), ("125", "25", "20"),
                       ("100", "10", "20"), ("150", "10", "20")):
    for prot in ("oracle", "eol"):
        v = lambda arm: seq[cap][prot][arm][delta][cf]["rate"][0] * 1e3
        lab = rf"$\Rmax={cap}$, $\Delta={delta}$, $c_f/c_p={cf}$" if prot == "oracle" else ""
        L.append(f"{lab} & {'oracle' if prot == 'oracle' else 'end of life'} & {v('CP'):.1f} & {v('PCCP-R'):.1f} & {v('ACI'):.1f} & {v('PCCP-inf+ACI'):.1f} & {v('PCCP-R+ACI'):.1f} \\\\")
    L.append(r"\addlinespace")
L += [r"\bottomrule", r"\end{tabular}"]
w("tab_seq_sens.tex", L)
print("written tables to", GEN)
