# Stored results

These are the raw per-seed outputs of the experiments behind the paper (about 16 MB in total). Every table and figure that
`python reproduce.py tables|figures` regenerates is computed from these files, and `python reproduce.py check` re-derives
the delicate parts independently. You can also load them directly:

```python
import json, numpy as np
run = json.load(open("results/expA/expA_FD001_seed0_main.json"))      # metrics of one run
arrays = np.load("results/expA/expA_FD001_seed0_arrays.npz")           # calibration / test predictions of the same run
print(run["cap125"]["Kcap"], arrays["y_te"][:5])
```

| Folder / file | Produced by | Paper | Content |
|---|---|---|---|
| `expA/` | `src/expA_cmapss.py` | Tables 2, 3, 5, 6, Figs. 2-5 | C-MAPSS main protocol, one run per (subset, seed). `expA_FD00x_seed<k>_main.json`: metrics of the label convention R_max = 125 for every constraint variant (`none`, `K0inf`, `Kcap`, `Kage`, ...), per-engine sums, coverage by RUL band and by life fraction, engine-level calibration variants. `expA_FD00x_seed<k>_sens.json`: the same for R_max = 100, 150 and the uncapped label (few seeds). `expA_FD00x_seed<k>_arrays.npz`: `f_ca`, `y_ca` (calibration predictions and labels), `f_te`, `y_te` (test), `ute` (engine id), `age_te`, `life_te`. Seeds: 30 (FD001, FD003) and 12 (FD002, FD004) |
| `expB` (`battery.json`) | `src/expB_battery.py` | Table 11 | NASA batteries, leave-one-cell-out, 3 cells x 50 seeds = 150 folds. Per fold: `cell`, `seed`, `q_cp`, the caps `rmax_loo` (from the training cells only, used in the paper) and `rmax_legacy`, the metrics of CP / PCCP / FCP for `loo`, `legacy` and `noceil`, and `arrays` (`y`, `f`, `L`, `U`) |
| `expC/` | `src/expC_shift.py` | Tables 12, 13, 15-17, Figs. 6-8 | Deployment under shift FD001 -> FD004 (severe) and FD001 -> FD003 (mild), 5 seeds (`expC_seed<k>.json`): six constructors, oracle / delayed / end-of-life feedback, per-moment and sequential maintenance analyses, label caps 100 / 125 / 150. `expC_seed0_FD00x_arrays.npz`: the interval streams of seed 0 (`y` label capped at R_max, `y_unc` uncapped label, `f` point prediction, `unit` engine id, `cycle`, `Lcp`/`Ucp` fixed CP, `L_aci`/`U_aci` and `A` ACI intervals and level with oracle feedback, `L_d50`/`U_d50` ACI with delay 50, `L_eol`/`U_eol` ACI with end-of-life feedback, `q`, `cap`) |
| `expD/` | `src/expD_backbones.py` | Table 9 | PCCP on locally weighted CP, CQR, QR and MC dropout, one run per (subset, seed): `expD_FD00x_seed<k>.json` and `_arrays.npz` (`y`, `ute`, `age`, and the endpoints `L_*`, `U_*` of each construction). Seeds: 12 (FD001, FD003) and 6 (FD002, FD004). The plain split-CP row is recomputed from `expA/` |
| `expE/` | `src/expE_cqr_aci.py` | Table 14 | CQR as the base construction of ACI under the same shifts, 5 seeds (`expE_seed<k>.json`) |
| `backbones/` | `legacy_notebooks/PCCP_Improved_v9.py` | Table 10 | Raw output of the legacy backbone script (windowed architecture, one split; the file names carry the numbering of an earlier draft) |
| `numbers/` | `python reproduce.py tables` | text and tables | Aggregated numbers quoted in the paper (`numbers_A`, `numbers_C`, `numbers_D`, `numbers_E`, `numbers_crit`, `numbers_thm1`). `python reproduce.py tables` regenerates the same numbers (the files can differ only in the line-ending convention of your operating system) and `check` compares them with the raw files |

Retraining (`python reproduce.py retrain expX`) writes new files with the same names into `output/retrain/`, never into this folder.
