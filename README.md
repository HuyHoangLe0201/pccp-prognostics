# PCCP: Post-hoc Constrained Conformal Prediction for Prognostics

Code, stored results and a one-command reproduction script for

> H. H. Le, D. Huynh, K.-A. Nguyen. **Post-hoc Constrained Conformal Prediction for Prognostics: Feasible Intervals that Compose with
> Adaptive Calibration under Distribution Shift.** *Heliyon* (under revision, manuscript HELIYON-D-26-06430).

**The idea.** A conformal prediction interval `C(x)` for remaining useful life (RUL) can contain impossible values, for example a negative
RUL or a value above a known cap. PCCP cuts the interval at a feasible set `K(x)`: `C_proj(x) = C(x) ∩ K(x)`, after the fact, without
retraining or recalibration. The paper shows that, when the true value always lies in `K`, this keeps the coverage of `C` (marginally and on
every subgroup) and shortens the interval by exactly the part that lay outside `K`; that combined with adaptive conformal inference (ACI) it
leaves the ACI level sequence unchanged, so it keeps ACI's long-run coverage under distribution shift; and how much it helps in practice
(C-MAPSS turbofans, NASA batteries, maintenance decisions).

## Reproduce the paper in a minute (no data, no GPU, no PyTorch)

Download this repository (green **Code** button, then *Download ZIP*, or `git clone`), open a terminal in its folder and run

```bash
pip install -r requirements.txt        # numpy, pandas, matplotlib
python reproduce.py check              # about 10 s: verifies the stored results, ends with ALL CHECKS PASSED
python reproduce.py all                # about 1 min: check + every table (LaTeX) and figure (PDF), written to output/
```

Then look in `output/tables/` (LaTeX table bodies and the numbers quoted in the text) and `output/figures/` (`Fig_2.pdf` ... `Fig_8.pdf`).
On the authors' machine the regenerated tables are byte-identical, and the figures pixel-identical, to those in the paper. Figure 1 is a
hand-drawn schematic, so there is no code behind it.

`python reproduce.py --help` lists all commands; `python reproduce.py list` shows which command produces which table or figure.

## What you can do

| I want to ... | Command | Needs | Time |
|---|---|---|---|
| verify that the stored raw results satisfy the identities of the paper | `python reproduce.py check` | numpy | 10 s |
| regenerate the tables (Tables 2, 3, 5, 6, 7 (i, ii), 9, 11-17) | `python reproduce.py tables` | + pandas | 20 s |
| regenerate Figures 2-8 | `python reproduce.py figures` | + matplotlib | 15 s |
| regenerate one item, e.g. Table 9 or Figure 4 | `python reproduce.py item table9`, `python reproduce.py item fig4` | as above | seconds |
| recompute the C-MAPSS analyses from the raw data files | `python reproduce.py verify-data` | + [data](data/README.md) | 10 s |
| get the NASA data (12 MB + 210 MB) | `python reproduce.py data --download all` | internet | minutes |
| retrain an experiment from scratch, smoke test with one seed | `python reproduce.py retrain expA --quick` | + PyTorch, data (`pip install -r requirements-train.txt`) | 15-50 s |
| retrain an experiment with the settings of the paper | `python reproduce.py retrain expA` (`expB` ... `expE`) | as above | minutes to hours |

Retraining one seed of every experiment reproduced the stored results **bit for bit** on the authors' machine; on other machines the
results can differ in the last digits (see [docs/PROTOCOL_NOTES.md](docs/PROTOCOL_NOTES.md)). Retraining never overwrites `results/`; new files go to `output/retrain/`.

## Which command reproduces which table or figure

| Paper | What | Command | Output in `output/` |
|---|---|---|---|
| Tables 2, 3 | where the width reduction comes from; label-cap sensitivity | `item table2`, `item table3` | `tables/tab_prov.tex`, `tables/tab_cap.tex` |
| Tables 5, 6 | unit-level and stage-wise coverage; engine-level calibration | `item table5`, `item table6` | `tables/tab_unit.tex`, `tables/tab_hier.tex` |
| Table 7, rows (i), (ii) | PCCP equals FCP when every prediction is feasible | `item table7` | `tables/numbers_thm1.json` |
| Table 9 | PCCP on locally weighted CP, CQR, QR, MC dropout | `item table9` | `tables/tab_backbones.tex` |
| Table 11 | NASA batteries, leave-one-cell-out | `item table11` | `tables/table11_battery.txt` |
| Tables 12, 13, 15, 16, 17 | FD001 to FD004 / FD003 shift, feedback timing, maintenance decisions | `item table12` ... `item table17` | `tables/tab_shift.tex`, `tab_delay.tex`, `tab_decision.tex`, `tab_seq.tex`, `tab_seq_sens.tex` |
| Table 14 | CQR as the base construction of ACI | `item table14` | `tables/tab_cqraci.tex` |
| Figure 2 | CP vs PCCP on one test engine | `item fig2` | `figures/Fig_2.pdf` |
| Figures 3, 4 | coverage and width sweeps with seed bands | `item fig3`, `item fig4` | `figures/Fig_3.pdf`, `Fig_4.pdf` |
| Figure 5 | coverage by stage of life and per engine | `item fig5` | `figures/Fig_5.pdf` |
| Figures 6, 7 | streaming behaviour and summary under shift | `item fig6`, `item fig7` | `figures/Fig_6.pdf`, `Fig_7.pdf` |
| Figure 8 | sequential maintenance simulation | `item fig8` | `figures/Fig_8.pdf` |
| Sections 5.1-5.4 | numbers quoted in the text (constraint variants, engine-level coverage, stages) | `tables` | `tables/expA_summary.txt` |
| Sections 5.3, 5.7 | critical-region sentence (CP versus CQR) | `tables` | `tables/numbers_crit.json` |
| Tables 1, 4, 7 (other rows), 8, 10 | analyses carried over from the first version | notebooks, see [legacy_notebooks](legacy_notebooks/README.md) | not regenerated by `reproduce.py` |

## Repository layout

```
reproduce.py            the one script to run (standard library only)
src/                    experiments (expA-expE), aggregation, table and figure scripts, verification; each can also be run by hand
results/                stored raw per-seed outputs of all experiments, 16 MB   (see results/README.md)
data/                   put the NASA files here; they are not redistributed     (see data/README.md)
legacy_notebooks/       code of the analyses carried over from the first version (Tables 1, 4, 7, 8, 10)
docs/PROTOCOL_NOTES.md  protocol details, reproducibility test, known issues
requirements.txt        numpy, pandas, matplotlib                (check, tables, figures)
requirements-train.txt  + scipy, scikit-learn, PyTorch           (retraining)
output/                 created by reproduce.py, ignored by git
```

The scripts in `src/` take their input and output folders as arguments, and the top of each file says how, for example
`python src/make_tables_A.py results/expA output/tables`. `reproduce.py` only calls them in the right order.

## Troubleshooting

* *Missing Python package(s)*: run the `pip install` line that the message shows.
* *`findfont: Font family 'Times New Roman' not found`* while drawing figures: harmless, the figures then use DejaVu Serif.
* *`Could not find the number of physical cores ... [WinError 2]`* (Windows 11, when retraining `expC` or `expE`): a harmless warning of scikit-learn / joblib.
* PyTorch on Linux downloads a large CUDA build by default; the CPU build is enough: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
* Data problems: `python reproduce.py data` says what is missing; see [data/README.md](data/README.md).
* Tested with Python 3.12 on Windows 11. `.github/workflows/check.yml` makes GitHub run `check`, `tables` and `figures` on Linux at every push.

## Citation and license

Please cite the paper above; [CITATION.cff](CITATION.cff) has the details (GitHub shows a *Cite this repository* button). If you use the NASA data,
cite the references given in [data/README.md](data/README.md). The code is released under the MIT license ([LICENSE](LICENSE)).
Questions and bug reports: please open a GitHub issue.
