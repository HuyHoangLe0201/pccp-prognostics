# Legacy notebooks and script

These files are the code of the analyses that were carried over from the first version of the paper and of the first version of
Figs. 2-4. They are kept unchanged for provenance and are **not** part of the quick start (`reproduce.py` does not run them).
They need the NASA data (see `../data/README.md`), a Python environment with `requirements-train.txt`, and are written for Google Colab
(the C-MAPSS notebook reads the files from `/content`, where Colab puts uploaded files).

| File | Paper | Notes |
|---|---|---|
| `PCCP_FCP_CMAPSS_Colab.ipynb` | Tables 1, 4, 7 (except rows (i) and (ii), see `../src/check_thm1_ii.py`), 8 | C-MAPSS main protocol with 50 seeds: CP / PCCP / FCP, subgroups, FCP bound, wasted budget. The outputs of the original run are stored in the notebook |
| `PCCP_Improved_v9.py` | Table 10 | Windowed architecture with MC dropout, ensemble, QR, CQR and constrained QR (one split). Table 10 was produced with `run_full_experiment(subset, alpha=0.1, adaptive_sensors=...)`, `False` for FD001 and FD003 and `True` for FD002 and FD004; see `../docs/PROTOCOL_NOTES.md`. One subset takes 20 minutes to several hours on a CPU. Raw output: `../results/backbones/`. The only change with respect to the original script is that the data folder is read from `CMAPSS_DIR` (default `data/CMAPSSData`) |
| `Battery_RUL_PCCP_Colab_v4.ipynb` | superseded by `../src/expB_battery.py` | Original battery notebook. Its cap R_max = 124 is the largest RUL over all cells including the held-out one; the paper reports the leak-free cap (`expB_battery.py`, mode `loo`), and `expB_battery.py` also runs this original mode (`legacy`) to quantify the effect |
| `PCCP_Full_Figures_v8_originally_submitted_Figs2-4.ipynb` | superseded | Notebook of the first version of Figs. 2-4 (outputs stripped). The figures of the paper are made by `../src/make_fig2.py` and `../src/make_fig34.py` |

The comments inside `PCCP_Improved_v9.py` refer to an earlier review round of a previous version of the manuscript.
