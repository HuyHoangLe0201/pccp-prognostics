# Protocol notes

Details that are useful when you compare the code with the paper, or when you change it. Section, table and figure numbers refer to the paper.

## Does retraining reproduce the stored results?

On the authors' machine (Windows 11, CPU only, Python 3.12.6, numpy 2.2.4, pandas 2.3.3, scipy 1.15.2, scikit-learn 1.6.1, torch 2.9.1+cpu),
retraining one seed of each experiment with `python reproduce.py retrain expX --quick` gave the stored raw results **bit for bit**:

| Experiment | Run compared | Numbers compared | Time |
|---|---|---|---|
| A | FD001, seed 0 (`expA_FD001_seed0_main.json` and `_arrays.npz`) | 246 values and 7 arrays, all identical | 16 s |
| B | 3 cells, seed 0 (folds of `battery.json`) | 2,280 values, all identical | 15 s |
| C | seed 0 (`expC_seed0.json` and both `_arrays.npz`) | 9,143 values and 32 arrays, all identical | 51 s |
| D | FD001, seed 0 | 69 values, all identical | 46 s |
| E | seed 0 | 186 values, all identical | 24 s |

The random seeds fix the random numbers, but not the order of floating-point additions, which depends on the processor, the number of threads
and the versions of numpy, PyTorch and scikit-learn. On another machine the results can therefore differ in the last digits (and, for the
networks, slightly more). The stored files are the ones the paper reports. To use the exact library versions of the test above:

```bash
pip install numpy==2.2.4 pandas==2.3.3 matplotlib==3.10.1 scipy==1.15.2 scikit-learn==1.6.1 torch==2.9.1
```

## Predictor and calibration (Experiments A, D, E; Section 4.3)

* **Predictor.** Single-cycle MLP (128-64-32, ReLU), Adam (1e-3), batch 256, 80 epochs, engine-level 70/15/15 train/calibration/test
  split, `alpha = 0.1`. Mini-batches are drawn from a random permutation of the training rows in each epoch instead of through a
  `DataLoader`; the distribution of batches is the same, only faster on CPU. Features are min-max scaled with the statistics of the
  whole training file (all engines, no labels), as in the legacy notebook.
* **Conformal quantile.** The battery experiment and the deployment experiment use the `ceil((n+1)(1-alpha))`-th smallest score.
  The C-MAPSS notebook and `common.py` (hence Tables 1-8 and every analysis in `src/` that calls `common.conformal_quantile`) use numpy's
  `higher` interpolation at level `ceil((n+1)(1-alpha))/n`, which returns the *next* order statistic: conservative by one rank
  (0.03 cycles, +0.0003 PICP on FD001). The backbone script uses numpy's default linear interpolation. Coverage is valid in all three
  cases; none changes a result beyond the last digit.
* **Seeds.** These experiments reuse the engine-level splits of the first seeds (0-29, or 0-11) of the 50-seed protocol of Table 1 but
  retrain the network with the re-implemented loop, so cycle-level statistics agree with Table 1 only up to that difference and to
  seed variability. Standard deviations in the tables generated here are sample standard deviations (n-1) over seeds; the legacy
  notebook reports population standard deviations.

## Deployment experiments (`expC_shift.py`; Sections 4.7, 5.9, 5.10)

The FD001-trained predictor sees the same 14 sensor channels on the target subset; FD004 is normalized per operating condition
(k-means, k = 6, on the unlabelled FD004 training file), FD003 reuses the FD001 scaling. Point predictions are clipped to `[0, Rmax]`
before intervals are formed. ACI uses `gamma = 0.03` and a trailing window of 500 revealed scores; the conformal level is clipped to the
range the window can attain (largest window score when `alpha_t < 1/(m+1)`, smallest when `alpha_t >= 1`) instead of returning the whole
line / the empty set, so `alpha_t` is *not* confined to `[-gamma, 1+gamma]` and the coverage bound of the original recursion does not
apply verbatim; the exact identity `PICP = 1 - alpha + (alpha_{T+1} - alpha_1)/(gamma T)` does, and `verify_stored_results.py` checks it.
The three ACI-based constructors share one level trajectory per feedback protocol, and the script records the number of steps at which
the miss indicators of ACI and of projected ACI differ (Theorem 2(i); zero in every run). Stream A is the official test engines
(coverage, delay and per-moment analyses), Stream B the complete run-to-failure engines of the target training file (sequential
maintenance simulation). The failure share of a cost-minimizing policy is not stable across seeds when the intervals are uninformative
(end-of-life ACI on FD004: 0-1 % in two seeds, 74-91 % in three); costs are.

The original script of the FD001 -> FD004 experiment was not available, so `expC_shift.py` is a full re-implementation whose protocol is
specified in Section 4.7 of the paper; its numbers are the ones reported.

## Other constructions (`expD_backbones.py`, `expE_cqr_aci.py`; Sections 5.7, 5.9)

All networks use the single-cycle MLP of Section 4.3 (128-64-32, Adam 1e-3, batch 256, 80 epochs) and the engine-level splits of
Experiment A; `agg_expD.py` recomputes the split-CP row from `results/expA/*arrays.npz` for the same seeds. QR/CQR: two outputs, pinball
loss at 0.05/0.95, outputs sorted, CQR score `max(q_lo - y, y - q_hi)`; locally weighted CP: Gaussian-NLL network (mean, log-variance),
score `|y - mu| / sigma`; MC dropout: dropout 0.2, 50 passes, mean +- 1.645 sd, uncalibrated. The projection follows the paper's
convention (an empty intersection is a miss of zero width, no boundary guard), so `agg_expD.py` can assert that the PICP of the
projections equals that of the base intervals *exactly*. In `expE_cqr_aci.py` the quantile band is deliberately **not** clipped before the
conformal correction: with a negative correction a clipped band puts the upper endpoint of every cap-valued row below `Rmax` and misses
the label `Y = Rmax`, an artefact of the clipping and not of CQR; clipping is what the projection does, on the final interval. The ACI
recursion, window, step and level clipping are those of `expC_shift.py`, with the signed CQR scores.

## Battery experiment (`expB_battery.py`; Sections 4.5, 5.8)

`expB_battery.py` also runs the original ("legacy") battery protocol, whose cap is the largest RUL over *all* cells including the
held-out one; in that mode it reproduces the first version of the battery table exactly, which allows the effect of removing the leakage
to be quantified on identical trained models. The paper reports the leak-free ("loo") mode.

## Known issue in the legacy backbone script (Table 10)

`legacy_notebooks/PCCP_Improved_v9.py` selects "constant" sensors with a coefficient-of-variation test that flags all 21 C-MAPSS sensors
as constant on FD001 and FD003 when `adaptive_sensors=True`. Table 10 was therefore produced with

```python
import PCCP_Improved_v9 as v9
for subset, adaptive in (("FD001", False), ("FD002", True), ("FD003", False), ("FD004", True)):
    results = v9.run_full_experiment(subset, alpha=0.1, adaptive_sensors=adaptive)
```

(`False` uses the fixed literature list of 7 constant sensors on the single-condition subsets; `True` keeps all sensors on the
multi-condition subsets, where the test drops none). The copy in this repository reads the data folder from `CMAPSS_DIR` (default
`data/CMAPSSData`); nothing else in it was changed. One subset takes 20 minutes to several hours on a CPU. The raw outputs are in
`results/backbones/`.

## First version of Figs. 2-4

The first version of Figs. 2-4 was produced by the authors' figure notebook
(`legacy_notebooks/PCCP_Full_Figures_v8_originally_submitted_Figs2-4.ipynb`, outputs stripped): windowed 510-feature predictor with
batch normalization, seed 42, evaluation on the 10,196 windows of the official FD001 test engines, Fig. 2(a-d) drawn for the first
*validation* engine, Fig. 4 subgroups defined by the *true* RUL and the "physical consistency" bar of Fig. 3(d) counting the lower bound
only. These figures are superseded: Figs. 2-4 of the paper use the predictor and protocol of Section 4.3 / Table 1
(`src/make_fig2.py`, `src/make_fig34.py`), Fig. 4 uses the predicted-RUL subgroups of Table 4, and "feasible" means contained in
`[0, Rmax]` throughout.

## Fonts in the figures

The figure scripts ask for Times New Roman and fall back to DejaVu Serif when it is not installed (Linux, most CI machines). The data in
the figures are identical; only the shapes of the letters differ, and matplotlib prints a harmless "findfont" warning.
