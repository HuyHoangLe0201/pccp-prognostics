# Data

The two public NASA data sets used in the paper are **not** stored in this repository. You only need them if you want to
re-run the experiments or the extra checks that recompute from the raw files (`python reproduce.py verify-data`, `python reproduce.py retrain ...`).
Everything else (`check`, `tables`, `figures`, `all`) works from the stored results in `results/` without any data.

## Easiest way

```bash
python reproduce.py data --download all      # about 12 MB (C-MAPSS) + 210 MB (batteries); needs an internet connection
```

This downloads the two archives from NASA into this folder, copies out the 12 + 4 files that are needed and prints `[ok]` for each data set.
If the download is blocked on your network, use the manual way below.

## Manual way

1. Open the NASA Prognostics Center of Excellence data repository:
   <https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>
2. Download the two entries (direct links, taken from that page):

   | Entry on the page | File | Size |
   |---|---|---|
   | Turbofan Engine Degradation Simulation (C-MAPSS) | <https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip> | 12.4 MB |
   | Battery Data Set | <https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip> | 209.7 MB |

3. Put the downloaded `.zip` files into this folder (`data/`) and run

   ```bash
   python reproduce.py data
   ```

   The command looks inside the archives (also inside the zip files that the battery archive contains) and copies only the files
   the code reads:

   | Folder | Files |
   |---|---|
   | `data/CMAPSSData/` | `train_FD001.txt` ... `train_FD004.txt`, `test_FD001.txt` ... `test_FD004.txt`, `RUL_FD001.txt` ... `RUL_FD004.txt` |
   | `data/NASABattery/` | `B0005.mat`, `B0006.mat`, `B0007.mat`, `B0018.mat` |

You can also unzip by hand and put the same files there, or keep the data somewhere else and point to it:

```bash
export CMAPSS_DIR=/path/to/CMAPSSData        # Windows PowerShell:  $env:CMAPSS_DIR = "D:\path\to\CMAPSSData"
export BATTERY_DIR=/path/to/NASABattery
```

## Terms of use and citation

The data belong to NASA and are distributed under the terms stated on the page above; they are never redistributed here. If you use them, cite

* A. Saxena, K. Goebel, D. Simon, N. Eklund, *Damage propagation modeling for aircraft engine run-to-failure simulation*, 2008 International Conference on Prognostics and Health Management, IEEE, 2008, pp. 1-9 (C-MAPSS).
* B. Saha, K. Goebel, *Battery Data Set*, NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA, 2007 (batteries).
