#!/usr/bin/env python3
"""Reproduce the tables, figures and checks of

    Post-hoc Constrained Conformal Prediction for Prognostics: Feasible Intervals that Compose with
    Adaptive Calibration under Distribution Shift

One entry point, Python standard library only.  Start here (no data and no PyTorch needed):

    python reproduce.py check       verify the stored raw results        -> prints "ALL CHECKS PASSED"
    python reproduce.py tables      regenerate the tables                -> output/tables/
    python reproduce.py figures     regenerate Figures 2-8               -> output/figures/
    python reproduce.py all         all three

    python reproduce.py list                   which command produces which table / figure
    python reproduce.py item table9            regenerate one paper item (table2 ... table17, fig2 ... fig8)
    python reproduce.py data                   find (and unzip) the NASA data files in data/
    python reproduce.py data --download all    ... and download them from NASA first (about 220 MB)
    python reproduce.py verify-data            extra checks that recompute from the C-MAPSS files
    python reproduce.py retrain expA --quick   re-run an experiment from scratch (PyTorch + data; --quick = 1 seed)

The scripts that do the work are in src/ and can also be run by hand (see README.md).
"""
import argparse
import importlib.util
import io
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

if sys.version_info < (3, 9):
    sys.exit("Python 3.9 or newer is required.")
for _stream in (sys.stdout, sys.stderr):                     # never fail on a console that cannot show a character
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

# ------------------------------------------------------------------------------------------------ data
NASA_PAGE = "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/"
DATASETS = {
    "cmapss": {
        "title": "NASA C-MAPSS turbofan engine degradation",
        "env": "CMAPSS_DIR",
        "files": [f"{kind}_FD00{i}.txt" for kind in ("train", "test", "RUL") for i in (1, 2, 3, 4)],
        "zip": "6. Turbofan Engine Degradation Simulation Data Set.zip",
        "url": "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip",
        "size": 12429152,
        "entry": '"Turbofan Engine Degradation Simulation"',
    },
    "battery": {
        "title": "NASA Li-ion battery aging (cells B0005, B0006, B0007, B0018)",
        "env": "BATTERY_DIR",
        "files": ["B0005.mat", "B0006.mat", "B0007.mat", "B0018.mat"],
        "zip": "5. Battery Data Set.zip",
        "url": "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip",
        "size": 209708670,
        "entry": '"Battery Data Set"',
    },
}
DATA_ENV = {}                                                # environment variables handed to the scripts (folders found below)


def locate(key):
    """(folder, missing files).  Looks in the folder named by the environment variable if it is set, else anywhere below data/."""
    ds = DATASETS[key]
    given = os.environ.get(ds["env"])
    roots = [Path(given)] if given else [ROOT / "data"]
    best, best_n = None, 0
    for root in roots:
        if root.is_dir():
            for folder, _, names in os.walk(root):
                n = sum(f in names for f in ds["files"])
                if n > best_n:
                    best, best_n = Path(folder), n
    if best is None:
        return None, list(ds["files"])
    have = set(os.listdir(best))
    return best, [f for f in ds["files"] if f not in have]


def harvest(archive, wanted, found, depth=0):
    """Copy the wanted files (file name -> destination folder) out of a zip, also out of zips inside it (NASA's battery archive is a zip of zips)."""
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            name = Path(info.filename).name
            if info.is_dir():
                continue
            if name in wanted:
                target = wanted[name] / name
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(info) as source, open(target, "wb") as out:
                        shutil.copyfileobj(source, out)
                    found.append(target)
            elif name.lower().endswith(".zip") and depth < 3:
                with z.open(info) as inner:
                    harvest(io.BytesIO(inner.read()), wanted, found, depth + 1)


def unpack_archives():
    """Copy the files this repository needs out of every .zip below data/ into data/CMAPSSData/ and data/NASABattery/. Returns how many."""
    wanted = {name: ROOT / "data" / folder for key, folder in (("cmapss", "CMAPSSData"), ("battery", "NASABattery")) for name in DATASETS[key]["files"]}
    found = []
    for z in sorted((ROOT / "data").rglob("*.zip")):
        print(f"  reading {z.relative_to(ROOT)} ...")
        try:
            harvest(z, wanted, found)
        except zipfile.BadZipFile:
            print(f"  {z.name} is not a valid zip file (interrupted download?); delete it and download it again")
    for path in found:
        print(f"  extracted {path.relative_to(ROOT)}")
    return len(found)


def download(key):
    ds = DATASETS[key]
    target = ROOT / "data" / ds["zip"]
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {ds['zip']} ({ds['size'] / 1e6:.1f} MB) from {ds['url']}")
    part = target.with_name(target.name + ".part")
    try:
        with urllib.request.urlopen(ds["url"], timeout=60) as response, open(part, "wb") as out:
            total = int(response.headers.get("Content-Length") or ds["size"])
            got, next_report = 0, 0.1
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                out.write(block)
                got += len(block)
                if got / total >= next_report:
                    print(f"    {100 * got / total:3.0f} %")
                    next_report += 0.1
        part.replace(target)
    except OSError as err:
        part.unlink(missing_ok=True)
        print(f"  download failed ({err}).\n  Download the file by hand from {NASA_PAGE} (entry {ds['entry']}), put it into data/ and run:  python reproduce.py data")
        return False
    if target.stat().st_size != ds["size"]:
        print(f"  note: the file has {target.stat().st_size} bytes, the authors' copy had {ds['size']}; NASA may have updated it.")
    return True


def data_command(args, only=None):
    wanted = only or (list(DATASETS) if args.download in (None, "all") else [args.download])
    if args.download:
        for key in wanted:
            if locate(key)[1] and not (ROOT / "data" / DATASETS[key]["zip"]).exists():
                download(key)
    if any(locate(key)[1] for key in wanted):
        unpack_archives()
    ok = True
    for key in wanted:
        folder, missing = locate(key)
        ds = DATASETS[key]
        if not missing:
            print(f"  [ok]      {ds['title']}: {len(ds['files'])} files in {folder}")
            DATA_ENV[ds["env"]] = str(folder)
        else:
            ok = False
            given = os.environ.get(ds["env"])
            print(f"  [missing] {ds['title']}: not found" + (f" (only part of it in {folder}: missing {', '.join(missing)})" if folder else "")
                  + (f"  [{ds['env']}={given} is set and was searched instead of data/]" if given else ""))
    if not ok:
        print("\n  To get the data, either run   python reproduce.py data --download all\n"
              f"  or download the zip files by hand from {NASA_PAGE}\n"
              "  (entries " + " and ".join(d["entry"] for d in DATASETS.values()) + "), put them into data/ and run   python reproduce.py data")
    return 0 if ok else 1


def need_data(key):
    folder, missing = locate(key)
    if missing:
        print(f"The {DATASETS[key]['title']} data are needed for this command but were not found.")
        data_command(argparse.Namespace(download=None), only=[key])
        sys.exit(1)
    DATA_ENV[DATASETS[key]["env"]] = str(folder)


def need_modules(*names):
    missing = [n for n in names if importlib.util.find_spec(n) is None]
    if missing:
        heavy = {"torch", "sklearn", "scipy"}.intersection(missing)
        sys.exit("Missing Python package(s): " + ", ".join(missing) + ".\nInstall them with:  pip install -r "
                 + ("requirements-train.txt" if heavy else "requirements.txt"))


# ------------------------------------------------------------------------------------------------ stages
class Stage:
    def __init__(self, key, what, script, args, outputs=(), needs=(), tee=None, echo=False):
        self.key, self.what, self.script, self.args = key, what, script, [str(a) for a in args]
        self.outputs, self.needs, self.tee, self.echo = list(outputs), list(needs), tee, echo


STAGES = {}                                                  # filled by main() once the folders are known
VERBOSE = False                                              # -v: show the output of every script (it is always kept in output/logs/)


def make_stages(res, out):
    tab, fig = out / "tables", out / "figures"
    return {s.key: s for s in (
        Stage("check", "verify the stored raw results", "verify_stored_results.py", [res], echo=True),
        Stage("A", "Tables 2, 3, 5, 6", "make_tables_A.py", [res / "expA", tab],
              [tab / n for n in ("tab_prov.tex", "tab_cap.tex", "tab_unit.tex", "tab_hier.tex", "numbers_A.json")]),
        Stage("summaryA", "numbers quoted in Sections 5.1-5.4", "agg_expA.py", [res / "expA"], [tab / "expA_summary.txt"], tee=tab / "expA_summary.txt"),
        Stage("D", "Table 9", "agg_expD.py", [res / "expA", res / "expD", tab], [tab / "tab_backbones.tex", tab / "numbers_D.json"]),
        Stage("E", "Table 14", "agg_expE.py", [res / "expE", tab], [tab / "tab_cqraci.tex", tab / "numbers_E.json"]),
        Stage("aggC", "aggregate the deployment experiments", "agg_expC.py", [res / "expC", tab], [tab / "numbers_C.json"]),
        Stage("tabC", "Tables 12, 13, 15, 16, 17", "make_tables_C.py", [tab],
              [tab / n for n in ("tab_shift.tex", "tab_delay.tex", "tab_decision.tex", "tab_seq.tex", "tab_seq_sens.tex")], needs=["aggC"]),
        Stage("B", "Table 11 (battery)", "agg_battery.py", [res / "battery.json"], [tab / "table11_battery.txt"], tee=tab / "table11_battery.txt"),
        Stage("thm1", "Table 7, rows (i) and (ii)", "check_thm1_ii.py", [res / "expA", tab], [tab / "numbers_thm1.json"]),
        Stage("crit", "critical-region sentence of Sections 5.3 and 5.7", "crit_cqr.py", [res / "expA", res / "expD", tab], [tab / "numbers_crit.json"]),
        Stage("fig2", "Figure 2", "make_fig2.py", [res / "expA", fig], [fig / "Fig_2.pdf"]),
        Stage("fig34", "Figures 3, 4, 5", "make_fig34.py", [res / "expA", fig], [fig / f"Fig_{i}.pdf" for i in (3, 4, 5)]),
        Stage("figshift", "Figures 6, 7", "make_fig_shift.py", [res / "expC", tab / "numbers_C.json", fig], [fig / "Fig_6.pdf", fig / "Fig_7.pdf"], needs=["aggC"]),
        Stage("figdec", "Figure 8", "make_fig_decision.py", [tab / "numbers_C.json", fig], [fig / "Fig_8.pdf"], needs=["aggC"]),
        Stage("verifydata", "recompute the C-MAPSS analyses from the data files", "verify_with_data.py", [res], echo=True),
    )}


TABLE_STAGES = ["A", "summaryA", "D", "E", "aggC", "tabC", "B", "thm1", "crit"]
FIGURE_STAGES = ["fig2", "fig34", "figshift", "figdec"]

# paper item -> (stage, output file relative to the output folder)  or  (None, where it comes from)
ITEMS = {
    "table1": (None, "legacy_notebooks/PCCP_FCP_CMAPSS_Colab.ipynb (50 seeds; needs the C-MAPSS files)"),
    "table2": ("A", "tables/tab_prov.tex"), "table3": ("A", "tables/tab_cap.tex"),
    "table4": (None, "legacy_notebooks/PCCP_FCP_CMAPSS_Colab.ipynb"),
    "table5": ("A", "tables/tab_unit.tex"), "table6": ("A", "tables/tab_hier.tex"),
    "table7": ("thm1", "tables/numbers_thm1.json (rows i, ii; the other rows: legacy_notebooks/PCCP_FCP_CMAPSS_Colab.ipynb)"),
    "table8": (None, "legacy_notebooks/PCCP_FCP_CMAPSS_Colab.ipynb"),
    "table9": ("D", "tables/tab_backbones.tex"),
    "table10": (None, "legacy_notebooks/PCCP_Improved_v9.py (raw output: results/backbones/)"),
    "table11": ("B", "tables/table11_battery.txt"),
    "table12": ("tabC", "tables/tab_shift.tex"), "table13": ("tabC", "tables/tab_delay.tex"),
    "table14": ("E", "tables/tab_cqraci.tex"),
    "table15": ("tabC", "tables/tab_decision.tex"), "table16": ("tabC", "tables/tab_seq.tex"), "table17": ("tabC", "tables/tab_seq_sens.tex"),
    "fig1": (None, "hand-drawn schematic, no code behind it"),
    "fig2": ("fig2", "figures/Fig_2.pdf"),
    "fig3": ("fig34", "figures/Fig_3.pdf"), "fig4": ("fig34", "figures/Fig_4.pdf"), "fig5": ("fig34", "figures/Fig_5.pdf"),
    "fig6": ("figshift", "figures/Fig_6.pdf"), "fig7": ("figshift", "figures/Fig_7.pdf"),
    "fig8": ("figdec", "figures/Fig_8.pdf"),
}


def shown(path):
    """A path relative to the current folder when it lies below it (shorter to read), else as it is."""
    try:
        return Path(path).relative_to(Path.cwd())
    except ValueError:
        return Path(path)


def child_env():
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    env.update(DATA_ENV)
    return env


def run_stage(stage, out, done, results):
    """Run a stage (after the stages it needs), once per invocation."""
    if stage.key in done:
        return
    for dep in stage.needs:
        run_stage(STAGES[dep], out, done, results)
    done.add(stage.key)
    for p in stage.outputs:
        p.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n== {stage.what}   [python src/{stage.script} ...]")
    t0 = time.time()
    log = out / "logs" / f"{stage.key}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    err_path = out / "logs" / f"{stage.key}.err"
    with open(err_path, "w", encoding="utf-8") as err_file:
        proc = subprocess.Popen([sys.executable, str(SRC / stage.script), *stage.args], cwd=SRC, env=child_env(), stdout=subprocess.PIPE,
                                stderr=err_file, text=True, encoding="utf-8", errors="replace")
        lines = []
        for line in proc.stdout:
            lines.append(line)
            if stage.echo or VERBOSE:
                print("   " + line.rstrip("\n"))
        code = proc.wait()
    text = "".join(lines)
    log.write_text(text, encoding="utf-8")
    if stage.tee is not None and code == 0:
        stage.tee.write_text(text, encoding="utf-8")
    missing = [p for p in stage.outputs if not p.exists()]
    ok = code == 0 and not missing
    if not ok:
        errors = err_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        print(f"   FAILED (exit code {code}){'; missing ' + ', '.join(p.name for p in missing) if missing else ''}")
        for line in errors[-12:]:
            print("   | " + line)
    else:
        made = ", ".join(str(p.relative_to(out)) for p in stage.outputs)
        print(f"   ok ({time.time() - t0:.0f} s)" + (f" -> {made}" if made else ""))
    results.append((stage.key, ok))


def run_stages(keys, out):
    done, results = set(), []
    for key in keys:
        run_stage(STAGES[key], out, done, results)
    failed = [k for k, ok in results if not ok]
    if failed:
        print("\nFAILED step(s): " + ", ".join(failed) + f"   (logs in {shown(out / 'logs')})")
    else:
        print(f"\nDone: {len(results)} step(s) succeeded. Generated files are in {shown(out)}")
    return 1 if failed else 0


# ------------------------------------------------------------------------------------------------ retraining
def retrain_plan(exp, out, workers, quick):
    r = out / "retrain"
    if exp == "expA":
        return "expA_cmapss.py", [r / "expA", workers, *(["FD001:1", "FD001:0"] if quick else ["FD001:30,FD002:12,FD003:30,FD004:12", "FD001:4,FD002:2,FD003:4,FD004:2"])], "cmapss"
    if exp == "expB":
        return "expB_battery.py", [r / "battery.json", workers, 1 if quick else 50], "battery"
    if exp == "expC":
        return "expC_shift.py", [r / "expC", workers, 1 if quick else 5], "cmapss"
    if exp == "expD":
        return "expD_backbones.py", [r / "expD", workers, "FD001:1" if quick else "FD001:12,FD002:6,FD003:12,FD004:6"], "cmapss"
    return "expE_cqr_aci.py", [r / "expE", workers, 1 if quick else 5], "cmapss"


def retrain_command(args, out):
    script, params, dataset = retrain_plan(args.experiment, out, args.workers, args.quick)
    cmd = [sys.executable, str(SRC / script), *map(str, params)]
    print("command:  python src/" + script + " " + " ".join(str(p) for p in params))
    if args.dry_run:
        return 0
    need_modules("torch", "numpy", "pandas", "scipy", "sklearn")
    need_data(dataset)
    print("This retrains the networks from scratch. " + ("--quick runs a single seed and takes about a minute on a laptop CPU."
          if args.quick else "With the settings of the paper it takes minutes to hours depending on the experiment (expA is the longest)."))
    (out / "retrain").mkdir(parents=True, exist_ok=True)
    code = subprocess.run(cmd, cwd=SRC, env=child_env()).returncode
    if code == 0:
        print(f"\nDone. New raw results are in {shown(out / 'retrain')}. The stored ones are in results/ (same file names), for comparison.")
    return code


# ------------------------------------------------------------------------------------------------ list
def list_command():
    print("Paper item  Command                                 Output (folder output/ by default)")
    for name, (stage, where) in ITEMS.items():
        label = ("Table " if name.startswith("table") else "Figure ") + re.sub(r"\D", "", name)
        cmd = f"python reproduce.py item {name}" if stage else "(not regenerated by reproduce.py)"
        print(f"{label:<11} {cmd:<39} {where}")
    print("\nAlso: python reproduce.py tables | figures | all | check | verify-data | data | retrain <expA|expB|expC|expD|expE> [--quick]")
    return 0


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--results", help="folder with the raw results (default: results/)")
    common.add_argument("--out", help="folder for everything that is generated (default: output/)")
    common.add_argument("-v", "--verbose", action="store_true", help="show the output of every script, not only of the checks")
    ap = argparse.ArgumentParser(prog="reproduce.py", description="Reproduce the tables, figures and checks of the PCCP paper "
                                 "(Post-hoc Constrained Conformal Prediction for Prognostics).",
                                 epilog="Start with:  python reproduce.py check     (then: python reproduce.py all)")
    sub = ap.add_subparsers(dest="cmd", metavar="command", required=True)
    sub.add_parser("check", parents=[common], help="verify the stored raw results (numpy only, seconds)")
    sub.add_parser("tables", parents=[common], help="regenerate the tables from the stored results")
    sub.add_parser("figures", parents=[common], help="regenerate Figures 2-8 from the stored results")
    sub.add_parser("all", parents=[common], help="check + tables + figures")
    p_item = sub.add_parser("item", parents=[common], help="regenerate one paper item, e.g. 'item table9' or 'item fig4'")
    p_item.add_argument("name", nargs="+", help="table2 ... table17, fig2 ... fig8")
    sub.add_parser("list", help="show which command produces which table and figure")
    p_data = sub.add_parser("data", help="find, unzip or download the NASA data (data/)")
    p_data.add_argument("--download", choices=["cmapss", "battery", "all"], help="download the archive(s) from NASA first (12 MB / 210 MB)")
    sub.add_parser("verify-data", parents=[common], help="extra checks that recompute from the C-MAPSS files (needs the data)")
    p_re = sub.add_parser("retrain", parents=[common], help="re-run an experiment from scratch (needs PyTorch and the data)")
    p_re.add_argument("experiment", choices=["expA", "expB", "expC", "expD", "expE"])
    p_re.add_argument("--quick", action="store_true", help="smoke test with a single seed")
    p_re.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="worker processes (default: half of the logical cores)")
    p_re.add_argument("--dry-run", action="store_true", help="only print the command")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        return list_command()
    if args.cmd == "data":
        print("Data files (never redistributed in this repository):")
        return data_command(args)
    res = Path(args.results).resolve() if args.results else ROOT / "results"
    out = Path(args.out).resolve() if args.out else ROOT / "output"
    global VERBOSE
    VERBOSE = args.verbose
    STAGES.update(make_stages(res, out))
    if args.cmd == "retrain":
        return retrain_command(args, out)
    need_modules("numpy")
    if args.cmd == "check":
        return run_stages(["check"], out)
    need_modules("pandas")
    if args.cmd == "verify-data":
        need_data("cmapss")
        return run_stages(["verifydata"], out)
    if args.cmd == "item":
        m = re.fullmatch(r"(tables?|tab|figures?|fig)(\d+)", re.sub(r"[\s._-]", "", "".join(args.name).lower()))
        name = ("table" if m and m.group(1).startswith("tab") else "fig") + (m.group(2) if m else "")
        if name not in ITEMS:
            sys.exit(f"unknown item {' '.join(args.name)!r}; try:  python reproduce.py list")
        stage, where = ITEMS[name]
        if stage is None:
            print(f"{name} is not regenerated by this script: {where}")
            return 0
        if STAGES[stage].script.startswith("make_fig"):
            need_modules("matplotlib")
        code = run_stages([stage], out)
        print(f"{name}: {shown(out / where.split(' ')[0])}")
        return code
    if args.cmd in ("figures", "all"):
        need_modules("matplotlib")
    keys = {"tables": TABLE_STAGES, "figures": FIGURE_STAGES, "all": ["check"] + TABLE_STAGES + FIGURE_STAGES}[args.cmd]
    return run_stages(keys, out)


if __name__ == "__main__":
    sys.exit(main())
