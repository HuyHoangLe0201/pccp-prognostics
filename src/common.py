"""Shared utilities for the PCCP experiments (C-MAPSS single-cycle protocol).

Everything here mirrors legacy_notebooks/PCCP_FCP_CMAPSS_Colab.ipynb (the code behind the
C-MAPSS main protocol of Table 1): engine-level 70/15/15 splits drawn with
np.random.RandomState(seed), min-max features, MLP [128, 64, 32], Adam(1e-3),
batch 256, 80 epochs, split-CP quantile with the +1/n correction.

PyTorch is needed only to train and evaluate the networks (MLP, train_mlp, predict). Everything else in
this module (data loading, engine-level splits, conformal quantile, projection) uses numpy / pandas only,
so the verification, table and figure scripts run without PyTorch.
"""
import os
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = nn = None

DATA_DIR = os.environ.get("CMAPSS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "CMAPSSData"))
ALPHA = 0.1
N_EPOCHS, LR, BATCH = 80, 1e-3, 256
HIDDEN = (128, 64, 32)

SENSORS = [f"s{i}" for i in range(1, 22)]
SETTINGS = ["op1", "op2", "op3"]
ALL_COLS = ["unit", "cycle"] + SETTINGS + SENSORS
FD001_CONST = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
SUBSET_CFG = {
    "FD001": {"drop": FD001_CONST, "settings": False},
    "FD002": {"drop": [], "settings": True},
    "FD003": {"drop": FD001_CONST, "settings": False},
    "FD004": {"drop": [], "settings": True},
}


def load_cmapss(subset):
    """Returns X (min-max), engine id, cycle (age), lifetime T, uncapped RUL T - t."""
    df = pd.read_csv(f"{DATA_DIR}/train_{subset}.txt", sep=r"\s+", header=None, names=ALL_COLS)
    life = df.groupby("unit")["cycle"].transform("max")
    cfg = SUBSET_CFG[subset]
    feats = (SETTINGS if cfg["settings"] else []) + [s for s in SENSORS if s not in cfg["drop"]]
    mn, mx = df[feats].min(), df[feats].max()
    X = ((df[feats] - mn) / (mx - mn).replace(0, 1)).to_numpy(np.float64)
    return {
        "X": X,
        "eid": df["unit"].to_numpy(),
        "age": df["cycle"].to_numpy(float),
        "life": life.to_numpy(float),
        "rul_true": (life - df["cycle"]).to_numpy(float),
        "features": feats,
    }


def split_by_engine(eid, seed, train_frac=0.7, cal_frac=0.15):
    rng = np.random.RandomState(seed)
    engines = np.unique(eid)
    rng.shuffle(engines)
    n = len(engines)
    a, b = int(train_frac * n), int(cal_frac * n)
    return (np.isin(eid, engines[:a]), np.isin(eid, engines[a:a + b]), np.isin(eid, engines[a + b:]))


if torch is not None:
    class MLP(nn.Module):
        def __init__(self, d, hidden=HIDDEN, out=1):
            super().__init__()
            layers, prev = [], d
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            layers.append(nn.Linear(prev, out))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            o = self.net(x)
            return o.squeeze(-1) if o.shape[-1] == 1 else o


    def train_mlp(X, y, seed, epochs=N_EPOCHS, lr=LR, batch=BATCH):
        """Same optimisation as the notebook (Adam, MSE, batch 256, 80 epochs, reshuffled every
        epoch); mini-batches are sliced from a random permutation instead of going through a
        DataLoader, which is ~10x faster on CPU and draws from the same distribution of batches."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        Xt = torch.from_numpy(X).float()
        yt = torch.from_numpy(y).float()
        n = Xt.shape[0]
        model = MLP(X.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                opt.zero_grad()
                loss_fn(model(Xt[idx]), yt[idx]).backward()
                opt.step()
        model.eval()
        return model


    @torch.no_grad()
    def predict(model, X):
        return model(torch.from_numpy(X).float()).numpy().astype(np.float64)
else:
    def _needs_torch(*args, **kwargs):
        raise ImportError("PyTorch is required to train the networks: pip install -r requirements-train.txt")

    MLP = train_mlp = predict = _needs_torch


def conformal_quantile(scores, alpha=ALPHA):
    n = len(scores)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    if level >= 1.0:
        return float(np.max(scores))
    return float(np.quantile(scores, level, method="higher"))


def project(L, U, lo=0.0, hi=None):
    """C ∩ K for K = [lo, hi(x)] with the notebook's empty-intersection guard L <= U."""
    L2 = np.maximum(L, lo)
    U2 = U if hi is None else np.minimum(U, hi)
    L2 = np.minimum(L2, U2)
    return L2, U2


def cov_width(L, U, y):
    return (y >= L) & (y <= U), np.maximum(U - L, 0.0)
