# -*- coding: utf-8 -*-
import os
import json
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

try:
    from scipy.stats import wilcoxon
    SCIPY_AVAILABLE = True
except Exception:
    wilcoxon = None
    SCIPY_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except Exception:
    LGBMRegressor = None
    LIGHTGBM_AVAILABLE = False

import torch
import torch.nn as nn
import torch.nn.functional as F


# CONFIG

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE_DIR, "outputs", "edge_samples_combined.csv")
GRAPH_CSV = os.path.join(BASE_DIR, "Edge_Distances3_.csv")
OUT_DIR = os.path.join(BASE_DIR, "asoc")

FEATURE_COLS = [
    "edge_distance",
    "mean_speed",
    "slowdown_idx",
    "std_speed",
    "turn_intensity",
    "mean_power_W",
    "std_power_W",
    "peak_power_W",
    "mean_current",
    "std_current",
    "wheel_diff_mean",
    "stop_ratio",
    "scanner_ratio",
    "obs_t_start",
]

TRAIN_SESSION = 1
TEST_SESSION = 2
VAL_RATIO = 0.20

GRAPH_MODELS = ["GGNN", "GCN", "GAT", "GraphSAGE"]
NON_GRAPH_MODELS = ["Ridge", "SVR-RBF", "KernelRidge-RBF", "LightGBM", "MLP"]

ENSEMBLE_SEEDS = [11, 21, 31, 41, 51]
ABLATION_SEEDS = [11, 21, 31, 41, 51]
ABLATION_GRAPH_ENCODER = "GGNN"

HIDDEN_DIM = 64
GGNN_STEPS = 3
DROPOUT = 0.15
EPOCHS = 250
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 30

TEST_CI_BOOTSTRAPS = 1000
CI_LEVEL = 95
SCALE_TARGETS_FOR_ALL_MODELS = True



# UTILITIES


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def metrics_dict(y_true_t, y_pred_t, y_true_e, y_pred_e) -> Dict[str, float]:
    return {
        "time_r2": float(r2_score(y_true_t, y_pred_t)),
        "time_mae": float(mean_absolute_error(y_true_t, y_pred_t)),
        "time_rmse": rmse(y_true_t, y_pred_t),
        "time_mape": mape(y_true_t, y_pred_t),
        "energy_r2": float(r2_score(y_true_e, y_pred_e)),
        "energy_mae": float(mean_absolute_error(y_true_e, y_pred_e)),
        "energy_rmse": rmse(y_true_e, y_pred_e),
        "energy_mape": mape(y_true_e, y_pred_e),
    }


def metric_value(metric_name: str, y_true_t, y_pred_t, y_true_e, y_pred_e) -> float:
    try:
        if metric_name == "time_r2":
            return float(r2_score(y_true_t, y_pred_t))
        if metric_name == "time_mae":
            return float(mean_absolute_error(y_true_t, y_pred_t))
        if metric_name == "time_rmse":
            return float(rmse(y_true_t, y_pred_t))
        if metric_name == "time_mape":
            return float(mape(y_true_t, y_pred_t))
        if metric_name == "energy_r2":
            return float(r2_score(y_true_e, y_pred_e))
        if metric_name == "energy_mae":
            return float(mean_absolute_error(y_true_e, y_pred_e))
        if metric_name == "energy_rmse":
            return float(rmse(y_true_e, y_pred_e))
        if metric_name == "energy_mape":
            return float(mape(y_true_e, y_pred_e))
    except Exception:
        return float("nan")
    raise ValueError(metric_name)


def bootstrap_metric_ci(
    y_true_t,
    y_pred_t,
    y_true_e,
    y_pred_e,
    n_boot: int = TEST_CI_BOOTSTRAPS,
    seed: int = SEED,
    ci_level: int = CI_LEVEL,
) -> Dict[str, float]:
    """Bootstrap test-set rows to report uncertainty caused by small held-out sample size."""
    rng = np.random.RandomState(seed)
    y_true_t = np.asarray(y_true_t)
    y_pred_t = np.asarray(y_pred_t)
    y_true_e = np.asarray(y_true_e)
    y_pred_e = np.asarray(y_pred_e)
    n = len(y_true_t)

    metrics = [
        "time_r2", "time_mae", "time_rmse", "time_mape",
        "energy_r2", "energy_mae", "energy_rmse", "energy_mape",
    ]
    vals = {m: [] for m in metrics}

    for _ in range(n_boot):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        for m in metrics:
            vals[m].append(
                metric_value(m, y_true_t[idx], y_pred_t[idx], y_true_e[idx], y_pred_e[idx])
            )

    out: Dict[str, float] = {}
    alpha = (100 - ci_level) / 2.0
    for m in metrics:
        arr = np.asarray(vals[m], dtype=float)
        arr = arr[np.isfinite(arr)]
        out[f"{m}_ci{ci_level}_low"] = float(np.percentile(arr, alpha)) if len(arr) else np.nan
        out[f"{m}_ci{ci_level}_high"] = float(np.percentile(arr, 100 - alpha)) if len(arr) else np.nan
        out[f"{m}_boot_std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan
    return out


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return np.nan
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def smart_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    try:
        df = pd.read_csv(path)
        if len(df.columns) == 1 and ";" in str(df.columns[0]):
            df = pd.read_csv(path, sep=";")
    except Exception:
        df = pd.read_csv(path, sep=";")
    return df


def find_first_existing_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Could not find any of {candidates}. Available columns: {list(df.columns)}")
    return None


def bootstrap_indices(n: int, rng: np.random.RandomState) -> np.ndarray:
    return rng.choice(np.arange(n), size=n, replace=True)


def existing_cols(cols: Sequence[str], df: pd.DataFrame) -> List[str]:
    return [c for c in cols if c in df.columns]



# DATA LOADING


def load_and_standardize_edge_samples(path: str) -> pd.DataFrame:
    df = smart_read_csv(path)

    session_col = find_first_existing_col(df, ["session_id", "session", "Session"])
    src_col = find_first_existing_col(df, ["u", "src", "source", "from", "u_node_id"])
    dst_col = find_first_existing_col(df, ["v", "dst", "target", "to", "v_node_id"])
    time_col = find_first_existing_col(df, ["target_time", "time_s", "travel_time_s", "t_target", "time"])
    energy_col = find_first_existing_col(df, ["target_energy", "energy_j", "energy_J", "travel_energy_j", "e_target", "energy"])

    df = df.rename(
        columns={
            session_col: "session",
            src_col: "u",
            dst_col: "v",
            time_col: "target_time",
            energy_col: "target_energy",
        }
    )

    required = ["session", "u", "v", "target_time", "target_energy"] + FEATURE_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nAvailable columns: {list(df.columns)}")

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=required).copy()
    df["session"] = df["session"].astype(int)
    df["u"] = df["u"].astype(int)
    df["v"] = df["v"].astype(int)
    return df


def chronological_train_val_split(df: pd.DataFrame, val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    time_col = None
    for c in ["obs_t_start", "timestamp", "time", "t_start"]:
        if c in df.columns:
            time_col = c
            break

    if time_col is not None:
        df = df.sort_values(time_col).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    n = len(df)
    split = int((1.0 - val_ratio) * n)
    train_df = df.iloc[:split].copy()
    val_df = df.iloc[split:].copy()
    return train_df, val_df


def build_node_mapping(df: pd.DataFrame) -> Dict[int, int]:
    nodes = sorted(set(df["u"].astype(int).tolist()) | set(df["v"].astype(int).tolist()))
    return {nid: i for i, nid in enumerate(nodes)}


def make_seen_unseen_mask(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    seen_edges = set(zip(train_df["u"].astype(int), train_df["v"].astype(int)))
    return np.array([(s, d) in seen_edges for s, d in zip(test_df["u"], test_df["v"])], dtype=bool)


def build_train_support_map(train_df: pd.DataFrame) -> Dict[Tuple[int, int], int]:
    counts = train_df.groupby(["u", "v"]).size()
    return {tuple(k): int(v) for k, v in counts.items()}


def assign_support_bucket(count: int) -> str:
    if count == 0:
        return "unseen"
    if count <= 2:
        return "low_1_2"
    if count <= 5:
        return "medium_3_5"
    return "high_gt5"



# SANITY CHECKS


def run_sanity_checks(df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, object]:
    report: Dict[str, object] = {}
    report["train_sessions"] = sorted(train_df["session"].unique().tolist())
    report["val_sessions"] = sorted(val_df["session"].unique().tolist())
    report["test_sessions"] = sorted(test_df["session"].unique().tolist())
    report["train_size"] = len(train_df)
    report["val_size"] = len(val_df)
    report["test_size"] = len(test_df)

    key_cols = ["u", "v", "target_time", "target_energy"] + FEATURE_COLS
    train_sig = set(map(tuple, np.round(train_df[key_cols].to_numpy(), 6)))
    test_sig = set(map(tuple, np.round(test_df[key_cols].to_numpy(), 6)))
    report["exact_train_test_overlap_rows"] = len(train_sig.intersection(test_sig))

    train_edges = set(zip(train_df["u"], train_df["v"]))
    test_edges = set(zip(test_df["u"], test_df["v"]))
    overlap_edges = train_edges.intersection(test_edges)
    report["train_unique_edges"] = len(train_edges)
    report["test_unique_edges"] = len(test_edges)
    report["overlap_unique_edges"] = len(overlap_edges)
    report["unseen_test_edges"] = len(test_edges - train_edges)
    return report



# GRAPH PREP


def load_graph_edges(path: str, node_to_idx: Dict[int, int], fallback_df: pd.DataFrame) -> torch.Tensor:
    edges = []

    if os.path.exists(path):
        try:
            gdf = smart_read_csv(path)
            src_col = find_first_existing_col(
                gdf,
                ["src", "source", "from", "u", "u_node_id", "Start_Node", "start", "node_from"],
                required=False,
            )
            dst_col = find_first_existing_col(
                gdf,
                ["dst", "target", "to", "v", "v_node_id", "End_Node", "end", "node_to"],
                required=False,
            )

            if src_col is not None and dst_col is not None:
                for s, d in zip(gdf[src_col], gdf[dst_col]):
                    try:
                        s = int(float(s))
                        d = int(float(d))
                        if s in node_to_idx and d in node_to_idx:
                            edges.append([node_to_idx[s], node_to_idx[d]])
                    except Exception:
                        pass
        except Exception as e:
            print(f"Warning: graph file parse failed, fallback to observed graph. Reason: {e}")

    if len(edges) == 0:
        observed_edges = fallback_df[["u", "v"]].drop_duplicates().astype(int).values.tolist()
        edges = [[node_to_idx[s], node_to_idx[d]] for s, d in observed_edges]

    if not edges:
        raise ValueError("No valid graph edges could be built.")

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def make_empty_edge_index() -> torch.Tensor:
    return torch.empty((2, 0), dtype=torch.long)


def make_random_edge_index(n_nodes: int, n_edges: int, seed: int = SEED) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    edges = []
    for _ in range(n_edges):
        s = int(rng.randint(0, n_nodes))
        d = int(rng.randint(0, n_nodes))
        if n_nodes > 1:
            while d == s:
                d = int(rng.randint(0, n_nodes))
        edges.append([s, d])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def compute_node_features(train_df: pd.DataFrame, node_to_idx: Dict[int, int]) -> np.ndarray:
    n_nodes = len(node_to_idx)
    arr = np.zeros((n_nodes, 4), dtype=np.float32)

    out_counts = train_df.groupby("u").size().to_dict()
    in_counts = train_df.groupby("v").size().to_dict()
    mean_time_src = train_df.groupby("u")["target_time"].mean().to_dict()
    mean_energy_src = train_df.groupby("u")["target_energy"].mean().to_dict()

    for node_id, idx in node_to_idx.items():
        arr[idx, 0] = float(out_counts.get(node_id, 0))
        arr[idx, 1] = float(in_counts.get(node_id, 0))
        arr[idx, 2] = float(mean_time_src.get(node_id, 0.0))
        arr[idx, 3] = float(mean_energy_src.get(node_id, 0.0))
    return arr



# TENSOR PACK


@dataclass
class TensorPack:
    x_edge: torch.Tensor
    y: torch.Tensor
    src_idx: np.ndarray
    dst_idx: np.ndarray


def to_tensor_pack(df: pd.DataFrame, node_to_idx: Dict[int, int], feature_cols: Sequence[str] = FEATURE_COLS) -> TensorPack:
    src_idx = df["u"].astype(int).map(node_to_idx).to_numpy()
    dst_idx = df["v"].astype(int).map(node_to_idx).to_numpy()
    x_edge = torch.tensor(df[list(feature_cols)].to_numpy(dtype=np.float32), dtype=torch.float32)
    y = torch.tensor(df[["target_time", "target_energy"]].to_numpy(dtype=np.float32), dtype=torch.float32)
    return TensorPack(x_edge=x_edge, y=y, src_idx=src_idx, dst_idx=dst_idx)

def scale_tensor_pack_targets(pack: TensorPack, y_scaler: StandardScaler) -> TensorPack:
    """Return a copy of a TensorPack with y scaled by train-only target scaler."""
    y_scaled = y_scaler.transform(pack.y.cpu().numpy()).astype(np.float32)
    return TensorPack(
        x_edge=pack.x_edge,
        y=torch.tensor(y_scaled, dtype=torch.float32),
        src_idx=pack.src_idx,
        dst_idx=pack.dst_idx,
    )



# GRAPH MODELS


class BaseGraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)

    def aggregate(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, node_x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(node_x)
        return self.aggregate(h, edge_index)


class GGNNEncoder(BaseGraphEncoder):
    def __init__(self, in_dim: int, hidden_dim: int, steps: int = 3):
        super().__init__(in_dim, hidden_dim)
        self.steps = steps
        self.msg = nn.Linear(hidden_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def aggregate(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        for _ in range(self.steps):
            m = torch.zeros_like(h)
            if src.numel() > 0:
                msgs = self.msg(h[src])
                m.index_add_(0, dst, msgs)
            h = self.gru(m, h)
        return h


class GCNEncoder(BaseGraphEncoder):
    def aggregate(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        out = torch.zeros_like(h)
        deg = torch.zeros(h.size(0), device=h.device)
        if src.numel() > 0:
            out.index_add_(0, dst, h[src])
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        deg = deg.clamp(min=1.0).unsqueeze(1)
        return F.relu(out / deg)


class GraphSAGEEncoder(BaseGraphEncoder):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__(in_dim, hidden_dim)
        self.combine = nn.Linear(hidden_dim * 2, hidden_dim)

    def aggregate(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        agg = torch.zeros_like(h)
        deg = torch.zeros(h.size(0), device=h.device)
        if src.numel() > 0:
            agg.index_add_(0, dst, h[src])
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        deg = deg.clamp(min=1.0).unsqueeze(1)
        agg = agg / deg
        out = torch.cat([h, agg], dim=1)
        return F.relu(self.combine(out))


class GATEncoder(BaseGraphEncoder):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__(in_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def aggregate(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        out = torch.zeros_like(h)
        norm = torch.zeros(h.size(0), device=h.device)
        if src.numel() > 0:
            hs = h[src]
            hd = h[dst]
            a = self.attn(torch.cat([hs, hd], dim=1)).squeeze(1)
            a = torch.sigmoid(a)
            msgs = hs * a.unsqueeze(1)
            out.index_add_(0, dst, msgs)
            norm.index_add_(0, dst, a)
        norm = norm.clamp(min=1e-6).unsqueeze(1)
        return F.relu(out / norm)


class GraphRegressor(nn.Module):
    def __init__(self, encoder_name: str, node_in_dim: int, edge_in_dim: int, hidden_dim: int = 64, steps: int = 3, dropout: float = 0.1):
        super().__init__()
        if encoder_name == "GGNN":
            self.encoder = GGNNEncoder(node_in_dim, hidden_dim, steps=steps)
        elif encoder_name == "GCN":
            self.encoder = GCNEncoder(node_in_dim, hidden_dim)
        elif encoder_name == "GAT":
            self.encoder = GATEncoder(node_in_dim, hidden_dim)
        elif encoder_name == "GraphSAGE":
            self.encoder = GraphSAGEEncoder(node_in_dim, hidden_dim)
        else:
            raise ValueError(f"Unsupported encoder_name: {encoder_name}")

        fusion_dim = hidden_dim * 2 + edge_in_dim
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, node_x: torch.Tensor, edge_index: torch.Tensor, edge_feat: torch.Tensor, src_idx: torch.Tensor, dst_idx: torch.Tensor):
        node_emb = self.encoder(node_x, edge_index)
        h_src = node_emb[src_idx]
        h_dst = node_emb[dst_idx]
        z = torch.cat([h_src, h_dst, edge_feat], dim=1)
        return self.regressor(z)



# GRAPH TRAINING


def train_graph_model(
    encoder_name: str,
    train_pack: TensorPack,
    val_pack: TensorPack,
    node_x: torch.Tensor,
    edge_index: torch.Tensor,
    seed: int,
) -> GraphRegressor:
    set_seed(seed)
    model = GraphRegressor(
        encoder_name=encoder_name,
        node_in_dim=node_x.shape[1],
        edge_in_dim=train_pack.x_edge.shape[1],
        hidden_dim=HIDDEN_DIM,
        steps=GGNN_STEPS,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.HuberLoss()

    node_x = node_x.to(DEVICE)
    edge_index = edge_index.to(DEVICE)
    tr_src = torch.tensor(train_pack.src_idx, dtype=torch.long, device=DEVICE)
    tr_dst = torch.tensor(train_pack.dst_idx, dtype=torch.long, device=DEVICE)
    va_src = torch.tensor(val_pack.src_idx, dtype=torch.long, device=DEVICE)
    va_dst = torch.tensor(val_pack.dst_idx, dtype=torch.long, device=DEVICE)
    xtr = train_pack.x_edge.to(DEVICE)
    ytr = train_pack.y.to(DEVICE)
    xva = val_pack.x_edge.to(DEVICE)
    yva = val_pack.y.to(DEVICE)

    best_state = None
    best_val = float("inf")
    wait = 0

    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = model(node_x, edge_index, xtr, tr_src, tr_dst)
        loss = loss_fn(pred, ytr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(node_x, edge_index, xva, va_src, va_dst)
            val_loss = loss_fn(pred_val, yva).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def infer_graph_model(
    model: GraphRegressor,
    pack: TensorPack,
    node_x: torch.Tensor,
    edge_index: torch.Tensor,
    y_scaler: Optional[StandardScaler] = None,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        src = torch.tensor(pack.src_idx, dtype=torch.long, device=DEVICE)
        dst = torch.tensor(pack.dst_idx, dtype=torch.long, device=DEVICE)
        pred = model(
            node_x.to(DEVICE),
            edge_index.to(DEVICE),
            pack.x_edge.to(DEVICE),
            src,
            dst,
        ).cpu().numpy()
    if y_scaler is not None:
        pred = y_scaler.inverse_transform(pred)
    return pred


def graph_ensemble_predict(
    encoder_name: str,
    train_pack: TensorPack,
    val_pack: TensorPack,
    test_pack: TensorPack,
    node_x: torch.Tensor,
    edge_index: torch.Tensor,
    seeds: Sequence[int],
    y_scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    preds = []
    for seed in seeds:
        model = train_graph_model(encoder_name, train_pack, val_pack, node_x, edge_index, seed)
        pred = infer_graph_model(model, test_pack, node_x, edge_index, y_scaler=y_scaler)
        preds.append(pred)
    preds = np.stack(preds, axis=0)
    return preds.mean(axis=0), preds.std(axis=0)



# NON-GRAPH MODEL FACTORIES


def make_ridge(seed: int):
    return Ridge(alpha=1.0)


def make_svr(seed: int):
    return MultiOutputRegressor(SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale"))


def make_krr(seed: int):
    return KernelRidge(alpha=1.0, kernel="rbf", gamma=None)


def make_lightgbm(seed: int):
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM is not installed. Install it with: pip install lightgbm")
    return MultiOutputRegressor(
        LGBMRegressor(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=4,
            min_child_samples=5,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed,
            verbosity=-1,
        )
    )


class NonGraphMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_non_graph_mlp(
    x_train: np.ndarray,
    y_train_scaled: np.ndarray,
    x_val: np.ndarray,
    y_val_scaled: np.ndarray,
    seed: int,
) -> NonGraphMLP:
    """Train non-graph MLP on already target-scaled y for fairness with graph neural models."""
    set_seed(seed)

    model = NonGraphMLP(in_dim=x_train.shape[1], hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.HuberLoss()

    xtr = torch.tensor(x_train, dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor(y_train_scaled.astype(np.float32), dtype=torch.float32, device=DEVICE)
    xva = torch.tensor(x_val, dtype=torch.float32, device=DEVICE)
    yva = torch.tensor(y_val_scaled.astype(np.float32), dtype=torch.float32, device=DEVICE)

    best_state = None
    best_val = float("inf")
    wait = 0

    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(xtr), ytr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(xva), yva).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def infer_non_graph_mlp(model: NonGraphMLP, x: np.ndarray, y_scaler: StandardScaler) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred_s = model(torch.tensor(x, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    return y_scaler.inverse_transform(pred_s)

def non_graph_bootstrap_predict(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    seeds: Sequence[int],
    y_scaler: StandardScaler,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Train a 5-member ensemble for each non-graph model.

    Fairness rule used here:
    - every model receives the same standardized input features;
    - every model learns standardized targets fitted on the training set only;
    - predictions are inverse-transformed before metrics are computed;
    - each reported model uses 5 ensemble members.
    """
    factories = {
        "Ridge": make_ridge,
        "SVR-RBF": make_svr,
        "KernelRidge-RBF": make_krr,
        "LightGBM": make_lightgbm,
    }

    if model_name == "LightGBM" and not LIGHTGBM_AVAILABLE:
        print("    LightGBM not available in this Python interpreter; skipping.")
        print("    Use /opt/anaconda3/bin/python or install LightGBM in the active interpreter.")
        return None, None

    y_train_scaled_all = y_scaler.transform(y_train)
    y_val_scaled = y_scaler.transform(y_val)

    preds = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        idx = bootstrap_indices(len(x_train), rng)
        x_boot = x_train[idx]
        y_boot_scaled = y_train_scaled_all[idx]

        if model_name == "MLP":
            model = train_non_graph_mlp(x_boot, y_boot_scaled, x_val, y_val_scaled, seed)
            pred = infer_non_graph_mlp(model, x_test, y_scaler)
        else:
            model = factories[model_name](seed)
            model.fit(x_boot, y_boot_scaled)
            pred_scaled = model.predict(x_test)
            pred = y_scaler.inverse_transform(pred_scaled)

        preds.append(pred)

    preds = np.stack(preds, axis=0)
    return preds.mean(axis=0), preds.std(axis=0)



# ANALYSIS HELPERS


def evaluate_subset_rows(model_name: str, subset_name: str, y_true_t, y_pred_t, y_true_e, y_pred_e) -> Dict[str, object]:
    row = metrics_dict(y_true_t, y_pred_t, y_true_e, y_pred_e)
    row["model"] = model_name
    row["subset"] = subset_name
    row["n"] = len(y_true_t)
    return row


def build_support_bucket_df(detailed_df: pd.DataFrame, pred_cols_map: Dict[str, Tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for model_name, (pt_col, pe_col) in pred_cols_map.items():
        for bucket, sub in detailed_df.groupby("support_bucket"):
            if len(sub) == 0:
                continue
            rows.append(
                evaluate_subset_rows(
                    model_name,
                    bucket,
                    sub["target_time"].to_numpy(),
                    sub[pt_col].to_numpy(),
                    sub["target_energy"].to_numpy(),
                    sub[pe_col].to_numpy(),
                )
            )
    return pd.DataFrame(rows)


def build_seen_unseen_df(detailed_df: pd.DataFrame, pred_cols_map: Dict[str, Tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for model_name, (pt_col, pe_col) in pred_cols_map.items():
        for label, sub in detailed_df.groupby("seen_in_train"):
            name = "seen" if label else "unseen"
            if len(sub) == 0:
                continue
            rows.append(
                evaluate_subset_rows(
                    model_name,
                    name,
                    sub["target_time"].to_numpy(),
                    sub[pt_col].to_numpy(),
                    sub["target_energy"].to_numpy(),
                    sub[pe_col].to_numpy(),
                )
            )
    return pd.DataFrame(rows)


def build_uncertainty_corr_df(detailed_df: pd.DataFrame, uncertainty_cols_map: Dict[str, Tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for model_name, (ut_col, ue_col) in uncertainty_cols_map.items():
        rows.append(
            {
                "model": model_name,
                "time_unc_err_corr": safe_corr(
                    detailed_df[ut_col].to_numpy(),
                    detailed_df[f"{model_name}_time_abs_err"].to_numpy(),
                ),
                "energy_unc_err_corr": safe_corr(
                    detailed_df[ue_col].to_numpy(),
                    detailed_df[f"{model_name}_energy_abs_err"].to_numpy(),
                ),
            }
        )
    return pd.DataFrame(rows)


def build_selective_prediction_df(detailed_df: pd.DataFrame, uncertainty_cols_map: Dict[str, Tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for model_name, (ut_col, _) in uncertainty_cols_map.items():
        sub = detailed_df.sort_values(ut_col, ascending=True).reset_index(drop=True)
        for keep_frac in [1.0, 0.9, 0.8, 0.7]:
            k = max(1, int(len(sub) * keep_frac))
            kept = sub.iloc[:k]
            rows.append(
                {
                    "model": model_name,
                    "keep_fraction": keep_frac,
                    "samples": len(kept),
                    "time_mae": float(kept[f"{model_name}_time_abs_err"].mean()),
                    "energy_mae": float(kept[f"{model_name}_energy_abs_err"].mean()),
                }
            )
    return pd.DataFrame(rows)

def build_behavior_error_df(
    detailed_df: pd.DataFrame,
    pred_cols_map: Dict[str, Tuple[str, str]],
    feature_source_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Behavior-aware error analysis for reviewer response:
    - turning-heavy vs non-turning-heavy samples
    - stop-heavy vs non-stop-heavy samples
    - slowdown-heavy vs non-slowdown-heavy samples

    The threshold is computed from the held-out test session using the 75th percentile.
    """
    rows = []
    work = detailed_df.copy().reset_index(drop=True)
    feat = feature_source_df.copy().reset_index(drop=True)

    candidate_features = [
        ("turn_intensity", "turning-heavy"),
        ("stop_ratio", "stop-heavy"),
        ("slowdown_idx", "slowdown-heavy"),
    ]

    for feature_col, label_name in candidate_features:
        if feature_col not in feat.columns:
            continue

        values = feat[feature_col].to_numpy(dtype=float)
        threshold = float(np.nanpercentile(values, 75))

        high_mask = values >= threshold
        low_mask = values < threshold

        for model_name, (pt_col, pe_col) in pred_cols_map.items():
            if pt_col not in work.columns or pe_col not in work.columns:
                continue

            for group_name, mask in [
                (f"{label_name}_high_q75", high_mask),
                (f"{label_name}_low_below_q75", low_mask),
            ]:
                sub = work.loc[mask].copy()
                if len(sub) == 0:
                    continue

                row = evaluate_subset_rows(
                    model_name,
                    group_name,
                    sub["target_time"].to_numpy(),
                    sub[pt_col].to_numpy(),
                    sub["target_energy"].to_numpy(),
                    sub[pe_col].to_numpy(),
                )
                row["feature"] = feature_col
                row["threshold_q75"] = threshold
                rows.append(row)

    return pd.DataFrame(rows)


def build_difficult_cases_df(
    detailed_df: pd.DataFrame,
    model_name: str = "GraphSAGE",
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Extract the most difficult prediction cases for one representative model.
    Ranking is based on the average of normalized absolute time and energy errors.
    """
    time_err_col = f"{model_name}_time_abs_err"
    energy_err_col = f"{model_name}_energy_abs_err"

    if time_err_col not in detailed_df.columns or energy_err_col not in detailed_df.columns:
        return pd.DataFrame()

    work = detailed_df.copy()

    time_scale = max(float(work[time_err_col].median()), 1e-8)
    energy_scale = max(float(work[energy_err_col].median()), 1e-8)

    work["difficulty_score"] = 0.5 * (
        work[time_err_col] / time_scale + work[energy_err_col] / energy_scale
    )

    cols = [
        "session",
        "u",
        "v",
        "seen_in_train",
        "train_support_count",
        "support_bucket",
        "target_time",
        f"{model_name}_time_pred",
        time_err_col,
        "target_energy",
        f"{model_name}_energy_pred",
        energy_err_col,
        "difficulty_score",
    ]

    cols = [c for c in cols if c in work.columns]

    return work.sort_values("difficulty_score", ascending=False)[cols].head(top_k)

def add_ci_to_row(row: Dict[str, object], y_true_t, y_pred_t, y_true_e, y_pred_e) -> Dict[str, object]:
    row.update(bootstrap_metric_ci(y_true_t, y_pred_t, y_true_e, y_pred_e))
    return row



# PAPER TABLES


def build_asoc_eswa_difference_table(out_dir: str) -> pd.DataFrame:
    rows = [
        {
            "dimension": "Main scientific aim",
            "ASOC_benchmark_paper": "Estimator-behavior benchmark under sparse cross-session AGV telemetry",
            "ESWA_framework_paper": "Planner-compatible decision-support framework for learned edge costs",
        },
        {
            "dimension": "Core contribution",
            "ASOC_benchmark_paper": "Comparison of graph and non-graph regressors with robustness, support-sensitivity and uncertainty analysis",
            "ESWA_framework_paper": "Integration of learned edge costs with graph-search planning and temporal trajectory realization",
        },
        {
            "dimension": "Models emphasized",
            "ASOC_benchmark_paper": "GGNN, GCN, GAT, GraphSAGE, Ridge, SVR-RBF, KernelRidge-RBF, LightGBM, and MLP",
            "ESWA_framework_paper": "An inductive GNN edge-cost module embedded in a decision-support pipeline",
        },
        {
            "dimension": "Evaluation level",
            "ASOC_benchmark_paper": "Cross-session edge prediction, seen/unseen edges, support buckets, uncertainty-error correlation and selective prediction",
            "ESWA_framework_paper": "Edge-level prediction, route-level route comparison, mission-level ETA/energy and temporal rollout",
        },
        {
            "dimension": "What is intentionally excluded",
            "ASOC_benchmark_paper": "No route-planner deployment claim, no MES/OPC-UA runtime bridge, no temporal-trajectory rollout claim",
            "ESWA_framework_paper": "Not designed as a broad estimator-family benchmark or reliability-ranking study",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "table_asoc_vs_eswa_difference.csv"), index=False)
    return df


def build_model_implementation_table(out_dir: str) -> pd.DataFrame:
    rows = [
        {
            "model": "Ridge",
            "family": "Linear non-graph",
            "implementation": "scikit-learn Ridge",
            "main_hyperparameters": "alpha=1.0",
            "uncertainty_method": "bootstrap ensemble over training samples",
        },
        {
            "model": "SVR-RBF",
            "family": "Kernel non-graph",
            "implementation": "scikit-learn SVR with MultiOutputRegressor",
            "main_hyperparameters": "kernel=RBF, C=10.0, epsilon=0.1, gamma=scale",
            "uncertainty_method": "bootstrap ensemble over training samples",
        },
        {
            "model": "KernelRidge-RBF",
            "family": "Kernel non-graph",
            "implementation": "scikit-learn KernelRidge",
            "main_hyperparameters": "alpha=1.0, kernel=RBF, gamma=None/default",
            "uncertainty_method": "bootstrap ensemble over training samples",
        },
        {
            "model": "LightGBM",
            "family": "Gradient-boosted tree non-graph",
            "implementation": "LightGBM LGBMRegressor with MultiOutputRegressor",
            "main_hyperparameters": "n_estimators=300, learning_rate=0.03, num_leaves=15, max_depth=4",
            "uncertainty_method": "bootstrap ensemble over training samples",
        },
        {
            "model": "MLP",
            "family": "Neural non-graph",
            "implementation": "PyTorch two-hidden-layer feed-forward network",
            "main_hyperparameters": f"hidden_dim={HIDDEN_DIM}, dropout={DROPOUT}, HuberLoss, Adam lr={LR}, weight_decay={WEIGHT_DECAY}",
            "uncertainty_method": "bootstrap ensemble over training samples",
        },
        {
            "model": "GGNN",
            "family": "Graph neural network",
            "implementation": "PyTorch gated message-passing encoder + regression head",
            "main_hyperparameters": f"hidden_dim={HIDDEN_DIM}, steps={GGNN_STEPS}, dropout={DROPOUT}, HuberLoss, Adam lr={LR}",
            "uncertainty_method": "multi-seed ensemble",
        },
        {
            "model": "GCN",
            "family": "Graph neural network",
            "implementation": "PyTorch mean-neighborhood graph convolution + regression head",
            "main_hyperparameters": f"hidden_dim={HIDDEN_DIM}, dropout={DROPOUT}, HuberLoss, Adam lr={LR}",
            "uncertainty_method": "multi-seed ensemble",
        },
        {
            "model": "GAT",
            "family": "Graph neural network",
            "implementation": "PyTorch single-head attention-style aggregation + regression head",
            "main_hyperparameters": f"hidden_dim={HIDDEN_DIM}, dropout={DROPOUT}, HuberLoss, Adam lr={LR}",
            "uncertainty_method": "multi-seed ensemble",
        },
        {
            "model": "GraphSAGE",
            "family": "Graph neural network",
            "implementation": "PyTorch mean-aggregation GraphSAGE-style encoder + regression head",
            "main_hyperparameters": f"hidden_dim={HIDDEN_DIM}, dropout={DROPOUT}, HuberLoss, Adam lr={LR}",
            "uncertainty_method": "multi-seed ensemble",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "table_model_implementation.csv"), index=False)
    return df



# ABLATION


def build_scaled_variant_frames(
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_v = train_raw.copy()
    val_v = val_raw.copy()
    test_v = test_raw.copy()
    scaler = StandardScaler()
    scaler.fit(train_v[list(feature_cols)].to_numpy())
    train_v.loc[:, list(feature_cols)] = scaler.transform(train_v[list(feature_cols)].to_numpy())
    val_v.loc[:, list(feature_cols)] = scaler.transform(val_v[list(feature_cols)].to_numpy())
    test_v.loc[:, list(feature_cols)] = scaler.transform(test_v[list(feature_cols)].to_numpy())
    return train_v, val_v, test_v


def run_graph_ablation_variant(
    variant_name: str,
    feature_cols: Sequence[str],
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    node_to_idx: Dict[int, int],
    node_x_variant: torch.Tensor,
    edge_index_variant: torch.Tensor,
    y_true_t: np.ndarray,
    y_true_e: np.ndarray,
    y_scaler: StandardScaler,
) -> Dict[str, object]:
    train_v, val_v, test_v = build_scaled_variant_frames(train_raw, val_raw, test_raw, feature_cols)
    train_pack_v_raw = to_tensor_pack(train_v, node_to_idx, feature_cols=feature_cols)
    val_pack_v_raw = to_tensor_pack(val_v, node_to_idx, feature_cols=feature_cols)
    test_pack_v = to_tensor_pack(test_v, node_to_idx, feature_cols=feature_cols)
    train_pack_v = scale_tensor_pack_targets(train_pack_v_raw, y_scaler)
    val_pack_v = scale_tensor_pack_targets(val_pack_v_raw, y_scaler)
    pred_mean, pred_std = graph_ensemble_predict(
        ABLATION_GRAPH_ENCODER,
        train_pack_v,
        val_pack_v,
        test_pack_v,
        node_x_variant,
        edge_index_variant,
        ABLATION_SEEDS,
        y_scaler=y_scaler,
    )
    row = metrics_dict(y_true_t, pred_mean[:, 0], y_true_e, pred_mean[:, 1])
    row.update(bootstrap_metric_ci(y_true_t, pred_mean[:, 0], y_true_e, pred_mean[:, 1], n_boot=1000))
    row.update(
        {
            "variant": variant_name,
            "base_model": ABLATION_GRAPH_ENCODER,
            "feature_count": len(feature_cols),
            "features": ";".join(feature_cols),
            "time_unc_mean": float(pred_std[:, 0].mean()),
            "energy_unc_mean": float(pred_std[:, 1].mean()),
        }
    )
    return row


def run_mlp_ablation_variant(
    variant_name: str,
    feature_cols: Sequence[str],
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    y_true_t: np.ndarray,
    y_true_e: np.ndarray,
    y_scaler: StandardScaler,
) -> Dict[str, object]:
    train_v, val_v, test_v = build_scaled_variant_frames(train_raw, val_raw, test_raw, feature_cols)
    x_train = train_v[list(feature_cols)].to_numpy()
    y_train = train_v[["target_time", "target_energy"]].to_numpy()
    x_val = val_v[list(feature_cols)].to_numpy()
    y_val = val_v[["target_time", "target_energy"]].to_numpy()
    x_test = test_v[list(feature_cols)].to_numpy()
    pred_mean, pred_std = non_graph_bootstrap_predict("MLP", x_train, y_train, x_val, y_val, x_test, ABLATION_SEEDS, y_scaler=y_scaler)
    if pred_mean is None:
        raise RuntimeError("MLP ablation failed unexpectedly.")
    row = metrics_dict(y_true_t, pred_mean[:, 0], y_true_e, pred_mean[:, 1])
    row.update(bootstrap_metric_ci(y_true_t, pred_mean[:, 0], y_true_e, pred_mean[:, 1], n_boot=1000))
    row.update(
        {
            "variant": variant_name,
            "base_model": "MLP",
            "feature_count": len(feature_cols),
            "features": ";".join(feature_cols),
            "time_unc_mean": float(pred_std[:, 0].mean()),
            "energy_unc_mean": float(pred_std[:, 1].mean()),
        }
    )
    return row


def build_feature_graph_ablation_table(
    out_dir: str,
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    node_to_idx: Dict[int, int],
    edge_index_real: torch.Tensor,
    y_scaler: StandardScaler,
) -> pd.DataFrame:
    print("\nRunning feature/graph-context ablation...")
    y_true_t = test_raw["target_time"].to_numpy()
    y_true_e = test_raw["target_energy"].to_numpy()

    node_features = compute_node_features(train_raw, node_to_idx)
    node_scaler = StandardScaler()
    node_x_real = torch.tensor(node_scaler.fit_transform(node_features).astype(np.float32), dtype=torch.float32)
    node_x_zero = torch.zeros_like(node_x_real)
    edge_index_empty = make_empty_edge_index()
    edge_index_random = make_random_edge_index(
        n_nodes=len(node_to_idx),
        n_edges=edge_index_real.shape[1],
        seed=SEED + 999,
    )

    distance_only = existing_cols(["edge_distance"], train_raw)
    kinematic = existing_cols(
        [
            "edge_distance",
            "mean_speed",
            "slowdown_idx",
            "std_speed",
            "turn_intensity",
            "wheel_diff_mean",
            "stop_ratio",
            "scanner_ratio",
            "obs_t_start",
        ],
        train_raw,
    )
    power_motion = existing_cols(
        [
            "edge_distance",
            "mean_power_W",
            "std_power_W",
            "peak_power_W",
            "mean_current",
            "std_current",
            "mean_speed",
            "slowdown_idx",
            "stop_ratio",
        ],
        train_raw,
    )

    rows = []

    variants = [
        ("A1_full_features_real_graph", FEATURE_COLS, node_x_real, edge_index_real),
        ("A2_full_features_no_message_passing_edges", FEATURE_COLS, node_x_real, edge_index_empty),
        ("A3_full_features_random_topology", FEATURE_COLS, node_x_real, edge_index_random),
        ("A4_full_features_zero_node_context", FEATURE_COLS, node_x_zero, edge_index_real),
        ("A5_distance_only_real_graph", distance_only, node_x_real, edge_index_real),
        ("A6_kinematic_features_real_graph", kinematic, node_x_real, edge_index_real),
        ("A7_power_motion_features_real_graph", power_motion, node_x_real, edge_index_real),
    ]

    for variant_name, cols, node_x_v, edge_index_v in variants:
        if len(cols) == 0:
            print(f"  Skipping {variant_name}: no valid feature columns")
            continue
        print(f"  {variant_name}")
        rows.append(
            run_graph_ablation_variant(
                variant_name,
                cols,
                train_raw,
                val_raw,
                test_raw,
                node_to_idx,
                node_x_v,
                edge_index_v,
                y_true_t,
                y_true_e,
                y_scaler,
            )
        )

    print("  A8_full_features_non_graph_MLP")
    rows.append(
        run_mlp_ablation_variant(
            "A8_full_features_non_graph_MLP",
            FEATURE_COLS,
            train_raw,
            val_raw,
            test_raw,
            y_true_t,
            y_true_e,
            y_scaler,
        )
    )

    df = pd.DataFrame(rows)
    lead_cols = ["variant", "base_model", "feature_count", "time_r2", "time_mae", "energy_r2", "energy_mae"]
    other_cols = [c for c in df.columns if c not in lead_cols]
    df = df[lead_cols + other_cols]
    df.to_csv(os.path.join(out_dir, "table_feature_graph_ablation.csv"), index=False)
    return df



# MAIN


def main():
    set_seed(SEED)
    ensure_dir(OUT_DIR)

    print(f"Device: {DEVICE}")
    print(f"LightGBM available: {LIGHTGBM_AVAILABLE}")
    if not LIGHTGBM_AVAILABLE:
        print("Install LightGBM with: pip install lightgbm")
    print("Loading data...")

    df = load_and_standardize_edge_samples(DATA_CSV)
    trainval_df = df[df["session"] == TRAIN_SESSION].copy().reset_index(drop=True)
    test_raw = df[df["session"] == TEST_SESSION].copy().reset_index(drop=True)

    if len(trainval_df) == 0 or len(test_raw) == 0:
        raise ValueError("Train/test session split is empty. Check session ids.")

    train_raw, val_raw = chronological_train_val_split(trainval_df, val_ratio=VAL_RATIO)

    print(f"Train/Val/Test sizes: {len(train_raw)} / {len(val_raw)} / {len(test_raw)}")
    print(f"Edge features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

    sanity_report = run_sanity_checks(df, train_raw, val_raw, test_raw)
    with open(os.path.join(OUT_DIR, "sanity_report.json"), "w", encoding="utf-8") as f:
        json.dump(sanity_report, f, indent=2)

    print("\nSanity report summary:")
    print(f"  Train sessions: {sanity_report['train_sessions']}")
    print(f"  Val sessions:   {sanity_report['val_sessions']}")
    print(f"  Test sessions:  {sanity_report['test_sessions']}")
    print(f"  Exact train-test overlap rows: {sanity_report['exact_train_test_overlap_rows']}")
    print(f"  Overlap unique edges: {sanity_report['overlap_unique_edges']}")
    print(f"  Unseen test edges: {sanity_report['unseen_test_edges']}")

    node_to_idx = build_node_mapping(df)
    edge_index = load_graph_edges(GRAPH_CSV, node_to_idx, df)

    # Scale feature columns using train statistics only.
    x_scaler = StandardScaler()
    x_scaler.fit(train_raw[FEATURE_COLS].to_numpy())
    train_df = train_raw.copy()
    val_df = val_raw.copy()
    test_df = test_raw.copy()
    train_df.loc[:, FEATURE_COLS] = x_scaler.transform(train_df[FEATURE_COLS].to_numpy())
    val_df.loc[:, FEATURE_COLS] = x_scaler.transform(val_df[FEATURE_COLS].to_numpy())
    test_df.loc[:, FEATURE_COLS] = x_scaler.transform(test_df[FEATURE_COLS].to_numpy())

    # Target scaling is fitted on training data only and used for every model.
    # This avoids giving neural/non-neural models unequal numerical optimization conditions.
    y_scaler = StandardScaler()
    y_scaler.fit(train_df[["target_time", "target_energy"]].to_numpy())

    node_features = compute_node_features(train_raw, node_to_idx)
    node_scaler = StandardScaler()
    node_features_scaled = node_scaler.fit_transform(node_features).astype(np.float32)
    node_x = torch.tensor(node_features_scaled, dtype=torch.float32)

    train_pack_raw = to_tensor_pack(train_df, node_to_idx)
    val_pack_raw = to_tensor_pack(val_df, node_to_idx)
    test_pack = to_tensor_pack(test_df, node_to_idx)
    train_pack = scale_tensor_pack_targets(train_pack_raw, y_scaler)
    val_pack = scale_tensor_pack_targets(val_pack_raw, y_scaler)

    y_true_t = test_df["target_time"].to_numpy()
    y_true_e = test_df["target_energy"].to_numpy()

    detailed_df = test_df[["session", "u", "v", "target_time", "target_energy"]].copy()

    
    # GRAPH FAMILY
    
    print("\nTraining graph family...")
    graph_summary_rows = []
    seedwise_rows = []
    pred_cols_map: Dict[str, Tuple[str, str]] = {}
    uncertainty_cols_map: Dict[str, Tuple[str, str]] = {}

    for model_name in GRAPH_MODELS:
        print(f"  {model_name}")
        ensemble_preds = []
        for seed in ENSEMBLE_SEEDS:
            print(f"    seed {seed}")
            model = train_graph_model(
                encoder_name=model_name,
                train_pack=train_pack,
                val_pack=val_pack,
                node_x=node_x,
                edge_index=edge_index,
                seed=seed,
            )
            pred = infer_graph_model(model, test_pack, node_x, edge_index, y_scaler=y_scaler)
            ensemble_preds.append(pred)

                # Seed-wise metric row for mean ± std analysis across random seeds

            seed_row = metrics_dict(y_true_t, pred[:, 0], y_true_e, pred[:, 1])

            seed_row["model"] = model_name

            seed_row["seed"] = seed

            seed_row["model_family"] = "graph"

            seedwise_rows.append(seed_row)

        ensemble_preds = np.stack(ensemble_preds, axis=0)
        pred_mean = ensemble_preds.mean(axis=0)
        pred_std = ensemble_preds.std(axis=0)

        row = metrics_dict(y_true_t, pred_mean[:, 0], y_true_e, pred_mean[:, 1])
        row = add_ci_to_row(row, y_true_t, pred_mean[:, 0], y_true_e, pred_mean[:, 1])
        row["model"] = model_name
        row["uncertainty_method"] = "multi_seed_ensemble"
        graph_summary_rows.append(row)

        detailed_df[f"{model_name}_time_pred"] = pred_mean[:, 0]
        detailed_df[f"{model_name}_energy_pred"] = pred_mean[:, 1]
        detailed_df[f"{model_name}_time_unc"] = pred_std[:, 0]
        detailed_df[f"{model_name}_energy_unc"] = pred_std[:, 1]
        detailed_df[f"{model_name}_time_abs_err"] = np.abs(y_true_t - pred_mean[:, 0])
        detailed_df[f"{model_name}_energy_abs_err"] = np.abs(y_true_e - pred_mean[:, 1])
        pred_cols_map[model_name] = (f"{model_name}_time_pred", f"{model_name}_energy_pred")
        uncertainty_cols_map[model_name] = (f"{model_name}_time_unc", f"{model_name}_energy_unc")

    graph_summary_df = pd.DataFrame(graph_summary_rows).sort_values(
        ["time_r2", "energy_r2"], ascending=False
    ).reset_index(drop=True)

    # SEED-WISE MEAN ± STD TABLE FOR GRAPH MODELS
    seedwise_df = pd.DataFrame(seedwise_rows)

    if len(seedwise_df):
     seedwise_summary_df = (
        seedwise_df
        .groupby("model")
        .agg({
            "time_r2": ["mean", "std"],
            "time_mae": ["mean", "std"],
            "time_rmse": ["mean", "std"],
            "time_mape": ["mean", "std"],
            "energy_r2": ["mean", "std"],
            "energy_mae": ["mean", "std"],
            "energy_rmse": ["mean", "std"],
            "energy_mape": ["mean", "std"],
        })
    )

     seedwise_summary_df.columns = [
        "_".join(col).strip() for col in seedwise_summary_df.columns.values
     ]
     seedwise_summary_df = seedwise_summary_df.reset_index()

    # Compact manuscript-ready mean ± std format
     seedwise_compact_df = pd.DataFrame()
     seedwise_compact_df["model"] = seedwise_summary_df["model"]
     seedwise_compact_df["time_r2_mean_std"] = seedwise_summary_df.apply(
        lambda r: f"{r['time_r2_mean']:.4f} ± {r['time_r2_std']:.4f}", axis=1
     )
     seedwise_compact_df["time_mae_mean_std"] = seedwise_summary_df.apply(
         lambda r: f"{r['time_mae_mean']:.3f} ± {r['time_mae_std']:.3f}", axis=1
     )
     seedwise_compact_df["energy_r2_mean_std"] = seedwise_summary_df.apply(
         lambda r: f"{r['energy_r2_mean']:.4f} ± {r['energy_r2_std']:.4f}", axis=1
     )
     seedwise_compact_df["energy_mae_mean_std"] = seedwise_summary_df.apply(
        lambda r: f"{r['energy_mae_mean']:.2f} ± {r['energy_mae_std']:.2f}", axis=1
     )
    else:
     seedwise_summary_df = pd.DataFrame()
     seedwise_compact_df = pd.DataFrame()
    
    # NON-GRAPH ENSEMBLES
    
    print("\nTraining non-graph ensembles with bootstrap...")
    non_graph_rows = []
    x_train = train_df[FEATURE_COLS].to_numpy()
    y_train = train_df[["target_time", "target_energy"]].to_numpy()
    x_val = val_df[FEATURE_COLS].to_numpy()
    y_val = val_df[["target_time", "target_energy"]].to_numpy()
    x_test = test_df[FEATURE_COLS].to_numpy()

    for model_name in NON_GRAPH_MODELS:
        print(f"  {model_name}")
        pred_mean, pred_std = non_graph_bootstrap_predict(
            model_name,
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            ENSEMBLE_SEEDS,
            y_scaler=y_scaler,
        )
        if pred_mean is None or pred_std is None:
            continue

        row = metrics_dict(y_true_t, pred_mean[:, 0], y_true_e, pred_mean[:, 1])
        row = add_ci_to_row(row, y_true_t, pred_mean[:, 0], y_true_e, pred_mean[:, 1])
        row["model"] = model_name
        row["uncertainty_method"] = "bootstrap_ensemble"
        non_graph_rows.append(row)

        detailed_df[f"{model_name}_time_pred"] = pred_mean[:, 0]
        detailed_df[f"{model_name}_energy_pred"] = pred_mean[:, 1]
        detailed_df[f"{model_name}_time_unc"] = pred_std[:, 0]
        detailed_df[f"{model_name}_energy_unc"] = pred_std[:, 1]
        detailed_df[f"{model_name}_time_abs_err"] = np.abs(y_true_t - pred_mean[:, 0])
        detailed_df[f"{model_name}_energy_abs_err"] = np.abs(y_true_e - pred_mean[:, 1])
        pred_cols_map[model_name] = (f"{model_name}_time_pred", f"{model_name}_energy_pred")
        uncertainty_cols_map[model_name] = (f"{model_name}_time_unc", f"{model_name}_energy_unc")

    non_graph_df = pd.DataFrame(non_graph_rows).sort_values(
        ["time_r2", "energy_r2"], ascending=False
    ).reset_index(drop=True)

    
    # SEEN / UNSEEN AND SUPPORT BUCKETS
    
    seen_mask = make_seen_unseen_mask(train_raw, test_raw)
    train_support = build_train_support_map(train_raw)
    detailed_df["seen_in_train"] = seen_mask

    support_counts = []
    support_buckets = []
    for s, d in zip(test_raw["u"].astype(int), test_raw["v"].astype(int)):
        c = train_support.get((s, d), 0)
        support_counts.append(c)
        support_buckets.append(assign_support_bucket(c))
    detailed_df["train_support_count"] = support_counts
    detailed_df["support_bucket"] = support_buckets

    seen_unseen_df = build_seen_unseen_df(detailed_df, pred_cols_map)
    support_bucket_df = build_support_bucket_df(detailed_df, pred_cols_map)
    behavior_error_df = build_behavior_error_df(
        detailed_df=detailed_df,
        pred_cols_map=pred_cols_map,
        feature_source_df=test_raw,
    )

    difficult_cases_df = build_difficult_cases_df(
        detailed_df=detailed_df,
        model_name="GraphSAGE",
        top_k=10,
    )

    
    # UNCERTAINTY FOR ALL MODELS
    
    uncertainty_corr_df = build_uncertainty_corr_df(detailed_df, uncertainty_cols_map)
    selective_prediction_df = build_selective_prediction_df(detailed_df, uncertainty_cols_map)

# WILCOXON SIGNIFICANCE TESTS ON ABSOLUTE ERRORS
    wilcoxon_rows = []

    if SCIPY_AVAILABLE:
      model_pairs = [
        ("GraphSAGE", "GGNN"),
        ("GraphSAGE", "MLP"),
        ("GGNN", "MLP"),
        ("GraphSAGE", "SVR-RBF"),
        ("GraphSAGE", "GCN"),
        ("GraphSAGE", "GAT"),
    ]

    for m1, m2 in model_pairs:
        if (
            f"{m1}_time_abs_err" in detailed_df.columns
            and f"{m2}_time_abs_err" in detailed_df.columns
            and f"{m1}_energy_abs_err" in detailed_df.columns
            and f"{m2}_energy_abs_err" in detailed_df.columns
        ):
            for target in ["time", "energy"]:
                e1 = detailed_df[f"{m1}_{target}_abs_err"].to_numpy()
                e2 = detailed_df[f"{m2}_{target}_abs_err"].to_numpy()

                try:
                    stat, p_value = wilcoxon(e1, e2, zero_method="wilcox", alternative="two-sided")
                except Exception:
                    stat, p_value = np.nan, np.nan

                wilcoxon_rows.append({
                    "comparison": f"{m1} vs {m2}",
                    "target": target,
                    f"{m1}_mean_abs_error": float(np.mean(e1)),
                    f"{m2}_mean_abs_error": float(np.mean(e2)),
                    "mean_error_difference_m1_minus_m2": float(np.mean(e1 - e2)),
                    "wilcoxon_statistic": stat,
                    "p_value": p_value,
                    "significant_at_0.05": bool(p_value < 0.05) if not np.isnan(p_value) else False,
                })

    wilcoxon_df = pd.DataFrame(wilcoxon_rows)
    
    # COMBINED MAIN TABLE
    
    main_cross_session_df = pd.concat([graph_summary_df, non_graph_df], ignore_index=True)
    main_cross_session_df = main_cross_session_df.sort_values(
        ["time_r2", "energy_r2"], ascending=False
    ).reset_index(drop=True)

    
    # EXTRA TABLES
    
    asoc_eswa_df = build_asoc_eswa_difference_table(OUT_DIR)
    model_impl_df = build_model_implementation_table(OUT_DIR)
    ablation_df = build_feature_graph_ablation_table(
        OUT_DIR,
        train_raw,
        val_raw,
        test_raw,
        node_to_idx,
        edge_index,
        y_scaler=y_scaler,
    )

    
    # SAVE
    
    main_cross_session_df.to_csv(os.path.join(OUT_DIR, "table_main_cross_session.csv"), index=False)
    graph_summary_df.to_csv(os.path.join(OUT_DIR, "table_graph_family_cross_session.csv"), index=False)
    non_graph_df.to_csv(os.path.join(OUT_DIR, "table_non_graph_cross_session.csv"), index=False)
    seen_unseen_df.to_csv(os.path.join(OUT_DIR, "table_seen_unseen.csv"), index=False)
    support_bucket_df.to_csv(os.path.join(OUT_DIR, "table_support_buckets.csv"), index=False)
    behavior_error_df.to_csv(os.path.join(OUT_DIR, "table_behavior_error_turn_stop_slowdown.csv"), index=False)
    difficult_cases_df.to_csv(os.path.join(OUT_DIR, "table_difficult_prediction_cases_graphsage.csv"), index=False)    
    uncertainty_corr_df.to_csv(os.path.join(OUT_DIR, "table_uncertainty_corr_all_models.csv"), index=False)
    selective_prediction_df.to_csv(os.path.join(OUT_DIR, "table_selective_prediction_all_models.csv"), index=False)
    detailed_df.to_csv(os.path.join(OUT_DIR, "detailed_predictions.csv"), index=False)

    # Additional ASOC statistical-rigor outputs

    seedwise_df.to_csv(os.path.join(OUT_DIR, "table_seedwise_graph_model_metrics.csv"), index=False)

    seedwise_summary_df.to_csv(os.path.join(OUT_DIR, "table_seedwise_graph_model_mean_std.csv"), index=False)

    seedwise_compact_df.to_csv(os.path.join(OUT_DIR, "table_seedwise_graph_model_mean_std_compact.csv"), index=False)

    wilcoxon_df.to_csv(os.path.join(OUT_DIR, "table_wilcoxon_significance_tests.csv"), index=False)

    run_config = {
        "device": DEVICE,
        "lightgbm_available": LIGHTGBM_AVAILABLE,
        "data_csv": DATA_CSV,
        "graph_csv": GRAPH_CSV,
        "train_session": TRAIN_SESSION,
        "test_session": TEST_SESSION,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "feature_cols": FEATURE_COLS,
        "graph_models": GRAPH_MODELS,
        "non_graph_models": NON_GRAPH_MODELS,
        "ensemble_seeds": ENSEMBLE_SEEDS,
        "ablation_seeds": ABLATION_SEEDS,
        "ablation_graph_encoder": ABLATION_GRAPH_ENCODER,
        "non_graph_uncertainty_method": "bootstrap_ensemble",
        "graph_uncertainty_method": "multi_seed_ensemble",
        "confidence_interval": f"bootstrap_{CI_LEVEL}pct",
        "test_ci_bootstraps": TEST_CI_BOOTSTRAPS,
        "scaler_fit_on": "train_only",
        "target_scaling": "train_only_StandardScaler_for_all_models",
    }
    with open(os.path.join(OUT_DIR, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print("\nDone.")
    print(f"Saved files in: {OUT_DIR}")

    print("\nMain cross-session summary:")
    print(main_cross_session_df[["model", "time_r2", "time_mae", "energy_r2", "energy_mae"]].to_string(index=False))

    print("\nGraph-family summary:")
    print(graph_summary_df[["model", "time_r2", "time_mae", "energy_r2", "energy_mae"]].to_string(index=False))

    print("\nNon-graph summary:")
    if len(non_graph_df):
        print(non_graph_df[["model", "time_r2", "time_mae", "energy_r2", "energy_mae"]].to_string(index=False))
    else:
        print("No non-graph results were produced.")

    print("\nUncertainty correlation summary:")
    print(uncertainty_corr_df.to_string(index=False))

    print("\nAblation summary:")
    print(ablation_df[["variant", "base_model", "time_r2", "time_mae", "energy_r2", "energy_mae"]].to_string(index=False))

    print("\nGenerated paper-support tables:")
    print("  table_asoc_vs_eswa_difference.csv")
    print("  table_model_implementation.csv")
    print("  table_feature_graph_ablation.csv")
    print("  table_main_cross_session.csv with CI/std columns")


if __name__ == "__main__":
    main()
