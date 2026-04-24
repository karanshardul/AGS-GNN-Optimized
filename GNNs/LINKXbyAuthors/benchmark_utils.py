"""
Benchmarking helpers: timing (time.time), metrics (accuracy, macro F1), JSON I/O.
Does not modify model forward/backward logic — only measurement utilities.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_test_accuracy_and_f1_macro(
    dataset,
    split_idx,
    out: torch.Tensor,
    use_rocauc: bool,
) -> Tuple[float, Optional[float]]:
    """
    Test-set accuracy (or test ROC-AUC when use_rocauc) and macro F1 on test nodes.

    For multi-class node classification, F1 uses average='macro'.
    For ROC-AUC setups, F1 is omitted (None) when not well-defined for the task.
    """
    y_true = dataset.label[split_idx["test"]].detach().cpu()
    y_logits = out[split_idx["test"]].detach().cpu()

    if use_rocauc:
        from data_utils import eval_rocauc

        acc_or_auc = float(eval_rocauc(y_true, y_logits))
        return acc_or_auc, None

    # Single-task class indices
    if y_true.dim() > 1 and y_true.size(1) > 1:
        # Multi-label: macro F1 over labels
        y_pred = (torch.sigmoid(y_logits) > 0.5).long().numpy()
        yt = y_true.numpy()
        f1 = float(f1_score(yt, y_pred, average="macro", zero_division=0))
        # "accuracy" as mean label-wise accuracy
        correct = (yt == y_pred).astype(float)
        acc = float(correct.mean())
        return acc, f1

    y_true_1d = y_true.squeeze(1).numpy()
    y_pred_1d = y_logits.argmax(dim=-1).cpu().numpy()
    acc = float((y_true_1d == y_pred_1d).mean())
    f1 = float(
        f1_score(y_true_1d, y_pred_1d, average="macro", zero_division=0)
    )
    return acc, f1


def save_benchmark_json(
    path: str,
    payload: Dict[str, Any],
) -> None:
    abspath = os.path.abspath(path)
    parent = os.path.dirname(abspath)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_benchmark_json(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def submodular_root() -> str:
    """Path to AGS-GNN-main/Submodular from this file (LINKXbyAuthors)."""
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "Submodular")
    )


def measure_ags_precompute_knn_submodular(dataset, device: torch.device) -> float:
    """
    Wall time for sequential KNN + submodular edge-weight precomputation (seconds).
    Requires ``Submodular/ags_pipeline`` and torch_geometric Data.
    """
    import sys

    root = submodular_root()
    if root not in sys.path:
        sys.path.insert(0, root)

    from torch_geometric.data import Data

    from ags_pipeline import (
        KNNWeightOptimized,
        SubModularWeightFacilityOptimized,
    )

    g = dataset.graph
    data = Data(
        x=g["node_feat"].to(device),
        edge_index=g["edge_index"].to(device),
        num_nodes=g["num_nodes"],
    )

    t0 = time.time()
    KNNWeightOptimized(data, metric="cosine", log=False, device=device).compute_weights()
    SubModularWeightFacilityOptimized(data, metric="cosine", log=False, device=device).compute_weights()
    return time.time() - t0


def build_hyperparameter_dict(args) -> Dict[str, Any]:
    """Snapshot of argparse namespace for JSON (JSON-serializable values only)."""
    out: Dict[str, Any] = {}
    for k, v in vars(args).items():
        if k.startswith("_"):
            continue
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = str(v)
    return out
