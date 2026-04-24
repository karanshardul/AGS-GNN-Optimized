"""
Tensor kernels matching the original sklearn-based `calculate_pairwise_distances`
(SubmodularWeights.ipynb) and similarity used in KNNWeights.

All ops prefer GPU when `device` is CUDA; use float64 for parity-sensitive paths.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_similarity_query_neighbors(
    x_query: torch.Tensor,
    x_neighbors: torch.Tensor,
    metric: str,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Similarity / distance scores between one query row and many neighbor rows.

    For metric='cosine', returns cosine similarity (higher is more similar), same
    convention as sklearn.metrics.pairwise.cosine_similarity for (1, f) vs (d, f).

    For metric='euclidean', returns Euclidean distance (lower is closer), same as
    sklearn.metrics.pairwise.euclidean_distances.
    """
    if dtype is None:
        dtype = x_query.dtype
    x_query = x_query.to(dtype=dtype).flatten(0, -1)
    x_neighbors = x_neighbors.to(dtype=dtype)
    if metric == "cosine":
        q = F.normalize(x_query, dim=-1, eps=1e-12)
        n = F.normalize(x_neighbors, dim=-1, eps=1e-12)
        return (n * q).sum(dim=-1)
    if metric == "euclidean":
        return torch.cdist(x_query.unsqueeze(0), x_neighbors, p=2).squeeze(0)
    raise ValueError(f"Unknown metric: {metric}")


def facility_location_kernel_matrix(
    X: torch.Tensor,
    metric: str,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Pairwise kernel matrix used by facility-location submodular selection.

    Matches `calculate_pairwise_distances` in SubmodularWeights.ipynb:
    - cosine: transform sklearn cosine distance through 1 - (1 - d)^2
    - euclidean: squared distances, then subtract from global max (higher = better)
    """
    if dtype is None:
        dtype = torch.float64 if X.device.type == "cpu" else torch.float32
    X = X.to(dtype=dtype)
    if metric == "cosine":
        # sklearn cosine distance = 1 - cos_sim; then kernel = 1 - (1 - dist)^2 = 1 - cos_sim^2
        x_norm = F.normalize(X, dim=-1, eps=1e-12)
        cos_sim = x_norm @ x_norm.t()
        return 1.0 - cos_sim * cos_sim
    if metric == "euclidean":
        d2 = torch.cdist(X, X, p=2).pow(2)
        return d2.max() - d2
    raise ValueError(f"Unknown metric for facility kernel: {metric}")


def rank_neighbor_order(
    scores: torch.Tensor,
    sign: float,
) -> torch.Tensor:
    """
    Permutation indices that sort neighbors in the same sense as the original
    `np.argsort(sign * scores)`.

    `sign=-1` for cosine (descending similarity), `sign=1` for Euclidean (ascending distance).
    """
    s = sign * scores
    # stable sort when available (PyTorch 2.x); 1.9 falls back to default order
    try:
        return torch.argsort(s, dim=-1, stable=True)
    except TypeError:
        return torch.argsort(s, dim=-1)
