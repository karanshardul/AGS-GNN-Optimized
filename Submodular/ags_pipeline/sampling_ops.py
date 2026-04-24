"""Vectorized helpers for weighted neighbor sampling (Python fallback path)."""

from __future__ import annotations

from typing import Tuple

import torch


def sample_neighbors(
    row: torch.Tensor,
    edge_weight: torch.Tensor,
    k: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Sample k neighbors without replacement using normalized weights — same distribution
    as `numpy.random.choice(..., replace=False, p=...)`, via `torch.multinomial`.
    """
    if k <= 0 or row.numel() == 0:
        return row.new_empty(0, dtype=torch.long)
    if k >= row.numel():
        return row.clone()

    w = edge_weight.float().clamp(min=0.0)
    s = w.sum()
    if s <= 0:
        p = torch.ones_like(w) / w.numel()
    else:
        p = w / s
    idx = torch.multinomial(
        p, num_samples=k, replacement=False, generator=generator
    )
    return row[idx]


def softmax_weights(edge_weight: torch.Tensor) -> torch.Tensor:
    """Normalize nonnegative weights to a probability vector (stable)."""
    w = edge_weight.float().clamp(min=0.0)
    return w / (w.sum() + 1e-12)
