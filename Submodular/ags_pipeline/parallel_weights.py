"""Run similarity (KNN) and diversity (submodular) weight precomputation in parallel."""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Dict, Tuple

import torch

from .benchmark import timed_section


def _knn_compute_task(payload: Tuple[Any, ...]) -> torch.Tensor:
    data, kwargs = payload
    from .knn_weights import KNNWeightOptimized

    knn = KNNWeightOptimized(data, **kwargs)
    return knn.compute_weights()


def _submod_compute_task(payload: Tuple[Any, ...]) -> torch.Tensor:
    data, kwargs = payload
    from .submodular_weights import SubModularWeightFacilityOptimized

    sub = SubModularWeightFacilityOptimized(data, **kwargs)
    return sub.compute_weights()


def compute_knn_and_submodular_parallel(
    data,
    knn_kwargs: Dict | None = None,
    submod_kwargs: Dict | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dual-channel precomputation: KNN + facility submodular weights simultaneously.

    Uses two processes so both pipelines run concurrently. Nested process pools are
    avoided by forcing the serial (GPU) code path inside each worker via a large
    ``mp_threshold`` unless you override it explicitly.
    """
    knn_kwargs = dict(knn_kwargs or {})
    submod_kwargs = dict(submod_kwargs or {})
    # Avoid Pool-inside-Pool when each channel would otherwise spawn workers.
    knn_kwargs.setdefault("mp_threshold", 10**12)
    submod_kwargs.setdefault("mp_threshold", 10**12)

    with timed_section("parallel_knn_submod_precompute"):
        ctx = mp.get_context("spawn")
        with ctx.Pool(2) as pool:
            r1 = pool.apply_async(_knn_compute_task, ((data, knn_kwargs),))
            r2 = pool.apply_async(_submod_compute_task, ((data, submod_kwargs),))
            w_knn = r1.get()
            w_sub = r2.get()
    return w_knn, w_sub
