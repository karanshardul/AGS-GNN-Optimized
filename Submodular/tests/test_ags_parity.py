"""
Numerical parity checks for ags_pipeline vs sklearn reference formulas.

Run (with PyTorch + torch_sparse + sklearn installed):
  cd Submodular && python -m pytest tests/test_ags_parity.py -q
or:
  python tests/test_ags_parity.py
"""

from __future__ import annotations

import sys


def test_kernel_cosine_facility():
    import numpy as np
    import torch
    from sklearn.metrics.pairwise import pairwise_distances

    from ags_pipeline.kernels import facility_location_kernel_matrix

    torch.manual_seed(0)
    X = torch.randn(6, 5, dtype=torch.float64)
    sk = pairwise_distances(X.numpy(), metric="cosine")
    ref = 1.0 - (1.0 - sk) ** 2
    got = facility_location_kernel_matrix(X, "cosine", dtype=torch.float64).numpy()
    assert np.allclose(ref, got, rtol=1e-6, atol=1e-7)


def test_knn_cosine_scores():
    import numpy as np
    import torch
    from sklearn.metrics.pairwise import cosine_similarity

    from ags_pipeline.kernels import compute_similarity_query_neighbors

    x = torch.randn(10, 4, dtype=torch.float64)
    u = 2
    cols = torch.tensor([1, 3, 5, 7])
    sk = cosine_similarity(x[u].numpy().reshape(1, -1), x[cols].numpy())[0]
    got = compute_similarity_query_neighbors(x[u], x[cols], "cosine", dtype=torch.float64).numpy()
    assert np.allclose(sk, got, rtol=1e-6, atol=1e-7)


def main():
    try:
        import torch  # noqa: F401
    except ImportError:
        print("PyTorch not installed; skip parity tests.", file=sys.stderr)
        return 1
    test_kernel_cosine_facility()
    test_knn_cosine_scores()
    print("ags_pipeline parity tests OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
