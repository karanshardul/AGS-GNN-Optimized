"""
Optimized AGS-GNN precomputation (similarity + diversity weights) and sampling helpers.

Public helpers (requested API):
- ``compute_similarity`` — batched similarity / distance query vs neighbors
- ``rank_neighbors`` — permutation from scores (same convention as ``np.argsort(sign * scores)``)
- ``sample_neighbors`` — weighted sampling without replacement (``torch.multinomial``)

Weight classes (drop-in for notebook implementations):
- ``KNNWeightOptimized`` / ``KNNWeight``
- ``SubModularWeightFacilityOptimized`` / ``SubModularWeightFacilityFaster``

Other:
- ``compute_knn_and_submodular_parallel`` — dual-channel precompute
- ``timed_section`` / ``log_timing`` — simple benchmarks
"""

from .benchmark import log_timing, timed_section
from .kernels import (
    compute_similarity_query_neighbors as compute_similarity,
    facility_location_kernel_matrix,
    rank_neighbor_order as rank_neighbors,
)
from .knn_weights import KNNWeight, KNNWeightOptimized
from .parallel_weights import compute_knn_and_submodular_parallel
from .sampling_ops import sample_neighbors
from .submodular_weights import (
    SubModularWeightFacilityFaster,
    SubModularWeightFacilityOptimized,
)

__all__ = [
    "compute_similarity",
    "rank_neighbors",
    "sample_neighbors",
    "facility_location_kernel_matrix",
    "KNNWeight",
    "KNNWeightOptimized",
    "SubModularWeightFacilityFaster",
    "SubModularWeightFacilityOptimized",
    "compute_knn_and_submodular_parallel",
    "timed_section",
    "log_timing",
]
