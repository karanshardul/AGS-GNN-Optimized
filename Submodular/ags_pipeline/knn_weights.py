"""KNN-based edge weights (Attribute-Guided similarity channel), optimized with PyTorch."""

from __future__ import annotations

import math
import multiprocessing as mp
from typing import List, Tuple

import torch
from torch_sparse import SparseTensor
from tqdm import tqdm

from .benchmark import timed_section
from .kernels import compute_similarity_query_neighbors, rank_neighbor_order


def _node_weight_knn(
    u: int,
    adj_row: torch.Tensor,
    adj_col: torch.Tensor,
    adj_edge_index: torch.Tensor,
    x: torch.Tensor,
    metric: str,
    sign: float,
    lambda1: float,
    lambda2: float,
    w1: float,
    w2: float,
    w3: float,
) -> Tuple[List[float], List[int]]:
    if adj_col.numel() == 0:
        return [], []

    l1 = math.ceil(len(adj_col) * lambda1)
    l2 = min(len(adj_col) - l1, math.ceil(len(adj_col) * lambda2))
    l3 = max(0, int(len(adj_col) - l1 - l2))

    target = compute_similarity_query_neighbors(x[u], x[adj_col], metric)
    ind = rank_neighbor_order(target, sign)

    S_G = [w1] * l1 + [w2] * l2 + [w3] * l3
    S_edge = adj_edge_index[ind].tolist()
    return S_G, S_edge


def _process_block_knn(
    args: Tuple[
        List[int],
        torch.Tensor,
        SparseTensor,
        str,
        float,
        float,
        float,
        float,
        float,
        float,
    ],
) -> Tuple[List[float], List[int], int]:
    (
        list_u,
        x,
        adj,
        metric,
        sign,
        lambda1,
        lambda2,
        w1,
        w2,
        w3,
    ) = args
    edge_weight: List[float] = []
    edge_index: List[int] = []
    for u in list_u:
        row, col, edge_index_t = adj[u, :].coo()
        w, e = _node_weight_knn(
            u,
            row,
            col,
            edge_index_t,
            x,
            metric,
            sign,
            lambda1,
            lambda2,
            w1,
            w2,
            w3,
        )
        edge_weight.extend(w)
        edge_index.extend(e)
    return edge_weight, edge_index, len(list_u)


class KNNWeightOptimized:
    """
    Drop-in replacement for `KNNWeight` from KNNWeights.ipynb.

    Uses vectorized similarity per node on the given `device` and optional
    multiprocessing over node blocks for large graphs (same strategy as the original).
    """

    def __init__(
        self,
        data,
        metric: str = "cosine",
        log: bool = False,
        lambda1: float = 0.25,
        lambda2: float = 0.25,
        w1: float = 1.0,
        w2: float = 0.5,
        w3: float = 0.1,
        device: torch.device | None = None,
        mp_threshold: int = 10000,
    ):
        self.N = data.num_nodes
        self.E = data.num_edges
        self.data = data
        self.log = log
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.metric = metric
        self.mp_threshold = mp_threshold

        if metric == "cosine":
            self.sign = -1.0
        elif metric == "euclidean":
            self.sign = 1.0
        else:
            raise ValueError("metric must be 'cosine' or 'euclidean'")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.x = data.x.to(device)
        self.adj = SparseTensor(
            row=data.edge_index[0],
            col=data.edge_index[1],
            value=torch.arange(self.E, device=data.edge_index.device),
            sparse_sizes=(self.N, self.N),
        ).to(device)

    def node_weight(self, u: int):
        row, col, edge_index = self.adj[u, :].coo()
        return _node_weight_knn(
            u,
            row,
            col,
            edge_index,
            self.x,
            self.metric,
            self.sign,
            self.lambda1,
            self.lambda2,
            self.w1,
            self.w2,
            self.w3,
        )

    def get_knn_weight_serial(self) -> torch.Tensor:
        edge_weight: List[float] = []
        edge_index: List[int] = []
        it = range(self.N)
        if self.log:
            it = tqdm(it, total=self.N, desc="KNN nodes")
        for u in it:
            w, e = self.node_weight(u)
            edge_weight.extend(w)
            edge_index.extend(e)

        assert len(edge_index) == self.E
        weight = torch.zeros(len(edge_index), device=self.data.edge_index.device)
        weight[torch.tensor(edge_index, device=weight.device)] = torch.tensor(
            edge_weight, dtype=torch.float32, device=weight.device
        )
        return weight

    def get_knn_weight_multiprocess(self) -> torch.Tensor:
        num_blocks = min(mp.cpu_count() or 4, max(1, self.N // 512))
        elem_size = self.N // num_blocks
        nodes: List[List[int]] = [
            list(range(i * elem_size, (i + 1) * elem_size)) for i in range(num_blocks)
        ]
        if num_blocks * elem_size < self.N:
            nodes.append(list(range(num_blocks * elem_size, self.N)))

        # Workers use CPU tensors to avoid CUDA multiprocessing issues
        x_cpu = self.data.x.cpu()
        adj_cpu = SparseTensor(
            row=self.data.edge_index[0],
            col=self.data.edge_index[1],
            value=torch.arange(self.E, device=self.data.edge_index.device),
            sparse_sizes=(self.N, self.N),
        )

        tasks = [
            (
                block,
                x_cpu,
                adj_cpu,
                self.metric,
                self.sign,
                self.lambda1,
                self.lambda2,
                self.w1,
                self.w2,
                self.w3,
            )
            for block in nodes
        ]
        procs = min(8, mp.cpu_count() or 4, len(tasks))

        edge_weight: List[float] = []
        edge_index: List[int] = []
        if self.log:
            print("KNN multiprocessing: %d blocks, %d workers" % (len(tasks), procs))
        with timed_section("knn_weight_multiprocess"):
            with mp.get_context("spawn").Pool(procs) as pool:
                it = pool.imap_unordered(_process_block_knn, tasks)
                if self.log:
                    it = tqdm(it, total=len(tasks), desc="KNN blocks")
                for w, e, _n in it:
                    edge_weight.extend(w)
                    edge_index.extend(e)

        assert len(edge_index) == self.E
        weight = torch.zeros(len(edge_index), device=self.data.edge_index.device)
        weight[torch.tensor(edge_index)] = torch.tensor(edge_weight, dtype=torch.float32)
        return weight

    def compute_weights(self) -> torch.Tensor:
        with timed_section("knn_compute_weights"):
            if self.N < self.mp_threshold:
                return self.get_knn_weight_serial()
            return self.get_knn_weight_multiprocess()


# Backwards-compatible alias
KNNWeight = KNNWeightOptimized
