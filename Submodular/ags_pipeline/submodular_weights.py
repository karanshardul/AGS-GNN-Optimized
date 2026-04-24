"""Facility-location diversity weights with lazy greedy (same logic as SubmodularWeights.ipynb)."""

from __future__ import annotations

import heapq
import math
import multiprocessing as mp
from typing import List, Tuple

import numpy as np
import torch
from torch_sparse import SparseTensor
from tqdm import tqdm

from .benchmark import timed_section
from .kernels import facility_location_kernel_matrix


def _lazy_greedy_facility(
    kernel_dist: np.ndarray,
    col: torch.Tensor,
    edge_index: torch.Tensor,
    v2i: dict,
    u: int,
    lambda1: float,
    lambda2: float,
    w1: float,
    w2: float,
    w3: float,
) -> Tuple[List[float], List[int]]:
    """Lazy greedy — line-equivalent to SubmodularWeights.ipynb `lazy_greedy_weight`."""
    gain_of_u = np.sum(kernel_dist[v2i[u], :])
    gain_list = [
        (
            -1
            * (
                np.sum(
                    np.max(kernel_dist[[v2i[u], v2i[v.item()]], :], axis=0)
                )
                - gain_of_u
            ),
            v.item(),
            e.item(),
        )
        for v, e in zip(col, edge_index)
    ]
    heapq.heapify(gain_list)

    S = [u]
    S_G: List[float] = []
    S_edge: List[int] = []
    S_index = [v2i[u]]

    l1 = math.ceil(len(col) * lambda1)
    l2 = min(len(col) - l1, math.ceil(len(col) * lambda2))
    l3 = max(0, int(len(col) - l1 - l2))

    rank = 1
    S_index_gain = gain_of_u

    while gain_list:
        gain_v, v, e = heapq.heappop(gain_list)
        gain_v = -1 * gain_v

        if len(gain_list) == 0:
            S.append(v)
            if gain_v < 1e-6:
                gain_v = 1e-6
            if rank <= l1:
                S_G.append(w1)
            elif rank <= l1 + l2:
                S_G.append(w2)
            else:
                S_G.append(w3)
            rank += 1
            S_edge.append(e)
            S_index.append(v2i[v])
            break

        if len(gain_list) < l3:
            S.append(v)
            S_G.append(w3)
            rank += 1
            S_edge.append(e)
            S_index.append(v2i[v])
            continue

        gain_v_update = np.sum(np.max(kernel_dist[np.append(S_index, v2i[v]), :], axis=0)) - S_index_gain

        gain_v_second, v_second, _ = gain_list[0]
        gain_v_second = -1 * gain_v_second

        if gain_v_update >= gain_v_second:
            if gain_v_update < 1e-6:
                gain_v_update = 1e-6
            gain_v_update = -1 * gain_v_update
            S.append(v)
            S_index_gain = np.sum(
                np.max(kernel_dist[np.append(S_index, v2i[v]), :], axis=0)
            )
            if rank <= l1:
                S_G.append(w1)
            elif rank <= l1 + l2:
                S_G.append(w2)
            else:
                S_G.append(w3)
            rank += 1
            S_edge.append(e)
            S_index.append(v2i[v])
        else:
            heapq.heappush(gain_list, (-1 * gain_v_update, v, e))

    return S_G, S_edge


def _submodular_node(
    u: int,
    adj: SparseTensor,
    x: torch.Tensor,
    metric: str,
    lambda1: float,
    lambda2: float,
    w1: float,
    w2: float,
    w3: float,
    device: torch.device,
) -> Tuple[List[float], List[int]]:
    _row, col, edge_index = adj[u, :].coo()
    if col.numel() == 0:
        return [], []

    vertices = [u] + col.tolist()
    v2i = {vid: j for j, vid in enumerate(vertices)}

    xv = x[vertices]
    kd_t = facility_location_kernel_matrix(xv, metric, dtype=torch.float64)
    kernel_dist = kd_t.detach().cpu().numpy()

    return _lazy_greedy_facility(
        kernel_dist,
        col,
        edge_index,
        v2i,
        u,
        lambda1,
        lambda2,
        w1,
        w2,
        w3,
    )


def _process_block_submodular(args):
    list_u, ei0, ei1, e_vals, n_nodes, n_edges, x, metric, l1, l2, w1, w2, w3, dev = args
    edge_index = torch.stack([ei0, ei1])
    adj = SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        value=e_vals,
        sparse_sizes=(n_nodes, n_nodes),
    )
    device = torch.device(dev)
    x = x.to(device)
    adj = adj.to(device)

    edge_weight: List[float] = []
    edge_out: List[int] = []
    for u in list_u:
        w, e = _submodular_node(
            u, adj, x, metric, l1, l2, w1, w2, w3, device
        )
        edge_weight.extend(w)
        edge_out.extend(e)
    return edge_weight, edge_out, len(list_u)


class SubModularWeightFacilityOptimized:
    """Drop-in replacement for `SubModularWeightFacilityFaster` with torch kernels on GPU."""

    def __init__(
        self,
        data,
        sub_func: str = "facility",
        metric: str = "cosine",
        log: bool = True,
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
        self.metric = metric
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.mp_threshold = mp_threshold

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

    def lazy_greedy_weight(self, u: int):
        return _submodular_node(
            u,
            self.adj,
            self.x,
            self.metric,
            self.lambda1,
            self.lambda2,
            self.w1,
            self.w2,
            self.w3,
            self.device,
        )

    def get_submodular_weight_serial(self) -> torch.Tensor:
        edge_weight: List[float] = []
        edge_index: List[int] = []
        it = range(self.N)
        if self.log:
            it = tqdm(it, total=self.N, desc="Submodular nodes")
        for u in it:
            w, e = self.lazy_greedy_weight(u)
            edge_weight.extend(w)
            edge_index.extend(e)

        assert len(edge_index) == self.E
        weight = torch.zeros(len(edge_index), device=self.data.edge_index.device)
        weight[torch.tensor(edge_index, device=weight.device)] = torch.tensor(
            edge_weight, dtype=torch.float32, device=weight.device
        )
        return weight

    def get_submodular_weight_multiprocess(self) -> torch.Tensor:
        num_blocks = min(mp.cpu_count() or 4, max(1, self.N // 256))
        elem_size = self.N // num_blocks
        nodes: List[List[int]] = [
            list(range(i * elem_size, (i + 1) * elem_size)) for i in range(num_blocks)
        ]
        if num_blocks * elem_size < self.N:
            nodes.append(list(range(num_blocks * elem_size, self.N)))

        ei = self.data.edge_index
        e_vals = torch.arange(self.E)
        dev = "cpu"  # avoid CUDA in worker processes
        x_cpu = self.data.x.cpu()
        tasks = [
            (
                block,
                ei[0].cpu(),
                ei[1].cpu(),
                e_vals,
                self.N,
                self.E,
                x_cpu,
                self.metric,
                self.lambda1,
                self.lambda2,
                self.w1,
                self.w2,
                self.w3,
                dev,
            )
            for block in nodes
        ]

        edge_weight: List[float] = []
        edge_index: List[int] = []
        if self.log:
            print("Submodular multiprocessing blocks:", len(tasks))
        procs = min(len(tasks), mp.cpu_count() or 4)
        with timed_section("submodular_weight_multiprocess"):
            with mp.get_context("spawn").Pool(procs) as pool:
                it = pool.imap_unordered(_process_block_submodular, tasks)
                if self.log:
                    it = tqdm(it, total=len(tasks), desc="Submodular blocks")
                for w, e, _ in it:
                    edge_weight.extend(w)
                    edge_index.extend(e)

        assert len(edge_index) == self.E
        weight = torch.zeros(len(edge_index), device=self.data.edge_index.device)
        weight[torch.tensor(edge_index)] = torch.tensor(edge_weight, dtype=torch.float32)
        return weight

    def compute_weights(self) -> torch.Tensor:
        with timed_section("submodular_compute_weights"):
            if self.N < self.mp_threshold:
                return self.get_submodular_weight_serial()
            return self.get_submodular_weight_multiprocess()


SubModularWeightFacilityFaster = SubModularWeightFacilityOptimized
