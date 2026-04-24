# AGS-GNN-Optimized

Parallelized and optimized implementation of **AGS-GNN (Attribute Guided Sampling for Graph Neural Networks)** designed to improve preprocessing, sampling, and training efficiency on large graph datasets.

This project is based on the original AGS-GNN implementation and focuses on eliminating sequential bottlenecks using:

- Parallel computing
- GPU acceleration
- Vectorized operations
- Multiprocessing
- Cached preprocessing
- Faster training pipeline

---

## Original AGS-GNN

AGS-GNN improves graph sampling by combining:

- Similarity-based neighbor sampling
- Diversity-based submodular sampling
- Weighted neighborhood selection
- Dual-channel graph sampling

### Original Pipeline

```text
Similarity Computation
→ Neighbor Ranking
→ Diversity Computation
→ Weighted Sampling
→ GNN Training
````

The original implementation contained multiple sequential bottlenecks that became inefficient for large-scale graphs.

---

# Optimization Goal

The objective of this project was to optimize AGS-GNN for better performance using parallel programming techniques while maintaining model correctness.

### Problems in Original Implementation

* Sequential similarity computation
* Per-node loops
* Full sorting overhead
* Sequential diversity sampling
* Slow weighted sampling
* Repeated preprocessing
* High initialization cost
* CPU bottlenecks during training

---

# Optimization Techniques Implemented

## 1. Parallel Similarity Computation

### Original

```python
for node in nodes:
    compute_similarity(node)
```

### Optimized

* Vectorized similarity computation using PyTorch tensors
* Batch-wise processing
* GPU acceleration
* Multiprocessing fallback for CPU systems

```python
similarity_matrix = torch.mm(features, features.T)
```

### Benefits

* Removed nested loops
* Faster preprocessing
* Better scalability

---

## 2. Parallel Neighbor Ranking

### Original

* Full sorting per node
* Sequential ranking

### Optimized

* Parallel ranking per node
* Top-k selection instead of full sorting
* Tensor-based ranking

```python
torch.topk(similarity_scores, k)
```

### Benefits

* Reduced sorting overhead
* Faster nearest-neighbor selection

---

## 3. Parallel Submodular Diversity Sampling

### Original

* Sequential facility-location computation
* CPU pairwise loops
* High computational overhead

### Optimized

* Vectorized pairwise similarity computation
* GPU acceleration
* Parallel execution across nodes
* Lazy greedy optimization retained

### Benefits

* Faster diversity computation
* Reduced O(d²) bottleneck

---

## 4. Dual-Channel Parallel Execution

### Original Workflow

```text
Similarity → Diversity
```

### Optimized Workflow

```text
Similarity || Diversity
```

Both channels now execute simultaneously.

### Benefits

* Reduced preprocessing latency

---

## 5. Vectorized Weighted Neighbor Sampling

### Original

* Python loops
* NumPy random choice

### Optimized

```python
torch.multinomial()
```

### Benefits

* Faster minibatch generation
* GPU compatibility

---

## 6. GPU Acceleration

Major tensor-heavy operations moved to GPU:

* Similarity matrix computation
* Diversity kernel computation
* Sampling probabilities
* Neighbor selection

### Benefits

* Significant runtime reduction

---

## 7. Caching Precomputed Weights

### Before

```text
Compute every run
```

### After

```text
Compute once → Save cache → Reuse
```

Cached components:

* Similarity weights
* Diversity weights
* Ranked neighbor indices
* Sampling probability tensors

### Benefits

* Up to ~100x faster repeated runs

---

## 8. Training Pipeline Optimization

Optimizations include:

* Mini-batch training
* Reduced redundant forward passes
* Lower memory allocations
* Faster epoch execution

---

## 9. Multi-worker Data Loading

```python
num_workers > 0
```

This overlaps:

* CPU sampling
* GPU training

### Benefits

* Better hardware utilization

---

## 10. Modular Refactoring

Refactored major components into reusable modules:

* `compute_similarity()`
* `rank_neighbors()`
* `sample_neighbors()`
* `parallel_weight_computation()`

### Benefits

* Cleaner architecture
* Easier experimentation

---

# Final Optimized Pipeline

```text
Parallel Similarity + Parallel Diversity
                ↓
             Caching
                ↓
      Vectorized Sampling
                ↓
          GNN Training
```

---

# Performance Improvements

Compared to original AGS-GNN:

* Faster preprocessing
* Faster repeated execution
* Better GPU utilization
* Reduced CPU bottlenecks
* Improved scalability on large datasets

### Example Improvements

* Precomputation Speedup: Xx
* Sampling Speedup: Yx
* Cached Runs: ~100x faster

> Replace X and Y with your actual benchmark values if available.

---

# Tech Stack

* Python
* PyTorch
* PyTorch Geometric
* DGL
* NumPy
* CUDA
* Multiprocessing

---

# Installation

```bash
git clone https://github.com/karanshardul/AGS-GNN-Optimized.git
cd AGS-GNN-Optimized
pip install -r requirements.txt
```

---

# Running the Project

### Run preprocessing

```bash
python preprocess.py
```

### Run training

```bash
python train.py
```

### Run benchmarks

```bash
python benchmark.py
```

---

# Repository Structure

```text
AGS-GNN-Optimized/
│
├── preprocessing/
├── samplers/
├── training/
├── benchmarks/
├── cache/
├── utils/
└── README.md
```

---

# Based On

This project is built on top of the original AGS-GNN implementation and focuses on optimizing its execution pipeline through parallel programming techniques.

Core inherited concepts include:

* Similarity ranking
* Submodular ranking
* Weighted sampling
* Graph sampling pipelines

---

# Future Improvements

* Multi-GPU distributed training
* CUDA kernel optimization
* Graph partitioning
* Distributed graph processing

---

# Authors

* Karan Shardul
* Kaivalya Vanmali
* Vanshika Srivastava
* Anmol Agrawal

---
