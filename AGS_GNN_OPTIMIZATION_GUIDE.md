# AGS-GNN Optimization and Parallelization Report

## 1. Introduction

Graph Neural Networks (GNNs) often face scalability challenges when applied to large datasets due to expensive preprocessing operations, neighbor selection overhead, and high training costs. The AGS-GNN framework uses both similarity-based and diversity-based neighbor selection strategies, which improve learning quality but introduce computational bottlenecks.

This project focuses on optimizing the AGS-GNN pipeline by improving execution efficiency in preprocessing, sampling, and training while maintaining model accuracy. The primary objective was to reduce computation time and improve scalability through parallelization, GPU acceleration, memory optimization, and efficient sampling techniques.

---

## 2. Existing System Architecture

The original AGS-GNN workflow consists of the following stages:

**Input Graph Data**

* Node features
* Edge connections
* Labels

**Processing Pipeline**

1. Similarity-based weight computation
2. Submodular diversity weight computation
3. Combination of edge weights
4. Neighbor sampling
5. Model training
6. Evaluation

### Workflow

```text
Input Data
   → Similarity Weight Computation
   → Submodular Weight Computation
   → Combined Edge Weights
   → Neighbor Sampling
   → AGSGNN Training
   → Performance Evaluation
```

---

## 3. Problem Identification

During analysis of the original implementation, several performance bottlenecks were identified:

### 3.1 Sequential Preprocessing

Similarity computation and submodular weight computation were executed sequentially, increasing preprocessing time.

### 3.2 Expensive Neighbor Ranking

The system performed full sorting of neighbor nodes even when only top-ranked neighbors were required.

**Complexity:**
O(d log d)

where *d* represents neighborhood size.

### 3.3 High Cost Submodular Optimization

The diversity selection module used computationally expensive greedy operations.

### 3.4 Repeated Weight Computation

Edge weights were recomputed every time the model was trained.

### 3.5 Inefficient Data Loading

Training performance suffered due to slow CPU-GPU data transfer.

### 3.6 Lack of Performance Monitoring

There was no structured benchmarking mechanism to evaluate runtime improvements.

---

# 4. Optimization Techniques Implemented

## 4.1 Parallel Precomputation

To reduce preprocessing time, similarity weight computation and submodular weight computation were executed in parallel.

```python
with mp.get_context("spawn").Pool(2) as pool:
    r1 = pool.apply_async(knn_task)
    r2 = pool.apply_async(submod_task)
```

### Benefits

* Reduces preprocessing latency
* Improves CPU utilization
* Enables simultaneous execution of independent tasks

---

## 4.2 Optimized Neighbor Ranking

Instead of performing full sorting, top-k neighbor selection was introduced.

```python
torch.topk(scores, k)
```

### Benefits

* Reduces sorting overhead
* Faster neighbor selection
* Improves scalability for large graphs

---

## 4.3 Efficient Submodular Optimization

A lazy greedy algorithm was implemented to reduce redundant computations in facility location optimization.

### Improvements

* Heap-based priority selection
* Reduced repeated gain calculations
* Faster approximation

This significantly reduced computational overhead compared to naive greedy methods.

---

# 5. GPU Acceleration

Several operations were shifted to GPU execution.

## Similarity Computation

```python
q = F.normalize(x_query)
n = F.normalize(x_neighbors)
similarity = (n * q).sum(dim=-1)
```

## Kernel Matrix Computation

Matrix multiplication operations were executed on GPU for faster performance.

### Benefits

* Faster tensor operations
* Reduced execution time for large datasets
* Better utilization of available hardware resources

---

# 6. Memory Optimization

## 6.1 Weight Caching

Previously computed weights were stored and reused.

```python
torch.save(weights, path)
weights = torch.load(path)
```

### Benefits

* Eliminates repeated preprocessing
* Faster experimentation cycles

---

## 6.2 Reduced Precision Storage

Where appropriate, float16 precision was used to reduce memory consumption.

---

## 6.3 Intermediate Tensor Cleanup

Unused tensors were removed to prevent memory overflow.

```python
del tensor
torch.cuda.empty_cache()
```

---

# 7. Training Optimization

## Mini-Batch Training

Mini-batch training was used to improve training scalability.

```python
batch_size = 1024
```

---

## Optimized Data Loading

Training loaders were improved using:

* `pin_memory=True`
* `persistent_workers=True`
* Multiple workers

### Benefits

* Faster CPU-GPU transfer
* Reduced training bottlenecks

---

# 8. Benchmarking and Performance Evaluation

A benchmarking module was introduced to measure:

* Preprocessing time
* Per-epoch training time
* Accuracy
* F1-score
* Total runtime

## Example Output

```text
Epoch 1: Accuracy = 81%
Epoch 2: Accuracy = 85%
Best Accuracy = 87%
```

This helped evaluate optimization effectiveness.

---

# 9. Experimental Improvements

The implemented optimizations resulted in:

| Metric              | Improvement |
| ------------------- | ----------- |
| Preprocessing Speed | Improved    |
| Training Efficiency | Improved    |
| Memory Usage        | Reduced     |
| GPU Utilization     | Increased   |
| Reusability         | Improved    |

The optimized system performed significantly better than the baseline implementation while preserving prediction quality.

---

# 10. Validation

To ensure correctness:

* Model outputs were compared with the original implementation
* Accuracy and F1 scores were monitored
* Sampling correctness was verified
* Training stability was evaluated

This ensured that optimization changes did not negatively impact model performance.

---

# 11. Technologies Used

* Python
* PyTorch
* PyTorch Geometric
* Multiprocessing
* CUDA
* NumPy

---

# 12. Conclusion

This project successfully optimized the AGS-GNN framework by addressing major computational bottlenecks in preprocessing, sampling, and training.

The introduction of parallel processing, GPU acceleration, caching, and efficient sampling significantly improved system performance while maintaining model accuracy.

These optimizations make AGS-GNN more suitable for handling large-scale graph datasets efficiently.

---

## Future Work

* Distributed multi-GPU training
* Mixed precision training
* Further optimization for extremely large graphs
* Real-world deployment on larger datasets
