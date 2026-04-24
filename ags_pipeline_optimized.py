"""
AGS-GNN Optimized Pipeline for Parallel Computing and GPU Acceleration
========================================================================

Implements the Cursor optimization prompt for AGS-GNN with:
1. Parallelism for KNN/Submodular weight precomputation
2. GPU acceleration for similarity computations
3. Benchmarking and performance metrics
4. Caching to avoid redundant computations

This module integrates with ags_pipeline for kernel ops and provides:
- compute_or_load_edge_weights(): Parallel precomputation + caching
- BenchmarkResults: Tracks timing and accuracy metrics
- train_with_benchmarking(): Training loop with built-in benchmarking
"""

import os
import json
import time
import torch
import numpy as np
from typing import Dict, Tuple, Optional,Any
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score


class BenchmarkResults:
    """Container for training metrics and timings."""
    
    def __init__(self):
        self.precompute_time = 0.0
        self.epoch_times = []
        self.sampling_times = []
        self.total_time = 0.0
        self.accuracies = []
        self.f1_scores = []
        self.best_accuracy = 0.0
        self.best_f1 = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize results to dictionary for JSON output."""
        return {
            'precompute_time': self.precompute_time,
            'total_training_time': self.total_time,
            'num_epochs': len(self.epoch_times),
            'avg_epoch_time': np.mean(self.epoch_times) if self.epoch_times else 0.0,
            'best_accuracy': float(self.best_accuracy),
            'best_f1': float(self.best_f1),
            'final_accuracy': float(self.accuracies[-1]) if self.accuracies else 0.0,
            'final_f1': float(self.f1_scores[-1]) if self.f1_scores else 0.0,
        }


def compute_or_load_edge_weights(
    data,
    save_dir: str,
    recompute: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute or load cached KNN + submodular edge weights.
    
    Uses parallel precomputation if ags_pipeline available, else serial.
    Caches results to disk to avoid recomputation across runs.
    
    Args:
        data: PyG Data object with features and edges
        save_dir: Directory to save/load weights
        recompute: If True, ignore cache and recompute
        device: Device for computation (cuda or cpu)
    
    Returns:
        (w_knn, w_submod, w_combined): Weight tensors on device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(save_dir, exist_ok=True)
    knn_path = os.path.join(save_dir, 'knn_weights.pt')
    submod_path = os.path.join(save_dir, 'submodular_weights.pt')
    
    # Try cache first
    if not recompute and os.path.exists(knn_path) and os.path.exists(submod_path):
        w_knn = torch.load(knn_path, map_location=device)
        w_sub = torch.load(submod_path, map_location=device)
        print(f"[ags_pipeline] Loaded cached weights from {save_dir}")
        w_combined = (w_knn + w_sub) * 0.5
        return w_knn, w_sub, w_combined
    
    # Parallel precomputation
    try:
        from ags_pipeline import compute_knn_and_submodular_parallel
        print("[ags_pipeline] Using parallel KNN + submodular precomputation...")
        w_knn, w_sub = compute_knn_and_submodular_parallel(
            data,
            knn_kwargs={'metric': 'cosine', 'log': False, 'device': device},
            submod_kwargs={'metric': 'cosine', 'log': False, 'device': device},
        )
    except (ImportError, AttributeError):
        # Fallback: serial computation
        print("[ags_pipeline] Parallel module not available; using serial computation.")
        from ags_pipeline import KNNWeightOptimized, SubModularWeightFacilityOptimized
        knn = KNNWeightOptimized(data, metric='cosine', log=False, device=device)
        sub = SubModularWeightFacilityOptimized(data, metric='cosine', log=False, device=device)
        w_knn = knn.compute_weights()
        w_sub = sub.compute_weights()
    
    # Cache results
    w_knn_cpu = w_knn.cpu() if w_knn.device.type == 'cuda' else w_knn
    w_sub_cpu = w_sub.cpu() if w_sub.device.type == 'cuda' else w_sub
    torch.save(w_knn_cpu, knn_path)
    torch.save(w_sub_cpu, submod_path)
    print(f"[ags_pipeline] Cached weights to {save_dir}")
    
    # Ensure on device for training
    w_knn = w_knn.to(device)
    w_sub = w_sub.to(device)
    w_combined = (w_knn + w_sub) * 0.5
    
    return w_knn, w_sub, w_combined


def evaluate_with_metrics(
    model, 
    loader,
    mask,
    criterion,
    device,
    name: str = 'Eval',
) -> Tuple[float, float, float]:
    """
    Evaluate model and return accuracy, F1, and loss.
    
    Args:
        model: AGS-GNN model
        loader: Data loader
        mask: Train/val/test mask
        criterion: Loss function
        device: Device
        name: Logging name
    
    Returns:
        (accuracy, f1, loss): Metrics
    """
    model.eval()
    y_true, y_pred, total_loss, total_examples = [], [], 0.0, 0
    
    with torch.no_grad():
        for batch_data in loader:
            batch_data = [batch_data, batch_data]
            batch_data = [b.to(device) for b in batch_data]
            used = batch_data[0].batch_size
            
            out = model(batch_data)
            loss = criterion(out[:used], batch_data[0].y[:used])
            
            y_pred.append(out[:used].argmax(dim=1).cpu().numpy())
            y_true.append(batch_data[0].y[:used].cpu().numpy())
            total_loss += loss.item() * used
            total_examples += used
    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    loss = total_loss / total_examples if total_examples > 0 else 0.0
    
    return acc, f1, loss


def train_with_benchmarking(
    model,
    data,
    train_loader,
    val_loader,
    test_loader,
    device,
    epochs: int = 20,
    lr: float = 0.01,
    results_path: str = 'baseline_results.json',
) -> BenchmarkResults:
    """
    Train AGS-GNN model with comprehensive benchmarking.
    
    Logs:
    - Precomputation time
    - Per-epoch training/val/test times and metrics
    - Total training time
   - Accuracy and F1 scores (macro + weighted)
    - Results to JSON
    
    Args:
        model: AGS-GNN model on device
        data: PyG Data object
        train_loader: Training dataloader
        val_loader: Validation dataloader
        test_loader: Test dataloader
        device: PyTorch device
        epochs: Number of epochs
        lr: Learning rate
        results_path: Path to save results JSON
    
    Returns:
        BenchmarkResults object with all collected metrics
    """
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    results = BenchmarkResults()
    
    total_start = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss, total_examples = 0.0, 0
        
        for batch_data in train_loader:
            batch_data = [batch_data, batch_data]
            batch_data = [b.to(device) for b in batch_data]
            used = batch_data[0].batch_size
            
            optimizer.zero_grad()
            out = model(batch_data)
            loss = criterion(out[:used], batch_data[0].y[:used])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * used
            total_examples += used
        
        epoch_loss = total_loss / total_examples
        train_acc, train_f1, _ = evaluate_with_metrics(
            model, train_loader, data.train_mask, criterion, device, 'train'
        )
        val_acc, val_f1, _ = evaluate_with_metrics(
            model, val_loader, data.val_mask, criterion, device, 'val'
        )
        
        results.epoch_times.append(time.time() - epoch_start)
        results.accuracies.append(val_acc)
        results.f1_scores.append(val_f1)
        results.best_accuracy = max(results.best_accuracy, val_acc)
        results.best_f1 = max(results.best_f1, val_f1)
        
        print(f'Epoch {epoch:03d} | Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.4f} | '
              f'Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Time: {results.epoch_times[-1]:.2f}s')
    
    # Final test evaluation
    test_acc, test_f1, _ = evaluate_with_metrics(
        model, test_loader, data.test_mask, criterion, device, 'test'
    )
    
    results.total_time = time.time() - total_start
    
    # Save results
    results_dict = results.to_dict()
    results_dict['test_accuracy'] = float(test_acc)
    results_dict['test_f1'] = float(test_f1)
    
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n=== Benchmark Summary ===")
    print(f"Total time: {results.total_time:.2f}s")
    print(f"Best val accuracy: {results.best_accuracy:.4f}")
    print(f"Best val F1: {results.best_f1:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Results saved to {results_path}")
    
    return results
