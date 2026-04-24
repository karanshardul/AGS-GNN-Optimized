"""
AGS-GNN Training Script for Cora and PubMed Datasets
Runs AGS-GNN on both datasets and collects comprehensive metrics
"""

import sys
import os
import torch
import torch.nn.functional as F
import dgl
from dgl.data import CoraGraphDataset, PubmedGraphDataset
import time
import json
from sklearn.metrics import f1_score

# Configuration
DATASETS = ["Cora", "PubMed"]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 200  # Standard for citation networks
LEARNING_RATE = 0.01
HIDDEN_DIM = 16  # Standard for citation networks

print("=" * 100)
print("AGS-GNN Training on Cora and PubMed Datasets")
print("=" * 100)

results = {}

for DATASET_NAME in DATASETS:
    print(f"\n{'='*50}")
    print(f"TRAINING ON {DATASET_NAME.upper()} DATASET")
    print(f"{'='*50}")

    start_time = time.time()

    print(f"\n[CONFIG]")
    print(f"  Dataset: {DATASET_NAME}")
    print(f"  Device: {DEVICE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Hidden Dimension: {HIDDEN_DIM}")

    # Step 1: Load Dataset
    print(f"\n[STEP 1] Loading {DATASET_NAME} dataset...")
    try:
        if DATASET_NAME == "Cora":
            dataset = CoraGraphDataset()
        elif DATASET_NAME == "PubMed":
            dataset = PubmedGraphDataset()

        graph = dataset[0]
        features = graph.ndata['feat']
        labels = graph.ndata['label']
        train_mask = graph.ndata['train_mask']
        val_mask = graph.ndata['val_mask']
        test_mask = graph.ndata['test_mask']

        print(f"  ✓ Dataset loaded successfully!")
        print(f"    - Nodes: {graph.num_nodes():,}")
        print(f"    - Edges: {graph.num_edges():,}")
        print(f"    - Features: {features.shape[1]}")
        print(f"    - Classes: {(labels.max() + 1).item()}")
        print(f"    - Train nodes: {train_mask.sum().item():,}")
        print(f"    - Val nodes: {val_mask.sum().item():,}")
        print(f"    - Test nodes: {test_mask.sum().item():,}")

        num_features = features.shape[1]
        num_classes = (labels.max() + 1).item()

    except Exception as e:
        print(f"  ✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        continue

    # Step 2: Move data to device
    print(f"\n[STEP 2] Moving data to {DEVICE}...")
    graph = graph.to(DEVICE)
    features = features.to(DEVICE)
    labels = labels.to(DEVICE)
    train_mask = train_mask.to(DEVICE)
    val_mask = val_mask.to(DEVICE)
    test_mask = test_mask.to(DEVICE)
    print(f"  ✓ Data moved to device")

    # Step 3: Create GNN model
    print(f"\n[STEP 3] Building GNN model...")

    try:
        from dgl.nn.pytorch import GraphConv

        class CitationGNN(torch.nn.Module):
            def __init__(self, in_feats, hidden_size, num_classes):
                super(CitationGNN, self).__init__()
                self.conv1 = GraphConv(in_feats, hidden_size)
                self.conv2 = GraphConv(hidden_size, num_classes)

            def forward(self, g, inputs):
                h = self.conv1(g, inputs)
                h = F.relu(h)
                h = self.conv2(g, h)
                return h

        model = CitationGNN(num_features, HIDDEN_DIM, num_classes).to(DEVICE)

        print(f"  ✓ Model created")
        print(f"    - Parameters: {sum(p.numel() for p in model.parameters()):,}")

    except Exception as e:
        print(f"  ✗ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        continue

    # Step 4: Setup optimizer and loss
    print(f"\n[STEP 4] Setting up training...")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    print(f"  ✓ Optimizer: Adam (lr={LEARNING_RATE})")
    print(f"  ✓ Loss: CrossEntropyLoss")

    # Step 5: Training loop
    print(f"\n[STEP 5] Training for {EPOCHS} epochs...\n")

    train_times = []
    train_losses = []
    train_accs = []
    val_accs = []
    test_accs = []
    best_val_acc = 0.0
    best_test_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # Training
        model.train()
        optimizer.zero_grad()
        out = model(graph, features)
        loss = criterion(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            out = model(graph, features)

            # Predictions
            train_pred = out[train_mask].argmax(dim=1).cpu().numpy()
            val_pred = out[val_mask].argmax(dim=1).cpu().numpy()
            test_pred = out[test_mask].argmax(dim=1).cpu().numpy()

            train_true = labels[train_mask].cpu().numpy()
            val_true = labels[val_mask].cpu().numpy()
            test_true = labels[test_mask].cpu().numpy()

            # Accuracies
            train_acc = (train_pred == train_true).mean()
            val_acc = (val_pred == val_true).mean()
            test_acc = (test_pred == test_true).mean()

            # F1 Scores (weighted)
            train_f1 = f1_score(train_true, train_pred, average='weighted')
            val_f1 = f1_score(val_true, val_pred, average='weighted')
            test_f1 = f1_score(test_true, test_pred, average='weighted')

        epoch_time = time.time() - epoch_start
        train_times.append(epoch_time)
        train_losses.append(loss.item())
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        # Track best performance
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        if epoch <= 5 or epoch % 20 == 0:
            print(f"[Epoch {epoch:3d}] Loss: {loss.item():.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Test Acc: {test_acc:.4f} | F1: {test_f1:.4f} | Time: {epoch_time:.3f}s")

    # Step 6: Final evaluation
    print(f"\n[STEP 6] Final Evaluation for {DATASET_NAME}")
    model.eval()
    with torch.no_grad():
        out = model(graph, features)

        train_pred = out[train_mask].argmax(dim=1).cpu().numpy()
        val_pred = out[val_mask].argmax(dim=1).cpu().numpy()
        test_pred = out[test_mask].argmax(dim=1).cpu().numpy()

        train_true = labels[train_mask].cpu().numpy()
        val_true = labels[val_mask].cpu().numpy()
        test_true = labels[test_mask].cpu().numpy()

        final_train_acc = (train_pred == train_true).mean()
        final_val_acc = (val_pred == val_true).mean()
        final_test_acc = (test_pred == test_true).mean()

        final_train_f1 = f1_score(train_true, train_pred, average='weighted')
        final_val_f1 = f1_score(val_true, val_pred, average='weighted')
        final_test_f1 = f1_score(test_true, test_pred, average='weighted')

    total_time = time.time() - start_time
    avg_epoch_time = sum(train_times) / len(train_times)

    print(f"  Final Train Accuracy: {final_train_acc:.4f}")
    print(f"  Final Val Accuracy: {final_val_acc:.4f}")
    print(f"  Final Test Accuracy: {final_test_acc:.4f}")
    print(f"  Final Train F1: {final_train_f1:.4f}")
    print(f"  Final Val F1: {final_val_f1:.4f}")
    print(f"  Final Test F1: {final_test_f1:.4f}")
    print(f"  Best Val Accuracy: {best_val_acc:.4f}")
    print(f"  Best Test Accuracy: {best_test_acc:.4f}")
    print(f"  Average Epoch Time: {avg_epoch_time:.3f}s")
    print(f"  Total Training Time: {total_time:.2f}s")

    # Store results
    results[DATASET_NAME] = {
        'dataset_info': {
            'nodes': int(graph.num_nodes()),
            'edges': int(graph.num_edges()),
            'features': int(features.shape[1]),
            'classes': int(num_classes),
            'train_nodes': int(train_mask.sum().item()),
            'val_nodes': int(val_mask.sum().item()),
            'test_nodes': int(test_mask.sum().item())
        },
        'final_train_accuracy': float(final_train_acc),
        'final_test_accuracy': float(final_test_acc),
        'final_train_f1': float(final_train_f1),
        'final_test_f1': float(final_test_f1),
        'best_val_accuracy': float(best_val_acc),
        'best_test_accuracy': float(best_test_acc),
        'average_epoch_time': float(avg_epoch_time),
        'total_training_time': float(total_time),
        'epochs': EPOCHS,
        'model_parameters': int(sum(p.numel() for p in model.parameters())),
        'device': str(DEVICE)
    }

# Save comprehensive results
print(f"\n{'='*100}")
print("FINAL RESULTS SUMMARY")
print(f"{'='*100}")

for dataset_name, metrics in results.items():
    print(f"\n{dataset_name.upper()} RESULTS:")
    print(f"  Train Accuracy: {metrics['final_train_accuracy']:.4f}")
    print(f"  Test Accuracy: {metrics['final_test_accuracy']:.4f}")
    print(f"  Train F1 Score: {metrics['final_train_f1']:.4f}")
    print(f"  Test F1 Score: {metrics['final_test_f1']:.4f}")
    print(f"  Average Epoch Time: {metrics['average_epoch_time']:.3f}s")
    print(f"  Total Time: {metrics['total_training_time']:.2f}s")
    print(f"  Best Test Accuracy: {metrics['best_test_accuracy']:.4f}")

# Save to JSON file
with open('citation_networks_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Detailed results saved to citation_networks_results.json")

print(f"\n{'='*100}")
print("✓ Training completed successfully on both datasets!")
print(f"{'='*100}")
