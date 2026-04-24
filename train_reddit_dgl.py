"""
AGS-GNN Training Script for Big Graph Datasets using DGL
Runs a GNN model on Reddit (231K nodes) large graph dataset
"""

import sys
import os
import torch
import torch.nn.functional as F
import dgl
from dgl.data import RedditDataset
import time

# Configuration
DATASET_NAME = "Reddit"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 10  # Reduced for demo
BATCH_SIZE = 2048
NUM_WORKERS = 0
LEARNING_RATE = 0.01

print("=" * 80)
print("AGS-GNN Big Graph Training")
print("=" * 80)

print(f"\n[CONFIG]")
print(f"  Dataset: {DATASET_NAME}")
print(f"  Device: {DEVICE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Learning Rate: {LEARNING_RATE}")

# Step 1: Load Dataset
print(f"\n[STEP 1] Loading {DATASET_NAME} dataset...")
try:
    dataset = RedditDataset()
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
    sys.exit(1)

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
    
    class DGLGraphConvNet(torch.nn.Module):
        def __init__(self, in_feats, hidden_size, num_classes):
            super(DGLGraphConvNet, self).__init__()
            self.conv1 = GraphConv(in_feats, hidden_size)
            self.conv2 = GraphConv(hidden_size, hidden_size)
            self.fc = torch.nn.Linear(hidden_size, num_classes)
            self.dropout = torch.nn.Dropout(0.5)
        
        def forward(self, g, inputs):
            h = self.conv1(g, inputs)
            h = F.relu(h)
            h = self.dropout(h)
            h = self.conv2(g, h)
            h = F.relu(h)
            h = self.dropout(h)
            h = self.fc(h)
            return h
    
    model = DGLGraphConvNet(num_features, 128, num_classes).to(DEVICE)
    
    print(f"  ✓ Model created")
    print(f"    - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
except Exception as e:
    print(f"  ✗ Error creating model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

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
val_accs = []

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    
    # Training
    model.train()
    optimizer.zero_grad()
    out = model(graph, features)
    loss = criterion(out[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        out = model(graph, features)
        train_acc = (out[train_mask].argmax(dim=1) == labels[train_mask]).sum().item() / train_mask.sum().item()
        val_acc = (out[val_mask].argmax(dim=1) == labels[val_mask]).sum().item() / val_mask.sum().item()
        test_acc = (out[test_mask].argmax(dim=1) == labels[test_mask]).sum().item() / test_mask.sum().item()
    
    epoch_time = time.time() - epoch_start
    train_times.append(epoch_time)
    train_losses.append(loss.item())
    val_accs.append(val_acc)
    
    print(f"[Epoch {epoch:3d}] Loss: {loss.item():.4f} | "
          f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
          f"Test Acc: {test_acc:.4f} | Time: {epoch_time:.2f}s")

# Step 6: Final evaluation
print(f"\n[STEP 6] Final Evaluation")
model.eval()
with torch.no_grad():
    out = model(graph, features)
    train_acc = (out[train_mask].argmax(dim=1) == labels[train_mask]).sum().item() / train_mask.sum().item()
    val_acc = (out[val_mask].argmax(dim=1) == labels[val_mask]).sum().item() / val_mask.sum().item()
    test_acc = (out[test_mask].argmax(dim=1) == labels[test_mask]).sum().item() / test_mask.sum().item()

print(f"  Final Train Accuracy: {train_acc:.4f}")
print(f"  Final Val Accuracy: {val_acc:.4f}")
print(f"  Final Test Accuracy: {test_acc:.4f}")
print(f"  Average Epoch Time: {sum(train_times) / len(train_times):.2f}s")
print(f"  Total Training Time: {sum(train_times):.2f}s")

# Save results
results = {
    'dataset': DATASET_NAME,
    'device': str(DEVICE),
    'epochs': EPOCHS,
    'final_train_acc': float(train_acc),
    'final_val_acc': float(val_acc),
    'final_test_acc': float(test_acc),
    'avg_epoch_time': float(sum(train_times) / len(train_times)),
    'total_training_time': float(sum(train_times)),
}

import json
with open('big_graph_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  ✓ Results saved to big_graph_results.json")

print("\n" + "=" * 80)
print("✓ Training completed successfully!")
print("=" * 80)
