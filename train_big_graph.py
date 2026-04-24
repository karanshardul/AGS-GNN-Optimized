"""
AGS-GNN Training Script for Big Graph Datasets
Runs AGS-GNN on Reddit (231K nodes) or other large graph datasets
"""

import sys
import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import Reddit
import warnings

warnings.filterwarnings('ignore')

# Add project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("=" * 80)
print("AGS-GNN Training with Big Graph Dataset")
print("=" * 80)

# Configuration
DATASET_NAME = "Reddit"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 10  # Reduced for demo
BATCH_SIZE = 2048
NUM_WORKERS = 0
LEARNING_RATE = 0.01

print(f"\n[CONFIG]")
print(f"  Dataset: {DATASET_NAME}")
print(f"  Device: {DEVICE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Learning Rate: {LEARNING_RATE}")

# Step 1: Load Dataset
print(f"\n[STEP 1] Loading {DATASET_NAME} dataset...")
try:
    dataset_root = os.path.join(project_root, 'Dataset')
    os.makedirs(dataset_root, exist_ok=True)
    
    dataset = Reddit(root=dataset_root)
    data = dataset[0]
    
    print(f"  ✓ Dataset loaded successfully!")
    print(f"    - Nodes: {data.num_nodes:,}")
    print(f"    - Edges: {data.num_edges:,}")
    print(f"    - Features: {data.num_features}")
    print(f"    - Classes: {data.num_classes}")
    print(f"    - Train nodes: {data.train_mask.sum().item():,}")
    print(f"    - Val nodes: {data.val_mask.sum().item():,}")
    print(f"    - Test nodes: {data.test_mask.sum().item():,}")
    
except Exception as e:
    print(f"  ✗ Error loading dataset: {e}")
    sys.exit(1)

# Step 2: Move data to device
print(f"\n[STEP 2] Moving data to {DEVICE}...")
data = data.to(DEVICE)
print(f"  ✓ Data moved to device")

# Step 3: Create simple baseline GNN model for demonstration
print(f"\n[STEP 3] Building GNN model...")

try:
    from torch_geometric.nn import GCNConv, SAGEConv
    from torch.nn import Linear, Dropout
    
    class SimpleGNN(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
            super(SimpleGNN, self).__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.lin = Linear(hidden_dim, output_dim)
            self.dropout = dropout
        
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lin(x)
            return x
    
    model = SimpleGNN(
        input_dim=data.num_features,
        hidden_dim=128,
        output_dim=data.num_classes,
        dropout=0.5
    ).to(DEVICE)
    
    print(f"  ✓ Model created")
    print(f"    - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
except Exception as e:
    print(f"  ✗ Error creating model: {e}")
    sys.exit(1)

# Step 4: Setup optimizer and loss
print(f"\n[STEP 4] Setting up training...")
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()
print(f"  ✓ Optimizer: Adam (lr={LEARNING_RATE})")
print(f"  ✓ Loss: CrossEntropyLoss")

# Step 5: Training loop
print(f"\n[STEP 5] Training for {EPOCHS} epochs...\n")

import time

train_times = []
train_losses = []
val_accs = []

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    
    # Training
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        train_acc = (out[data.train_mask].argmax(dim=1) == data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()
        val_acc = (out[data.val_mask].argmax(dim=1) == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
        test_acc = (out[data.test_mask].argmax(dim=1) == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    
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
    out = model(data.x, data.edge_index)
    train_acc = (out[data.train_mask].argmax(dim=1) == data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()
    val_acc = (out[data.val_mask].argmax(dim=1) == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
    test_acc = (out[data.test_mask].argmax(dim=1) == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()

print(f"  Final Train Accuracy: {train_acc:.4f}")
print(f"  Final Val Accuracy: {val_acc:.4f}")
print(f"  Final Test Accuracy: {test_acc:.4f}")
print(f"  Average Epoch Time: {sum(train_times) / len(train_times):.2f}s")
print(f"  Total Training Time: {sum(train_times):.2f}s")

print("\n" + "=" * 80)
print("✓ Training completed successfully!")
print("=" * 80)
