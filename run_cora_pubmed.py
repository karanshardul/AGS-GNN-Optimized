"""
Train and evaluate a GNN on Cora and PubMed datasets using DGL.
Reports train accuracy, test accuracy, per-epoch time, F1 score, and total runtime.
"""

import time
import os
import json
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from dgl.data import CoraGraphDataset, PubmedGraphDataset
from dgl.nn.pytorch import GraphConv

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleGNN(torch.nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, dropout=0.5):
        super(SimpleGNN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, hidden_size)
        self.fc = torch.nn.Linear(hidden_size, num_classes)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, g, x):
        h = self.conv1(g, x)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.fc(h)
        return h


def train_dataset(dataset_name, dataset_loader, epochs=200, lr=0.01):
    print(f"\n=== Running {dataset_name} ===")
    graph = dataset_loader[0]
    graph = graph.remove_self_loop().add_self_loop()

    features = graph.ndata['feat'].to(DEVICE)
    labels = graph.ndata['label'].to(DEVICE)
    train_mask = graph.ndata['train_mask'].to(DEVICE)
    val_mask = graph.ndata['val_mask'].to(DEVICE)
    test_mask = graph.ndata['test_mask'].to(DEVICE)

    model = SimpleGNN(
        in_feats=features.shape[1],
        hidden_size=64,
        num_classes=int(labels.max().item() + 1),
        dropout=0.5,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    epoch_results = []
    total_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        model.train()
        optimizer.zero_grad()
        logits = model(graph, features)
        loss = criterion(logits[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(graph, features)
            train_pred = logits[train_mask].argmax(dim=1)
            val_pred = logits[val_mask].argmax(dim=1)
            test_pred = logits[test_mask].argmax(dim=1)

            train_acc = (train_pred == labels[train_mask]).float().mean().item()
            test_acc = (test_pred == labels[test_mask]).float().mean().item()
            y_true_test = labels[test_mask].cpu().numpy()
            y_pred_test = test_pred.cpu().numpy()
            f1 = f1_score(y_true_test, y_pred_test, average='weighted')

        epoch_time = time.time() - epoch_start
        epoch_results.append({
            'epoch': epoch,
            'loss': loss.item(),
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'f1': float(f1),
            'epoch_time': epoch_time,
        })

        print(f"{dataset_name} Epoch {epoch:03d}: loss={loss.item():.4f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}, f1={f1:.4f}, time={epoch_time:.2f}s")

    total_time = time.time() - total_start
    summary = {
        'dataset': dataset_name,
        'total_time_seconds': total_time,
        'average_epoch_time_seconds': sum(r['epoch_time'] for r in epoch_results) / len(epoch_results),
        'last_train_accuracy': epoch_results[-1]['train_accuracy'],
        'last_test_accuracy': epoch_results[-1]['test_accuracy'],
        'last_f1': epoch_results[-1]['f1'],
        'epochs': epochs,
        'epoch_results': epoch_results,
    }
    print(f"{dataset_name} completed: total_time={total_time:.2f}s, avg_epoch_time={summary['average_epoch_time_seconds']:.2f}s")
    return summary


def main():
    datasets = [
        ('Cora', CoraGraphDataset()),
        ('PubMed', PubmedGraphDataset()),
    ]
    results = {}
    for name, loader in datasets:
        results[name] = train_dataset(name, loader, epochs=100, lr=0.01)

    with open('cora_pubmed_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved results to cora_pubmed_results.json')


if __name__ == '__main__':
    main()
