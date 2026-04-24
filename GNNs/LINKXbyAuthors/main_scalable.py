import argparse
import sys
import os
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, sort_edge_index
from torch_geometric.data import NeighborSampler, ClusterData, ClusterLoader, Data, GraphSAINTNodeSampler, GraphSAINTEdgeSampler, GraphSAINTRandomWalkSampler, RandomNodeSampler
from torch_scatter import scatter

from logger import Logger, SimpleLogger
from dataset import load_nc_dataset, NCDataset
from data_utils import normalize, gen_normalized_adjs, evaluate, eval_acc, eval_rocauc, to_sparse_tensor
from parse import parse_method, parser_add_main_args
from batch_utils import nc_dataset_to_torch_geo, torch_geo_to_nc_dataset, AdjRowLoader, make_loader
from benchmark_utils import (
    build_hyperparameter_dict,
    compute_test_accuracy_and_f1_macro,
    measure_ags_precompute_knn_submodular,
    save_benchmark_json,
    set_global_seed,
)

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
parser.add_argument('--train_batch', type=str, default='cluster', help='type of mini batch loading scheme for training GNN')
parser.add_argument('--no_mini_batch_test', action='store_true', help='whether to test on mini batches as well')
parser.add_argument('--batch_size', type=int, default=10000)
parser.add_argument('--num_parts', type=int, default=100, help='number of partitions for partition batching')
parser.add_argument('--cluster_batch_size', type=int, default=1, help='number of clusters to use per cluster-gcn step')
parser.add_argument('--saint_num_steps', type=int, default=5, help='number of steps for graphsaint')
parser.add_argument('--test_num_parts', type=int, default=10, help='number of partitions for testing')
# Benchmark / correctness (measurement only — does not change model math)
parser.add_argument(
    '--benchmark_json',
    type=str,
    default='',
    help='If set, write timing & metrics JSON to this path (e.g. baseline_results.json)',
)
parser.add_argument(
    '--seed',
    type=int,
    default=0,
    help='Random seed when --benchmark_json is set (dataset split + torch/numpy)',
)
parser.add_argument(
    '--benchmark_ags_precompute',
    action='store_true',
    help='Include KNN + submodular precompute time (requires Submodular/ags_pipeline)',
)
args = parser.parse_args()
print(args)

if args.benchmark_json:
    set_global_seed(args.seed)
else:
    np.random.seed(0)

device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
if args.cpu:
    device = torch.device('cpu')

### Load and preprocess data ###
dataset = load_nc_dataset(args.dataset, args.sub_dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

split_idx = dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
train_idx = split_idx['train']
train_idx = train_idx.to(device)

n = dataset.graph['num_nodes']
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

# whether or not to symmetrize matters a lot!! pay attention to this
# e.g. directed edges are temporally useful in arxiv-year,
# so we usually do not symmetrize, but for label prop symmetrizing helps
if not args.directed and args.dataset != 'ogbn-proteins':
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

benchmark_total_start = time.time() if args.benchmark_json else None

precompute_time = 0.0
if args.benchmark_json and args.benchmark_ags_precompute:
    try:
        precompute_time = measure_ags_precompute_knn_submodular(dataset, device)
        print(f"[benchmark] precompute_time (KNN + submodular): {precompute_time:.4f}s")
    except Exception as exc:
        print(f"[benchmark] AGS precompute skipped: {exc}")
        precompute_time = 0.0

train_loader, subgraph_loader = None, None

print(f"num nodes {n} | num classes {c} | num node feats {d}")

### Load method ###

model = parse_method(args, dataset, n, c, d, device)


# using rocauc as the eval function
if args.rocauc or args.dataset in ('yelp-chi', 'twitch-e', 'ogbn-proteins'):
    criterion = nn.BCEWithLogitsLoss()
    eval_func = eval_rocauc
else:
    criterion = nn.NLLLoss()
    eval_func = eval_acc

logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)



def _train_step(tg_batch):
    """Single batch forward-backward (shared by timed and plain loops)."""
    batch_train_idx = tg_batch.mask.to(torch.bool)
    batch_dataset = torch_geo_to_nc_dataset(tg_batch, device=device)
    optimizer.zero_grad()
    out = model(batch_dataset)
    if args.rocauc or args.dataset in ('yelp-chi', 'twitch-e', 'ogbn-proteins'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(batch_dataset.label, batch_dataset.label.max() + 1).squeeze(1)
        else:
            true_label = batch_dataset.label
        loss = criterion(out[batch_train_idx], true_label[batch_train_idx].to(out.dtype))
    else:
        out = F.log_softmax(out, dim=1)
        loss = criterion(
            out[batch_train_idx], batch_dataset.label.squeeze(1)[batch_train_idx])
    loss.backward()
    optimizer.step()
    return loss


def train():
    model.train()
    total_loss = 0
    for tg_batch in train_loader:
        loss = _train_step(tg_batch)
        total_loss += loss
    return total_loss


def train_benchmark():
    """Train one epoch with loader vs compute timing (time.time)."""
    model.train()
    total_loss = 0
    sampling_time = 0.0
    training_time = 0.0
    it = iter(train_loader)
    while True:
        t0 = time.time()
        try:
            tg_batch = next(it)
        except StopIteration:
            break
        sampling_time += time.time() - t0
        t1 = time.time()
        loss = _train_step(tg_batch)
        total_loss += loss
        training_time += time.time() - t1
    return total_loss, sampling_time, training_time


def test():
    model.eval()
    full_out = torch.zeros(n, c, device=device)
    with torch.no_grad():
        for tg_batch in test_loader:
            node_ids = tg_batch.node_ids
            batch_dataset = torch_geo_to_nc_dataset(tg_batch, device=device)
            out = model(batch_dataset)
            full_out[node_ids] = out
    result = evaluate(model, dataset, split_idx, eval_func, result=full_out, sampling=args.sampling, subgraph_loader=subgraph_loader)
    logger.add_result(run, result[:-1])
    return result


def test_benchmark():
    """Test pass with loader vs forward timing (time.time)."""
    model.eval()
    full_out = torch.zeros(n, c, device=device)
    sampling_time = 0.0
    inference_time = 0.0
    with torch.no_grad():
        it = iter(test_loader)
        while True:
            t0 = time.time()
            try:
                tg_batch = next(it)
            except StopIteration:
                break
            sampling_time += time.time() - t0
            t1 = time.time()
            node_ids = tg_batch.node_ids
            batch_dataset = torch_geo_to_nc_dataset(tg_batch, device=device)
            out = model(batch_dataset)
            full_out[node_ids] = out
            inference_time += time.time() - t1
    result = evaluate(model, dataset, split_idx, eval_func, result=full_out, sampling=args.sampling, subgraph_loader=subgraph_loader)
    logger.add_result(run, result[:-1])
    return result, sampling_time, inference_time


### Training loop ###
benchmark_epoch_walls = []
benchmark_train_sampling = []
benchmark_train_compute = []
benchmark_test_sampling = []
benchmark_test_compute = []
last_epoch_result = None
last_epoch_split_idx = None
if args.benchmark_json:
    if args.runs > 1:
        print(
            "[benchmark] Warning: --benchmark_json saves metrics from run 0 only; "
            "use --runs 1 for apples-to-apples comparisons."
        )

for run in range(args.runs):
    train_idx = split_idx['train']
    train_idx = train_idx.to(device)

    print('making train loader:', device)
    
    train_loader = make_loader(args, dataset, train_idx, device=device)
    if not args.no_mini_batch_test:
        test_loader = make_loader(args, dataset, train_idx, device=device, test=True)
    else:
        test_loader = make_loader(args, dataset, split_idx['test'], mini_batch = False, device=device)

    model.reset_parameters()
    if args.adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float('-inf')
    for epoch in range(args.epochs):
        if args.benchmark_json:
            t_ep = time.time()
            total_loss, tr_sample_t, tr_train_t = train_benchmark()
            result, te_sample_t, te_infer_t = test_benchmark()
            epoch_wall = time.time() - t_ep
            benchmark_epoch_walls.append(epoch_wall)
            benchmark_train_sampling.append(tr_sample_t)
            benchmark_train_compute.append(tr_train_t)
            benchmark_test_sampling.append(te_sample_t)
            benchmark_test_compute.append(te_infer_t)
        else:
            total_loss = train()
            result = test()
            epoch_wall = None

        if result[1] > best_val:
            best_out = F.log_softmax(result[-1], dim=1)
            best_val = result[1]

        if epoch % args.display_step == 0:
            if args.benchmark_json:
                test_acc_pct = 100.0 * result[2]
                print(
                    f'Epoch {epoch:02d} | Loss: {float(total_loss):.4f} | '
                    f'Acc: {test_acc_pct:.2f}% | Time: {epoch_wall:.2f}s'
                )
            else:
                print(f'Epoch: {epoch:02d}, '
                      f'Loss: {total_loss:.4f}, '
                      f'Train: {100 * result[0]:.2f}%, '
                      f'Valid: {100 * result[1]:.2f}%, '
                      f'Test: {100 * result[2]:.2f}%')
            if args.print_prop and not args.rocauc:
                pred = result[-1].argmax(dim=-1, keepdim=True)
                print("Predicted proportions:", pred.unique(return_counts=True)[1].float()/pred.shape[0])
        last_epoch_result = result
        last_epoch_split_idx = split_idx
    logger.print_statistics(run)

    if args.benchmark_json and run == 0:
        use_rocauc_eval = args.rocauc or args.dataset in (
            'yelp-chi', 'twitch-e', 'ogbn-proteins')
        acc, f1m = compute_test_accuracy_and_f1_macro(
            dataset, split_idx, result[-1], use_rocauc_eval)
        total_elapsed = time.time() - benchmark_total_start if benchmark_total_start else 0.0
        payload = {
            "accuracy": acc,
            "f1": f1m,
            "precompute_time": precompute_time,
            "epoch_time_avg": float(np.mean(benchmark_epoch_walls)) if benchmark_epoch_walls else 0.0,
            "total_time": total_elapsed,
            "sampling_time_per_epoch_avg": float(np.mean(benchmark_train_sampling)) if benchmark_train_sampling else 0.0,
            "training_time_per_epoch_avg": float(np.mean(benchmark_train_compute)) if benchmark_train_compute else 0.0,
            "test_sampling_time_per_epoch_avg": float(np.mean(benchmark_test_sampling)) if benchmark_test_sampling else 0.0,
            "test_inference_time_per_epoch_avg": float(np.mean(benchmark_test_compute)) if benchmark_test_compute else 0.0,
            "dataset": args.dataset,
            "sub_dataset": args.sub_dataset,
            "seed": args.seed,
            "hyperparameters": build_hyperparameter_dict(args),
        }
        save_benchmark_json(args.benchmark_json, payload)
        f1_str = "null" if f1m is None else f"{f1m:.6f}"
        print(
            f"[benchmark] Saved {args.benchmark_json} | accuracy={acc:.6f} | "
            f"f1={f1_str} | precompute_time={precompute_time:.4f}s | "
            f"epoch_time_avg={payload['epoch_time_avg']:.4f}s | total_time={payload['total_time']:.4f}s"
        )
        print(f"\nFinal metrics | Test accuracy: {acc:.6f} | F1 (macro): {f1_str}\n")

    split_idx = dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)

if (
    not args.benchmark_json
    and last_epoch_result is not None
    and last_epoch_split_idx is not None
):
    use_rocauc_eval = args.rocauc or args.dataset in (
        'yelp-chi', 'twitch-e', 'ogbn-proteins')
    _acc, _f1 = compute_test_accuracy_and_f1_macro(
        dataset, last_epoch_split_idx, last_epoch_result[-1], use_rocauc_eval)
    _f1s = "n/a" if _f1 is None else f"{_f1:.6f}"
    print(f"\nFinal metrics | Test accuracy: {_acc:.6f} | F1 (macro): {_f1s}\n")


### Save results ###
best_val, best_test = logger.print_statistics()
os.makedirs('results', exist_ok=True)
filename = f'results/{args.dataset}.csv'
print(f"Saving results to {filename}")
with open(f"{filename}", 'a+') as write_obj:
    sub_dataset = f'{args.sub_dataset},' if args.sub_dataset else ''
    write_obj.write(f"{args.method}," + f"{sub_dataset}" +
                    f"{best_val.mean():.3f} ± {best_val.std():.3f}," +
                    f"{best_test.mean():.3f} ± {best_test.std():.3f}\n")
