import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
import math
import os

from preprocess.dataset import SleepEDFNPZDataset, SleepEDFSubjectIterableDataset
from models.baseline import Conv1DBaseline
from .metrics import overall_metrics, expected_calibration_error, transition_error


def collate_fn(batch):
    xs = np.stack([b[0] for b in batch])
    ys = np.array([b[1] for b in batch])
    xs = torch.from_numpy(xs)
    ys = torch.from_numpy(ys).long()
    return xs, ys


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {'CPU' if device.type == 'cpu' else f'GPU, {torch.cuda.get_device_name(0)}'}" , flush=True)

    num_workers = max(0, args.num_workers)
    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0

    print(f"Loading test dataset from: {args.test_dir}", flush=True)
    test_files = sorted([os.path.join(args.test_dir, f) for f in os.listdir(args.test_dir) if f.endswith('.npz')])
    test_size = sum(int(np.load(f)['x'].shape[0]) for f in test_files)
    print(f"Found {len(test_files)} test files, {test_size} total samples", flush=True)

    # Create dataset based on loading strategy
    if args.subject_batched:
        test_ds = SleepEDFSubjectIterableDataset(files=test_files, shuffle=False, seed=args.seed)
    else:
        test_ds = SleepEDFNPZDataset(files=test_files, cache_size=args.dataset_cache_size)

    # Compute batch count for progress logging
    test_batches = math.ceil(test_size / args.batch_size) if test_size else 0
    shuffle_flag = not args.subject_batched

    if num_workers > 0:
        loader = DataLoader( test_ds, batch_size=args.batch_size, shuffle=shuffle_flag,collate_fn=collate_fn,
            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers,prefetch_factor=2)
    else:
        loader = DataLoader( test_ds, batch_size=args.batch_size, shuffle=shuffle_flag, collate_fn=collate_fn,
            num_workers=0, pin_memory=pin_memory, persistent_workers=False)

        
    print(f"Data loaded - Test batches: {test_batches} (workers={num_workers}, pin_memory={pin_memory})", flush=True)

    # Extract sample for input shape (works for both dataset types)
    if args.subject_batched:
        sample_x, _ = next(iter(test_ds))
    else:
        sample_x, _ = test_ds[0]
    input_length = sample_x.shape[-1]
    model = Conv1DBaseline(in_channels=1, n_classes=args.n_classes, input_length=input_length)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    ys = []
    ys_pred = []
    probs_all = []
    batch_count = 0
    eval_start = time.time()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device).float()
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            ys.extend(yb.numpy().tolist())
            ys_pred.extend(preds.tolist())
            probs_all.append(probs)
            batch_count += 1
            if batch_count % args.log_interval == 0:
                elapsed = time.time() - eval_start
                samples = batch_count * args.batch_size
                print(f"  eval progress batches {batch_count} samples {samples} elapsed {elapsed:.1f}s ({samples/elapsed:.1f} samples/s)", flush=True)
    probs_all = np.vstack(probs_all)

    mets = overall_metrics(np.array(ys), np.array(ys_pred))
    ece = expected_calibration_error(probs_all, np.array(ys))
    te = transition_error(np.array(ys), np.array(ys_pred))

    print("Evaluation results:")
    print(mets)
    print(f"ECE: {ece:.4f}")
    print(f"Transition error: {te:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_classes', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--dataset_cache_size', type=int, default=8, help='Number of decoded NPZ files to cache per worker')
    parser.add_argument('--subject_batched', action='store_true', help='Use subject-batched iterable dataset to minimize per-sample I/O')
    parser.add_argument('--log_interval', type=int, default=100, help='Print evaluation progress every N batches')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

def main():
    args = parse_args()
    evaluate(args)

if __name__ == '__main__':
    main()
