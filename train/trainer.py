import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import time
import math

from preprocess.dataset import SleepEDFNPZDataset, SleepEDFSubjectIterableDataset
from models.baseline import Conv1DBaseline


def collate_fn(batch):
    xs = np.stack([b[0] for b in batch])
    ys = np.array([b[1] for b in batch])
    xs = torch.from_numpy(xs)
    ys = torch.from_numpy(ys).long()
    return xs, ys


def split_subject_files(files, val_fraction=0.2, seed=42):
    files = sorted(files)
    if not files or val_fraction <= 0:
        return files, []
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(files))
    n_val = max(1, int(round(len(files) * val_fraction)))
    val_idx = sorted(perm[:n_val].tolist())
    train_idx = sorted(perm[n_val:].tolist())
    train_files = [files[i] for i in train_idx]
    val_files = [files[i] for i in val_idx]
    return train_files, val_files


def train(args):
    # Print device info and cpu/gpu 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running on CPU" if device.type == "cpu" else f"Running on GPU, {torch.cuda.get_device_name(0)}", flush=True)

    num_workers = max(0, args.num_workers)
    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0
    # Discover train/val files regardless of mode so we can compute exact sample counts
    if args.val_dir:
        train_files = sorted([os.path.join(args.train_dir, f) for f in os.listdir(args.train_dir) if f.endswith('.npz')])
        val_files = sorted([os.path.join(args.val_dir, f) for f in os.listdir(args.val_dir) if f.endswith('.npz')])
        print(f"Loading train dataset from directory: {args.train_dir}", flush=True)
        print(f"Loading validation dataset from directory: {args.val_dir}", flush=True)
    else:
        print(f"Scanning NPZ files in: {args.train_dir}", flush=True)
        all_files = sorted([os.path.join(args.train_dir, f) for f in os.listdir(args.train_dir) if f.endswith('.npz')])
        print(f"Found {len(all_files)} NPZ files; splitting with val_fraction={args.val_fraction}", flush=True)
        train_files, val_files = split_subject_files(all_files, args.val_fraction, args.seed)
        print(f"Loading {len(train_files)} train files and {len(val_files)} validation files", flush=True)

    # compute sizes by scanning headers (cheap relative to full epoch IO)
    train_size = sum(int(np.load(f)['x'].shape[0]) for f in train_files)
    val_size = sum(int(np.load(f)['x'].shape[0]) for f in val_files) if val_files else 0

    # create datasets
    if args.subject_batched:
        train_ds = SleepEDFSubjectIterableDataset(files=train_files, shuffle=True, seed=args.seed)
        val_ds = SleepEDFSubjectIterableDataset(files=val_files, shuffle=False, seed=args.seed) if val_files else None
    else:
        train_ds = SleepEDFNPZDataset(files=train_files, cache_size=args.dataset_cache_size)
        val_ds = SleepEDFNPZDataset(files=val_files, cache_size=args.dataset_cache_size) if val_files else None

    print(f"Train samples: {train_size}, Val samples: {val_size}", flush=True)
    # precompute batch counts (DataLoader length is not available for IterableDataset)
    train_batches = math.ceil(train_size / args.batch_size) if train_size else 0
    val_batches = math.ceil(val_size / args.batch_size) if val_size else 0

    shuffle_flag = not args.subject_batched

    if num_workers > 0:
        train_loader = DataLoader( train_ds, batch_size=args.batch_size, shuffle=shuffle_flag,collate_fn=collate_fn,
            num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers, prefetch_factor=2)
    else:
        train_loader = DataLoader( train_ds, batch_size=args.batch_size, shuffle=shuffle_flag, collate_fn=collate_fn,
            num_workers=0, pin_memory=pin_memory, persistent_workers=False)
    if val_ds:
        if num_workers > 0:
            val_loader = DataLoader( val_ds, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=num_workers,
                pin_memory=pin_memory, persistent_workers=persistent_workers, prefetch_factor=2)
        else:
            val_loader = DataLoader( val_ds, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=0,
                pin_memory=pin_memory, persistent_workers=False)
    else:
        val_loader = None
    print( f"Data loaded - Train batches: {train_batches}, Val batches: {val_batches} "
        f"(workers={num_workers}, pin_memory={pin_memory}, cache_size={args.dataset_cache_size})", flush=True)

    # infer input length from first sample
    if args.subject_batched:
        sample_x, _ = next(iter(train_ds))
    else:
        sample_x, _ = train_ds[0]
    input_length = sample_x.shape[-1]

    model = Conv1DBaseline(in_channels=1, n_classes=args.n_classes, input_length=input_length)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val = -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batch_count = 0
        epoch_start = time.time()
        for xb, yb in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            batch_count += 1
            if batch_count % args.log_interval == 0:
                elapsed = time.time() - epoch_start
                samples = batch_count * args.batch_size
                print(f"  epoch {epoch} batches {batch_count} samples {samples} elapsed {elapsed:.1f}s ({samples/elapsed:.1f} samples/s)", flush=True)
        avg_loss = total_loss / train_size
        print(f"Epoch {epoch}/{args.epochs} train_loss={avg_loss:.4f}", flush=True)

        if val_loader is not None:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device).float()
                    yb = yb.to(device)
                    logits = model(xb)
                    preds = logits.argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)
            val_acc = correct / total
            print(f"  val_acc={val_acc:.4f}", flush=True)
            if val_acc > best_val:
                best_val = val_acc
                torch.save(model.state_dict(), args.checkpoint)

    # final save
    torch.save(model.state_dict(), args.checkpoint_final)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--val_dir', default=None)
    parser.add_argument('--val_fraction', type=float, default=0.2, help='Subject-level validation fraction used when --val_dir is omitted')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for subject-level split')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_classes', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--dataset_cache_size', type=int, default=8, help='Number of decoded NPZ files to cache per worker')
    parser.add_argument('--subject_batched', action='store_true', help='Use subject-batched iterable dataset to minimize per-sample I/O')
    parser.add_argument('--log_interval', type=int, default=100, help='Print training progress every N batches')
    parser.add_argument('--checkpoint', default='checkpoints/checkpoint.pth')
    parser.add_argument('--checkpoint_final', default='checkpoints/checkpoint_final.pth')
    return parser.parse_args()

def main():
    args = parse_args()

    # Check if paths are valid
    if not os.path.exists(args.train_dir):
        raise FileNotFoundError(f"Train directory {args.train_dir} does not exist")
    if args.val_dir and not os.path.exists(args.val_dir):
        raise FileNotFoundError(f"Validation directory {args.val_dir} does not exist")
    checkpoint_dir = os.path.dirname(args.checkpoint)
    # Check if checkpoint directories exist or can be created
    for p in (args.checkpoint, args.checkpoint_final):
        d = os.path.dirname(p)
        if d:
            try:
                os.makedirs(d, exist_ok=True)
            except Exception:
                print(f"Warning: could not create checkpoint dir {d}", flush=True)

    train(args)
    
if __name__ == '__main__':
    main()
