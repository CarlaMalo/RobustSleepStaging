import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from preprocess import SleepEDFNPZDataset
from models import Conv1DBaseline
from .metrics import overall_metrics, expected_calibration_error, transition_error


def collate_fn(batch):
    xs = np.stack([b[0] for b in batch])
    ys = np.array([b[1] for b in batch])
    xs = torch.from_numpy(xs)
    ys = torch.from_numpy(ys).long()
    return xs, ys


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_ds = SleepEDFNPZDataset(args.test_dir)
    loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate_fn)

    sample_x, _ = test_ds[0]
    input_length = sample_x.shape[-1]
    model = Conv1DBaseline(in_channels=1, n_classes=args.n_classes, input_length=input_length)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    ys = []
    ys_pred = []
    probs_all = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device).float()
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            ys.extend(yb.numpy().tolist())
            ys_pred.extend(preds.tolist())
            probs_all.append(probs)
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
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
