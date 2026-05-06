"""Prepare Sleep-EDF PSG + hypnogram EDFs into per-subject NPZ files.

Usage example:
    python -m preprocess.prepare \
        --data_root data/physionet.org/files/sleep-edfx/1.0.0 \
        --output_dir data/sleep_edf_npz

The command scans both Sleep-Cassette and Sleep-Telemetry subfolders,
extracts a single EEG channel, segments 30-second epochs, removes
unknown/movement annotations, and writes one NPZ per subject.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import ntpath
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from mne.io import read_raw_edf

from .edf_reader import BaseEDFReader


W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
UNKNOWN = 5

stage_dict = {"W": W, "N1": N1, "N2": N2, "N3": N3, "REM": REM, "UNKNOWN": UNKNOWN}

ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 5,
    "Movement time": 5,
}

EPOCH_SEC_SIZE = 30


@dataclass
class PrepResult:
    split: str
    subject_id: str
    output_path: str
    n_epochs: int
    sampling_rate: float
    channel: str


def _discover_pairs(split_dir: str) -> List[Tuple[str, str]]:
    def _record_token(path: str, suffix: str) -> str:
        name = os.path.basename(path)
        if not name.endswith(suffix):
            return ""
        return name[: -len(suffix)]

    def _relaxed_key(token: str) -> str:
        # Sleep-EDF commonly differs only by the last character before suffix:
        # e.g., SC4011E0-PSG.edf vs SC4011EH-Hypnogram.edf.
        return token[:-1] if len(token) > 1 else token

    psg_fnames = sorted(glob.glob(os.path.join(split_dir, "*PSG.edf")))
    ann_fnames = sorted(glob.glob(os.path.join(split_dir, "*Hypnogram.edf")))

    ann_by_exact = {
        _record_token(ann, "-Hypnogram.edf"): ann for ann in ann_fnames if _record_token(ann, "-Hypnogram.edf")
    }
    ann_by_relaxed = {}
    for ann in ann_fnames:
        token = _record_token(ann, "-Hypnogram.edf")
        if not token:
            continue
        key = _relaxed_key(token)
        ann_by_relaxed.setdefault(key, []).append(ann)

    pairs = []
    for psg in psg_fnames:
        psg_token = _record_token(psg, "-PSG.edf")
        if not psg_token:
            continue

        # Prefer exact token match when present.
        ann = ann_by_exact.get(psg_token)
        if ann is None:
            candidates = ann_by_relaxed.get(_relaxed_key(psg_token), [])
            if len(candidates) == 1:
                ann = candidates[0]
            elif len(candidates) > 1:
                # Deterministic fallback in the unlikely ambiguous case.
                ann = sorted(candidates)[0]
                print( f"Warning: multiple hypnograms matched {os.path.basename(psg)}; using {os.path.basename(ann)}")

        if ann is not None:
            pairs.append((psg, ann))
    return pairs


def _resolve_channel(ch_names: Sequence[str], requested: str) -> str:
    if requested in ch_names:
        return requested
    candidate = requested.replace("EEG ", "")
    for ch in ch_names:
        if requested.lower() in ch.lower() or candidate.lower() in ch.lower():
            return ch
    for fallback in ("EEG Fpz-Cz", "Fpz-Cz", "EEG Pz-Oz", "Pz-Oz"):
        if fallback in ch_names:
            return fallback
    return ch_names[0]


def _read_headers(psg_path: str, ann_path: str):
    with open(psg_path, "rb") as f:
        reader_raw = BaseEDFReader(f)
        reader_raw.read_header()
        h_raw = reader_raw.header
    with open(ann_path, "rb") as f:
        reader_ann = BaseEDFReader(f)
        reader_ann.read_header()
        h_ann = reader_ann.header
    return h_raw, h_ann


def _collect_annotations(ann_path: str):
    with open(ann_path, "rb") as f:
        reader_ann = BaseEDFReader(f)
        reader_ann.read_header()
        records = list(reader_ann.records())
    return records


def _process_pair(psg_path: str, ann_path: str, select_ch: str, output_dir: str) -> PrepResult:
    raw = read_raw_edf(psg_path, preload=True, stim_channel=None, verbose="ERROR")
    sampling_rate = float(raw.info["sfreq"])
    channel = _resolve_channel(raw.ch_names, select_ch)
    raw_ch = raw.get_data(picks=[channel])[0]

    h_raw, h_ann = _read_headers(psg_path, ann_path)
    raw_start_dt = datetime.strptime(h_raw["date_time"], "%Y-%m-%d %H:%M:%S")
    ann_start_dt = datetime.strptime(h_ann["date_time"], "%Y-%m-%d %H:%M:%S")
    if raw_start_dt != ann_start_dt:
        raise AssertionError(f"Start times mismatch for {psg_path} and {ann_path}")

    ann_records = _collect_annotations(ann_path)
    all_events = [event for _, _, events in ann_records for event in events]

    remove_idx = []
    labels = []
    label_idx = []
    for onset_sec, duration_sec, ann_tokens in all_events:
        ann_str = "".join(ann_tokens).strip()
        label = ann2label.get(ann_str, UNKNOWN)
        sample_start = int(onset_sec * sampling_rate)
        sample_len = int(duration_sec * sampling_rate)
        idx = sample_start + np.arange(sample_len, dtype=np.int64)
        if label != UNKNOWN:
            if duration_sec % EPOCH_SEC_SIZE != 0:
                raise ValueError(f"Unexpected duration {duration_sec} in {ann_path}")
            duration_epoch = int(duration_sec / EPOCH_SEC_SIZE)
            labels.append(np.ones(duration_epoch, dtype=np.int32) * label)
            label_idx.append(idx)
        else:
            remove_idx.append(idx)

    labels = np.hstack(labels) if labels else np.asarray([], dtype=np.int32)

    if remove_idx:
        remove_idx = np.hstack(remove_idx)
        select_idx = np.setdiff1d(np.arange(len(raw_ch)), remove_idx)
    else:
        select_idx = np.arange(len(raw_ch))

    if label_idx:
        label_idx = np.hstack(label_idx)
        select_idx = np.intersect1d(select_idx, label_idx)
    else:
        raise ValueError(f"No labeled epochs found in {ann_path}")

    if len(label_idx) > len(select_idx):
        extra_idx = np.setdiff1d(label_idx, select_idx)
        if extra_idx.size and np.all(extra_idx > select_idx[-1]):
            n_label_trims = int(math.ceil(len(extra_idx) / (EPOCH_SEC_SIZE * sampling_rate)))
            if n_label_trims != 0:
                labels = labels[:-n_label_trims]

    raw_ch = raw_ch[select_idx]
    epoch_len = int(EPOCH_SEC_SIZE * sampling_rate)
    
    # Trim signal to nearest complete epoch to handle edge cases
    trim_len = (len(raw_ch) // epoch_len) * epoch_len
    raw_ch = raw_ch[:trim_len]
    
    # Trim labels to match signal
    n_epochs = len(raw_ch) // epoch_len
    labels = labels[:n_epochs]
    
    x = np.asarray(np.split(raw_ch, n_epochs)).astype(np.float32)
    y = labels.astype(np.int32)
    if len(x) != len(y):
        raise ValueError(f"Mismatched x/y lengths for {psg_path}: {len(x)} vs {len(y)}")

    nw_idx = np.where(y != stage_dict["W"])[0]
    if nw_idx.size == 0:
        raise ValueError(f"Only wake epochs found in {ann_path}")
    w_edge_mins = 30
    start_idx = max(0, nw_idx[0] - (w_edge_mins * 2))
    end_idx = min(len(y) - 1, nw_idx[-1] + (w_edge_mins * 2))
    sel = np.arange(start_idx, end_idx + 1)
    x = x[sel]
    y = y[sel]

    subject_id = ntpath.basename(psg_path).replace("-PSG.edf", "")
    filename = subject_id + ".npz"
    out_path = os.path.join(output_dir, filename)
    np.savez( out_path, x=x, y=y, fs=sampling_rate, ch_label=channel, header_raw=h_raw, header_annotation=h_ann, subject_id=subject_id)

    return PrepResult( split=os.path.basename(output_dir), subject_id=subject_id, output_path=out_path,
                      n_epochs=int(len(x)), sampling_rate=sampling_rate, channel=channel)


def _iter_requested_splits(data_root: str, splits: Sequence[str]) -> Iterable[Tuple[str, str]]:
    if len(splits) == 1 and splits[0].lower() == "all":
        splits = ("sleep-cassette", "sleep-telemetry")
    for split in splits:
        split_dir = os.path.join(data_root, split)
        if os.path.isdir(split_dir):
            yield split, split_dir


def prepare_sleep_edf(data_root: str, output_dir: str, select_ch: str, splits: Sequence[str], max_subjects: Optional[int] = None):
    #if os.path.exists(output_dir):
        #shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    manifest_rows: List[PrepResult] = []
    for split, split_dir in _iter_requested_splits(data_root, splits):
        split_out = os.path.join(output_dir, split)
        os.makedirs(split_out, exist_ok=True)
        pairs = _discover_pairs(split_dir)
        if max_subjects is not None:
            pairs = pairs[:max_subjects]
        for psg_path, ann_path in pairs:
            subject_id = ntpath.basename(psg_path).replace("-PSG.edf", "")
            filename = subject_id + ".npz"
            out_path = os.path.join(split_out, filename)
            
            # Skip if already processed
            if os.path.exists(out_path):
                print(f"Skipping {split}: {os.path.basename(psg_path)} (already exists)")
                # Load metadata from existing file
                data = np.load(out_path, allow_pickle=True)
                manifest_rows.append(PrepResult(split=split, subject_id=subject_id, output_path=out_path,
                                    n_epochs=int(data['x'].shape[0]), sampling_rate=float(data['fs']), channel=str(data['ch_label'])))
                continue
            
            print(f"Processing {split}: {os.path.basename(psg_path)}")
            manifest_rows.append(_process_pair(psg_path, ann_path, select_ch, split_out))

    manifest_path = os.path.join(output_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "subject_id", "output_path", "n_epochs", "sampling_rate", "channel"])
        for row in manifest_rows:
            writer.writerow([row.split, row.subject_id, row.output_path, row.n_epochs, row.sampling_rate, row.channel])

    print(f"Wrote {len(manifest_rows)} subjects to {output_dir}")
    print(f"Manifest: {manifest_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Root containing sleep-cassette and/or sleep-telemetry folders")
    parser.add_argument("--output_dir", type=str, default="data_edf_npz", help="Output directory for NPZ files")
    parser.add_argument("--select_ch", type=str, default="EEG Fpz-Cz", help="Selected EEG channel")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["all"],
        help='Split folders to process. Use "all" for both sleep-cassette and sleep-telemetry.',
    )
    parser.add_argument("--max_subjects", type=int, default=None, help="Limit subjects per split for smoke tests")
    return parser.parse_args()


def main():
    args = parse_args()
    prepare_sleep_edf(args.data_root, args.output_dir, args.select_ch, args.splits, args.max_subjects)


if __name__ == "__main__":
    main()
