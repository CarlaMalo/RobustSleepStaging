import os
import glob
from collections import OrderedDict
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset, get_worker_info
import math


class SleepEDFNPZDataset(Dataset):
    """Dataset that loads preprocessed per-subject .npz files containing epochs.

    Expects files with arrays 'x' (n_epochs, samples) and 'y' (n_epochs,)
    """
    def __init__(self, npz_dir=None, files=None, transform=None, cache_size=8):
        self.npz_dir = npz_dir
        if files is not None:
            self.files = sorted(files)
        elif npz_dir is not None:
            self.files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
        else:
            self.files = []
        self.transform = transform
        self.cache_size = max(0, int(cache_size))
        self._file_cache = OrderedDict()
        # Build an index mapping global epoch idx -> (file_idx, epoch_idx)
        self.index = []
        self._build_index()

    def _build_index(self):
        for fi, f in enumerate(self.files):
            try:
                with np.load(f) as d:
                    n = d['x'].shape[0]
            except Exception:
                n = 0
            for ei in range(n):
                self.index.append((fi, ei))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, ei = self.index[idx]
        file_path = self.files[fi]
        cached = self._file_cache.get(file_path)
        if cached is None:
            with np.load(file_path) as d:
                x_all = np.asarray(d['x'], dtype=np.float32)
                y_all = np.asarray(d['y'], dtype=np.int64)
            if self.cache_size > 0:
                self._file_cache[file_path] = (x_all, y_all)
                self._file_cache.move_to_end(file_path)
                while len(self._file_cache) > self.cache_size:
                    self._file_cache.popitem(last=False)
            x, y = x_all[ei], int(y_all[ei])
        else:
            x_all, y_all = cached
            self._file_cache.move_to_end(file_path)
            x, y = x_all[ei], int(y_all[ei])
        if self.transform:
            x = self.transform(x)
        # return (channels, samples) for conv1d: add channel dim
        x = np.expand_dims(x, 0)
        return x, y

    def subjects(self):
        """Return list of subject basenames in the order discovered."""
        return [os.path.splitext(os.path.basename(p))[0] for p in self.files]


class SleepEDFSubjectIterableDataset(IterableDataset):
    """Iterable dataset that yields epoch samples by loading each subject once per worker.

    This minimizes per-sample file I/O: each worker is assigned a subset of subject files
    and iterates over all epochs for each subject (optionally shuffling epochs).
    """

    def __init__(self, npz_dir=None, files=None, transform=None, shuffle=True, seed=42):
        super().__init__()
        self.npy_dir = npz_dir
        if files is not None:
            self.files = sorted(files)
        elif npz_dir is not None:
            self.files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
        else:
            self.files = []
        self.transform = transform
        self.shuffle = bool(shuffle)
        self.seed = int(seed) if seed is not None else None

    def _files_for_worker(self, worker_id, num_workers):
        # deterministic partitioning of files across workers
        if num_workers <= 1:
            return list(self.files)
        # simple round-robin partition
        return [f for i, f in enumerate(self.files) if (i % num_workers) == worker_id]

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        files = self._files_for_worker(worker_id, num_workers)
        rng = np.random.default_rng(self.seed + worker_id if self.seed is not None else None)

        for fpath in files:
            try:
                with np.load(fpath) as d:
                    x_all = np.asarray(d['x'], dtype=np.float32)
                    y_all = np.asarray(d['y'], dtype=np.int64)
            except Exception:
                continue
            n = x_all.shape[0]
            idxs = np.arange(n)
            if self.shuffle:
                rng.shuffle(idxs)
            for ei in idxs:
                x = x_all[ei]
                y = int(y_all[ei])
                if self.transform:
                    x = self.transform(x)
                x = np.expand_dims(x, 0)
                yield x, y
