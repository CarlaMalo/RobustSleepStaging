import numpy as np
from pathlib import Path


def load_per_record_checkpoints(checkpoint_dir, load_mode="ram"):
    """Load all .npz checkpoint files from a directory.

    Args:
        checkpoint_dir: Directory containing per-recording ``.npz`` files.
        load_mode: ``"ram"`` to concatenate into regular NumPy arrays,
            or ``"disk"`` to build/load a disk-backed memmap aggregate.

    Returns:
        X: ndarray (n_epochs, n_features)
        y: ndarray (n_epochs,)
        sids: ndarray (n_epochs,) subject id strings when available
    """
    checkpoint_dir = Path(checkpoint_dir)
    files = sorted(checkpoint_dir.glob("*.npz"))
    if not files:
        return np.empty((0, 0)), np.empty((0,), dtype=int), np.empty((0,), dtype=object)

    load_mode = str(load_mode).lower().strip()
    if load_mode not in {"ram", "disk"}:
        raise ValueError("load_mode must be either 'ram' or 'disk'")

    if load_mode == "disk":
        # Build or load a disk-backed aggregate so the full dataset does not
        # need to be materialized in RAM at once.
        print(f"[utils] load_mode=disk — scanning checkpoint directory: {checkpoint_dir}")
        print(f"[utils] found {len(files)} .npz files")
        agg_dir = checkpoint_dir / "aggregated"
        agg_dir.mkdir(parents=True, exist_ok=True)
        x_path = agg_dir / "X_features.dat"
        y_path = agg_dir / "y_labels.npy"
        sid_path = agg_dir / "subject_ids.npy"

        total_epochs = 0
        feat_dim = None
        y_dtype = None
        sid_dtype = None
        valid_files = []

        for i, f in enumerate(files):
            try:
                with np.load(f, allow_pickle=True) as d:
                    if "X_features" not in d or "y_labels" not in d:
                        continue
                    Xf = d["X_features"]
                    yf = d["y_labels"]
                    total_epochs += int(Xf.shape[0])
                    if feat_dim is None:
                        feat_dim = int(Xf.shape[1])
                        y_dtype = yf.dtype
                        sid_dtype = d["subject_ids"].dtype if "subject_ids" in d else np.dtype("U16")
                    valid_files.append(f)
            except Exception:
                # report bad file but continue
                print(f"[utils] warning: failed to read metadata from {f.name}")
                continue
            if (i + 1) % 50 == 0:
                print(f"[utils] scanned {i+1}/{len(files)} files — total_epochs so far: {total_epochs}")

        if total_epochs == 0 or feat_dim is None:
            return np.empty((0, 0)), np.empty((0,), dtype=int), np.empty((0,), dtype=object)

        if x_path.exists() and y_path.exists() and sid_path.exists():
            try:
                y_loaded = np.load(y_path, mmap_mode="r")
                if y_loaded.shape[0] == total_epochs:
                    X_all = np.memmap(x_path, dtype="float32", mode="r", shape=(total_epochs, feat_dim))
                    sid_all = np.load(sid_path, mmap_mode="r")
                    return X_all, y_loaded, sid_all
            except Exception:
                pass

        print(f"[utils] creating memmap array: {total_epochs} epochs x {feat_dim} features -> {x_path}")
        X_mem = np.memmap(x_path, dtype="float32", mode="w+", shape=(total_epochs, feat_dim))
        y_arr = np.empty((total_epochs,), dtype=y_dtype if y_dtype is not None else np.int16)
        sid_arr = np.empty((total_epochs,), dtype=sid_dtype if sid_dtype is not None else np.dtype("U16"))

        pos = 0
        for i, f in enumerate(valid_files):
            try:
                with np.load(f, allow_pickle=True) as d:
                    Xf = d["X_features"].astype(np.float32, copy=False)
                    yf = d["y_labels"]
                    sf = d.get("subject_ids", np.array([""] * Xf.shape[0], dtype=sid_arr.dtype))
                    n = Xf.shape[0]
                    X_mem[pos : pos + n] = Xf
                    y_arr[pos : pos + n] = yf
                    sid_arr[pos : pos + n] = sf
                    pos += n
            except Exception:
                print(f"[utils] warning: failed to load data from {f.name}")
                continue
            if (i + 1) % 50 == 0:
                print(f"[utils] filled from {i+1}/{len(valid_files)} files — epochs written: {pos}")

        X_mem.flush()
        np.save(y_path, y_arr)
        np.save(sid_path, sid_arr)
        print(f"[utils] finished memmap fill — total epochs written: {pos}")
        X_all = np.memmap(x_path, dtype="float32", mode="r", shape=(total_epochs, feat_dim))
        y_all = np.load(y_path, mmap_mode="r")
        sid_all = np.load(sid_path, mmap_mode="r")
        return X_all, y_all, sid_all

    parts = []
    sids = []
    y_parts = []
    print(f"[utils] load_mode=ram — loading {len(files)} .npz files into memory")
    for i, f in enumerate(files):
        try:
            with np.load(f, allow_pickle=True) as d:
                if 'X_features' in d and 'y_labels' in d:
                    Xf = d['X_features'].astype('float32')
                    parts.append(Xf)
                    y_parts.append(d['y_labels'])
                    sids.append(d.get('subject_ids', np.array([''] * Xf.shape[0])))
        except Exception:
            print(f"[utils] warning: failed to load {f.name}")
            continue
        if (i + 1) % 50 == 0:
            print(f"[utils] loaded {i+1}/{len(files)} files into lists")

    if not parts:
        return np.empty((0, 0)), np.empty((0,), dtype=int), np.empty((0,), dtype=object)

    X = np.concatenate(parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    sids = np.concatenate(sids, axis=0)
    print(f"[utils] concatenated into arrays — X: {X.shape}, y: {y.shape}")
    return X, y, sids
