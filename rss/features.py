import numpy as np
from mne.time_frequency import psd_array_multitaper
import time


def extract_psd_features(X_epochs, sfreq, fmin=0.5, fmax=40.0, adaptive=False, verbose=False):
    """Compute PSD per epoch and flatten channel×freq into feature vector.

    X_epochs: (n_epochs, n_channels, n_timepoints)
    returns: (n_epochs, n_channels * n_freqs)
    """
    if verbose:
        print(f"[features] computing PSD for {X_epochs.shape[0]} epochs (sfreq={sfreq})")
        t0 = time.time()
    psds, freqs = psd_array_multitaper(X_epochs, sfreq=sfreq, fmin=fmin, fmax=fmax, adaptive=adaptive, verbose=False)
    if verbose:
        dt = time.time() - t0
        print(f"[features] PSD computed — elapsed: {dt:.1f}s, psd shape: {psds.shape}")
    return psds.reshape(psds.shape[0], -1), freqs

def bin_psd(psds, freqs, bin_width=0.5, fmin=0.5, fmax=30.0, mode="mean"):
    """
    Reduce PSD frequency resolution by aggregating neighboring frequency bins.

    psds: shape (n_epochs, n_channels, n_freqs)
    freqs: shape (n_freqs,)
    returns:
        binned_psds: shape (n_epochs, n_channels, n_bins)
        bin_centers: shape (n_bins,)
    """
    bin_edges = np.arange(fmin, fmax + bin_width, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    binned = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (freqs >= lo) & (freqs < hi)

        if not np.any(mask):
            continue

        if mode == "mean":
            # Average PSD density within the bin
            val = psds[:, :, mask].mean(axis=-1)

        elif mode == "power":
            # Integrated power within the bin
            val = np.trapezoid(psds[:, :, mask], freqs[mask], axis=-1)

        else:
            raise ValueError("mode must be 'mean' or 'power'")

        binned.append(val)

    binned_psds = np.stack(binned, axis=-1)

    return binned_psds, bin_centers[:binned_psds.shape[-1]]


def extract_emg_features(X_emg_epochs):
    """
    Extract features from preprocessed submental EMG envelope.

    X_emg_epochs: shape (n_epochs, n_samples)
                  for 30 s epochs at 1 Hz, n_samples should be ~30

    returns: shape (n_epochs, n_features)
    """
    q25 = np.percentile(X_emg_epochs, 25, axis=1)
    q50 = np.percentile(X_emg_epochs, 50, axis=1)
    q75 = np.percentile(X_emg_epochs, 75, axis=1)
    q90 = np.percentile(X_emg_epochs, 90, axis=1)
    q95 = np.percentile(X_emg_epochs, 95, axis=1)

    features = np.column_stack([
        np.mean(X_emg_epochs, axis=1),
        q50,
        np.std(X_emg_epochs, axis=1),
        np.min(X_emg_epochs, axis=1),
        np.max(X_emg_epochs, axis=1),
        q75 - q25,
        q90,
        q95,
        np.sqrt(np.mean(X_emg_epochs ** 2, axis=1)),
    ])

    feature_names = [
        "emg_mean",
        "emg_median",
        "emg_std",
        "emg_min",
        "emg_max",
        "emg_iqr",
        "emg_p90",
        "emg_p95",
        "emg_rms",
    ]

    return features, feature_names


def save_recording_checkpoint(path, X_features, y_labels, feature_names=None, subject_ids=None, recording_id=None):
    """Save per-recording feature checkpoint (.npz)"""
    np.savez(path, X_features=X_features, y_labels=y_labels, feature_names=feature_names if feature_names is not None else np.array([]), subject_ids=subject_ids if subject_ids is not None else np.array([]), recording_id=recording_id)
