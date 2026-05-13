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


def save_recording_checkpoint(path, X_features, y_labels, subject_ids=None, recording_id=None):
    """Save per-recording feature checkpoint (.npz)"""
    np.savez(path, X_features=X_features, y_labels=y_labels, subject_ids=subject_ids if subject_ids is not None else np.array([]), recording_id=recording_id)
