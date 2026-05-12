import numpy as np
import mne


def load_subject_epochs(psg_path, hyp_path, common_channels, l_freq=0.5, h_freq=40.0, epoch_sec=30.0, verbose=False):
    """Load raw EDF + hypnogram and return epoched data and labels.

    Returns:
        X: np.array (n_epochs, n_channels, n_timepoints)
        y: np.array (n_epochs,) integer labels
        sfreq: sampling frequency
    """
    if verbose:
        print(f"[preprocess] loading PSG: {psg_path.name}, hypnogram: {hyp_path.name}")
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    available = [ch for ch in common_channels if ch in raw.ch_names]
    if not available:
        raise ValueError(f"None of requested channels present in {psg_path.name}")
    raw.pick_channels(available)
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose=False)

    annotations = mne.read_annotations(hyp_path)
    raw.set_annotations(annotations)
    if verbose:
        print(f"[preprocess] channels picked: {available}, sampling freq: {raw.info['sfreq']} Hz")

    KEEP_STAGES = list({k for k in annotations.description if True})
    # The notebook passes a mapping externally for label mapping; here we just
    # create epochs aligned to annotations and return the event ids as labels.
    events, event_id = mne.events_from_annotations(raw, chunk_duration=epoch_sec, verbose=False)
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0.0, tmax=epoch_sec - 1/raw.info['sfreq'], baseline=None, preload=True, verbose=False)

    X = epochs.get_data()
    y = epochs.events[:, 2]
    if verbose:
        print(f"[preprocess] extracted epochs: {X.shape[0]} epochs, shape per epoch: {X.shape[1:]} ")
    return X, y, raw.info['sfreq']
