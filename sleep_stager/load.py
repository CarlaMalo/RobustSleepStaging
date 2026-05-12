from pathlib import Path

import mne
import scipy

from sleep_stager.enum import COMMON_CHANNELS, COMMON_CHANNELS, STAGE_MAP


def find_psg_hypnogram_pairs(data_dir: str):
    """
    Finds pairs of PSG and hypnogram files in the given directory. Assumes PSG files are named like 'XXXXXX-PSG.edf' and hypnogram files are named like 'XXXXXX-Hypnogram.edf', where 'XXXXXX' is the subject ID.
    Args:
        data_dir (str): The directory to search for PSG and hypnogram files.
    Returns:
        List[Tuple[Path, Path]]: A list of tuples containing pairs of PSG and hypnogram file paths.
        
    """
    data_dir = Path(data_dir)

    psgs = sorted([f for f in data_dir.glob('*-PSG.edf')])
    pairs = []
    print(psgs)
    for psg in psgs:
        subject_id = psg.name[:6]
        hyp_files = list(data_dir.glob(f'{subject_id}*-Hypnogram.edf'))
        if hyp_files:
            pairs.append((psg, hyp_files[0]))
    return pairs


def load_subject(psg_path, hyp_path, do_bandpass=True):
    """
    Load one PSG recording and its hypnogram.

    Returns:
        X: np.array of shape (n_epochs, n_channels, n_timepoints)
        y: np.array of shape (n_epochs,) with sleep stage labels
    """
    psg_path = Path(psg_path)
    hyp_path = Path(hyp_path)

    keep_stages = list(STAGE_MAP.keys())

    raw = mne.io.read_raw_edf(
        psg_path,
        preload=True,
        verbose="ERROR",
    )

    available_channels = [ch for ch in COMMON_CHANNELS if ch in raw.ch_names]
    print(f"Available channels in {psg_path.name}: {available_channels}")

    if len(available_channels) == 0:
        raise ValueError(f"No common channels found in {psg_path.name}")

    raw.pick(available_channels)

    if do_bandpass:
        # only for EEG channels:
        eeg_channels = [ch for ch in available_channels if "EEG" in ch]
        eog_channels = [ch for ch in available_channels if "EOG" in ch]
        emg_channels = [ch for ch in available_channels if "EMG" in ch]

        eeg_picks = mne.pick_channels(raw.ch_names, eeg_channels)
        eog_picks = mne.pick_channels(raw.ch_names, eog_channels)
        emg_picks = mne.pick_channels(raw.ch_names, emg_channels)

        if len(eeg_picks) > 0:
            raw.filter(
                l_freq=0.3,
                h_freq=35.0,
                picks=eeg_picks,
                fir_design="firwin",
                verbose="ERROR",
            )

        if len(eog_picks) > 0:
            raw.filter(
                l_freq=0.3,
                h_freq=35.0,
                picks=eog_picks,
                fir_design="firwin",
                verbose="ERROR",
            )

        # if len(emg_picks) > 0:
        #     raw.filter(
        #         l_freq=10.0,
        #         h_freq=40.0,
        #         picks=emg_picks,
        #         fir_design="firwin",
        #         verbose="ERROR",
        #     )

    annotations = mne.read_annotations(hyp_path)
    raw.set_annotations(annotations)

    available_stage_labels = set(raw.annotations.description)
    used_stage_map = {
        stage: code
        for stage, code in STAGE_MAP.items()
        if stage in available_stage_labels
    }

    if len(used_stage_map) == 0:
        raise ValueError(
            f"No matching sleep stages found in {hyp_path.name}. "
            f"Available labels: {sorted(available_stage_labels)}"
        )

    events, event_id = mne.events_from_annotations(
        raw,
        event_id=used_stage_map,
        chunk_duration=30.0,
        verbose="ERROR",
    )

    sfreq = raw.info["sfreq"]

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=0.0,
        tmax=30.0 - 1 / sfreq,
        baseline=None,
        preload=True,
        reject_by_annotation=False,
        verbose="ERROR",
    )

    X = epochs.get_data()
    y = epochs.events[:, 2]

    return X, y, available_channels


def compute_welch_psd(
    X,
    sfreq,
    fmin=0.5,
    fmax=40.0,
    nperseg=None,
    noverlap=None,
):
    """
    Compute PSD from epoched NumPy data.

    Args:
        X: array, shape (n_epochs, n_channels, n_timepoints)
        sfreq: sampling frequency in Hz
        fmin: lower frequency bound
        fmax: upper frequency bound
        nperseg: Welch window length in samples
        noverlap: overlap between Welch windows in samples

    Returns:
        psds: array, shape (n_epochs, n_channels, n_frequencies)
        freqs: array, shape (n_frequencies,)
    """
    if nperseg is None:
        nperseg = int(4 * sfreq)   # 4-second windows

    if noverlap is None:
        noverlap = int(2 * sfreq)  # 50% overlap

    freqs, psds = scipy.signal.welch(
        X,
        fs=sfreq,
        nperseg=nperseg,
        noverlap=noverlap,
        axis=-1,
    )

    freq_mask = (freqs >= fmin) & (freqs <= fmax)

    freqs = freqs[freq_mask]
    psds = psds[..., freq_mask]

    return psds, freqs