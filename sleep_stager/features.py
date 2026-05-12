
import mne
import numpy as np


def extract_psd_features(X, sfreq=100.0, fmin=0.5, fmax=40.0):
    """
    Extract Power Spectral Density (PSD) features from epoched EEG data.
    
    PSD captures frequency-domain information relevant to sleep staging:
    - Delta (0.5-4 Hz): deep sleep (N3/N4)
    - Theta (4-8 Hz): light sleep (N1)
    - Alpha (8-13 Hz): relaxed wakefulness
    - Sigma (12-16 Hz): sleep spindles (N2)
    - Beta (13-30 Hz): active wakefulness / REM
    
    Returns:
        Feature matrix of shape (n_epochs, n_channels * n_freqs)
    """
    psds, freqs = mne.time_frequency.psd_array_multitaper(
        X, sfreq=sfreq, fmin=fmin, fmax=fmax,
        adaptive=False, verbose=False
    )
    return psds, freqs


# def extract_peak_to_peak(X):
#     return np.max(X, axis=-1) - np.min(X, axis=-1)


# def extract_eeg_features(X):
#     """
#     Extract EEG-specific features such as:
#     - delta power: 0.5–4 Hz
#     - theta power: 4–8 Hz
#     - alpha power: 8–12 Hz
#     - sigma power: 12–16 Hz
#     - beta power: 16–30 Hz
#     - peak-to-peak amplitude

#     Returns:
#         Feature matrix of shape (n_epochs, n_features)
#     """
#     delta_power, freqs = extract_psd_features(X, fmin=0.5, fmax=4.0)
#     theta_power, _ = extract_psd_features(X, fmin=4.0, fmax=8.0)
#     alpha_power, _ = extract_psd_features(X, fmin=8.0, fmax=12.0)
#     sigma_power, _ = extract_psd_features(X, fmin=12.0, fmax=16.0)
#     beta_power, _ = extract_psd_features(X, fmin=16.0, fmax=30.0)
#     peak_to_peak = extract_peak_to_peak(X)

#     features = np.concatenate([
#         delta_power, theta_power, alpha_power, sigma_power, beta_power, peak_to_peak
#     ], axis=-1)
#     return features

# def extract_eog_features(X):
#     """
#     Extract EOG-specific features such as:
#     - power in 0.5–2 Hz (slow eye movements)
#     - power in 2–5 Hz (rapid eye movements)
#     - root mean square amplitude
#     Returns:
#         Feature matrix of shape (n_epochs, n_features)
#     """
#     slow_eye_power, freqs = extract_psd_features(X, fmin=0.5, fmax=2.0)
#     rapid_eye_power, _ = extract_psd_features(X, fmin=2.0, fmax=5.0)
#     rms_amplitude = np.sqrt(np.mean(X**2, axis=-1))

#     features = np.concatenate([
#         slow_eye_power, rapid_eye_power, rms_amplitude
#     ], axis=-1)
#     return features

# def extract_emg_features(X):
#     """
#     Extract EMG-specific features such as:
#     - power in 10–30 Hz (muscle activity)
#     - line length (signal complexity)
#     - root mean square amplitude
#     Returns:
#         Feature matrix of shape (n_epochs, n_features)
#     """
#     muscle_tone_power, freqs = extract_psd_features(X, fmin=10.0, fmax=30.0)
#     line_length = np.sum(np.abs(np.diff(X, axis=-1)), axis=-1)
#     rms_amplitude = np.sqrt(np.mean(X**2, axis=-1))

#     features = np.concatenate([
#         muscle_tone_power, line_length, rms_amplitude
#     ], axis=-1)
#     return features

# def extract_features(X, channels):
#     eeg_channels = [ch for ch in channels if "EEG" in ch]
#     eog_channels = [ch for ch in channels if "EOG" in ch]
#     emg_channels = [ch for ch in channels if "EMG" in ch]

#     eeg_features = extract_eeg_features(X[:, eeg_channels])
#     eog_features = extract_eog_features(X[:, eog_channels])
#     emg_features = extract_emg_features(X[:, emg_channels])

#     return np.concatenate([eeg_features, eog_features, emg_features], axis=-1)