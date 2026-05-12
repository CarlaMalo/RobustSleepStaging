

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
    psds, freqs = psd_array_multitaper(
        X, sfreq=sfreq, fmin=fmin, fmax=fmax,
        adaptive=False, verbose=False
    )
    # Flatten channels x frequencies into single feature vector
    return psds.reshape(psds.shape[0], -1)


print('✓ Functions defined')
print(f'  Common channels: {COMMON_CHANNELS}')
print('  Preprocessing: bandpass filter 0.5-40 Hz, 30s epochs')
print('  Features: Power Spectral Density (PSD)')