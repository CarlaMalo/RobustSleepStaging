STAGE_MAP = {
    'Sleep stage W': 0,  # Wake
    'Sleep stage 1': 1,  # NREM 1
    'Sleep stage 2': 2,  # NREM 2
    'Sleep stage 3': 3,  # NREM 3
    'Sleep stage 4': 3,  # NREM 4 -> Merged with NREM 3
    'Sleep stage R': 5,  # REM
}
STAGE_NAMES = {0:'Wake', 1:'N1', 2:'N2', 3:'N3', 4:'N3', 5:'REM'}

# Channels available in both SC and ST cohorts
# As specified in project description: EEG (Fpz-Cz, Pz-Oz), EOG, EMG
# TODO 
COMMON_CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental']