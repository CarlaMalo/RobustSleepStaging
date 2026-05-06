RobustSleepStaging/
├── README.md                          
├── environment-cuda121.yml            # Conda env for GPU (CUDA 12.1)
├── environment-cpu.yml                # Conda env for CPU
├── __init__.py                        # Top-level package init
│
├── preprocess_sleep_edf.py            # Entry script: EDF → NPZ preprocessing
├── train_model.py                     # Entry script: train
├── evaluate_model.py                  # Entry script: evaluate
│
├── preprocess/                        # Data preprocessing module
│   ├── __init__.py
│   ├── prepare.py                     # Main EDF → NPZ converter
│   ├── edf_reader.py                  # Lightweight EDF+ parser
│   └── dataset.py                     # SleepEDFNPZDataset PyTorch dataloader
│
├── models/                            # Model definitions module
│   ├── __init__.py
│   └── baseline.py                    # Conv1DBaseline (1D CNN)
│
├── train/                             # Training logic module
│   ├── __init__.py
│   └── trainer.py                     # Training loop + subject-level splits
│
├── eval/                              # Evaluation and metrics module
│   ├── __init__.py
│   ├── metrics.py                     # overall_metrics, ECE, transition_error
│   └── evaluator.py                   # Evaluation script with CLI
│
└── data/                              # Typical data folder (not included)
    └── physionet.org/...              # Downloaded Sleep-EDF data
    └── sleep_edf_npz/                 # Output from preprocess_sleep_edf.py
        ├── manifest.csv
        ├── sleep-cassette/            # Per-subject .npz files
        └── sleep-telemetry/           # Per-subject .npz files
