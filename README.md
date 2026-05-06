# RobustSleepStaging 

This repository explores robust sleep staging under distribution shift for the Sleep-EDF dataset. The pipeline currently trains the baseline (1D CNN ) on Sleep-Cassette, and evaluates it's generalization on Sleep-Telemetry.

Quick notes:
- Preprocess EDF files into per-subject `.npz` with the bundled command.
- Place or keep cassette and telemetry outputs in separate split folders.
- Train with the cassette split and evaluate on telemetry.


## Setting up the Environment:

Run the following command (using conda) to setup the environment, depending on your hardware.

```
conda env create -f environment-cpu.yml
``` 

- `environment-cuda121.yml` : CUDA 12.1 GPU environment
- `environment-cpu.yml` : CPU-friendly environment

## Commands:

1. Run the following to preprocess the data (only once):

```
python preprocess_sleep_edf.py --data_root data/physionet.org/files/sleep-edfx/1.0.0 --output_dir data/sleep_edf_npz --splits all
```

2. Run the following command to train:

```
python train_model.py --train_dir data/sleep_edf_npz/sleep-cassette --val_fraction 0.2 --epochs 20 --batch_size 64 --checkpoint checkpoints/model_best.pth
```

3. Run the following command to evaluate:

```
python evaluate_model.py --test_dir data/sleep_edf_npz/sleep-telemetry --checkpoint checkpoints/model_best.pth
```