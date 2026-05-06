"""RobustSleepStaging: A modular pipeline for Sleep-EDF training and evaluation under distribution shift."""

__version__ = "0.1"

from preprocess import SleepEDFNPZDataset, prepare_sleep_edf
from models import Conv1DBaseline
from train import split_subject_files
from eval import overall_metrics, expected_calibration_error, transition_error

__all__ = [
    "SleepEDFNPZDataset",
    "prepare_sleep_edf",
    "Conv1DBaseline",
    "split_subject_files",
    "overall_metrics",
    "expected_calibration_error",
    "transition_error",
]
