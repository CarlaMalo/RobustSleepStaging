"""Data preprocessing and loading for RobustSleepStaging.

Keep package imports lightweight so training startup does not eagerly import
the EDF preparation pipeline (which pulls in heavy optional dependencies).
"""

from .dataset import SleepEDFNPZDataset

__all__ = ["SleepEDFNPZDataset", "load_edf", "BaseEDFReader", "prepare_sleep_edf"]


def __getattr__(name):
    if name in {"load_edf", "BaseEDFReader"}:
        from .edf_reader import BaseEDFReader, load_edf

        return {"load_edf": load_edf, "BaseEDFReader": BaseEDFReader}[name]
    if name == "prepare_sleep_edf":
        from .prepare import prepare_sleep_edf

        return prepare_sleep_edf
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
