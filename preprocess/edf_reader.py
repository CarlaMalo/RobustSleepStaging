"""Minimal EDF+ reader for Sleep-EDF annotation files.

This is a Python 3–safe adaptation of the tiny reader used in DeepSleepNet/
AttnSleep preprocessing. It is intentionally small and only implements the
subset we need for Sleep-EDF PSG + hypnogram pairs.
"""

from __future__ import annotations

import datetime as _dt
import logging
import operator
import re
from collections import namedtuple
from functools import reduce

import numpy as np

EVENT_CHANNEL = "EDF Annotations"
log = logging.getLogger(__name__)


class EDFEndOfData(Exception):
    pass


def tal(tal_bytes: bytes):
    """Parse a TAL (Time-stamped Annotation List) byte stream.

    Returns a list of (onset, duration, annotation_list) tuples.
    """

    tal_str = tal_bytes.decode("latin-1", errors="ignore")
    exp = (
        r"(?P<onset>[+\-]\d+(?:\.\d*)?)"
        r"(?:\x15(?P<duration>\d+(?:\.\d*)?))?"
        r"(?:\x14(?P<annotation>[^\x00]*))?"
        r"(?:\x14\x00)"
    )

    def annotation_to_list(annotation):
        return annotation.split("\x14") if annotation else []

    def parse(dic):
        return (
            float(dic["onset"]),
            float(dic["duration"]) if dic["duration"] else 0.0,
            annotation_to_list(dic["annotation"]),
        )

    return [parse(m.groupdict()) for m in re.finditer(exp, tal_str)]


def _read_str(f, nbytes):
    return f.read(nbytes).decode("ascii", errors="ignore").strip()


def edf_header(f):
    """Read EDF/EDF+ fixed header fields from a binary file object."""

    h = {}
    assert f.tell() == 0
    version = _read_str(f, 8)
    if version != "0":
        # EDF files are usually padded as '0       '.
        version = version.strip()

    h["local_subject_id"] = _read_str(f, 80)
    h["local_recording_id"] = _read_str(f, 80)

    date_bytes = _read_str(f, 8)
    time_bytes = _read_str(f, 8)
    day, month, year = [int(x) for x in re.findall(r"(\d+)", date_bytes)]
    hour, minute, sec = [int(x) for x in re.findall(r"(\d+)", time_bytes)]
    h["date_time"] = str(_dt.datetime(year + 2000, month, day, hour, minute, sec))

    header_nbytes = int(_read_str(f, 8))
    subtype = _read_str(f, 44)[:5]
    h["EDF+"] = subtype in ["EDF+C", "EDF+D"]
    h["contiguous"] = subtype != "EDF+D"
    h["n_records"] = int(_read_str(f, 8))
    h["record_length"] = float(_read_str(f, 8))
    nchannels = h["n_channels"] = int(_read_str(f, 4))

    channels = range(h["n_channels"])
    h["label"] = [_read_str(f, 16) for _ in channels]
    h["transducer_type"] = [_read_str(f, 80) for _ in channels]
    h["units"] = [_read_str(f, 8) for _ in channels]
    h["physical_min"] = np.asarray([float(_read_str(f, 8)) for _ in channels])
    h["physical_max"] = np.asarray([float(_read_str(f, 8)) for _ in channels])
    h["digital_min"] = np.asarray([float(_read_str(f, 8)) for _ in channels])
    h["digital_max"] = np.asarray([float(_read_str(f, 8)) for _ in channels])
    h["prefiltering"] = [_read_str(f, 80) for _ in channels]
    h["n_samples_per_record"] = [int(_read_str(f, 8)) for _ in channels]
    f.read(32 * nchannels)

    # EDF headers are padded to a fixed length; we do not strictly enforce it.
    _ = header_nbytes
    return h


class BaseEDFReader:
    def __init__(self, file_obj):
        self.file = file_obj

    def read_header(self):
        self.header = h = edf_header(self.file)

        self.dig_min = h["digital_min"]
        self.phys_min = h["physical_min"]
        phys_range = h["physical_max"] - h["physical_min"]
        dig_range = h["digital_max"] - h["digital_min"]
        assert np.all(phys_range > 0)
        assert np.all(dig_range > 0)
        self.gain = phys_range / dig_range

    def read_raw_record(self):
        result = []
        for nsamp in self.header["n_samples_per_record"]:
            samples = self.file.read(nsamp * 2)
            if len(samples) != nsamp * 2:
                raise EDFEndOfData
            result.append(samples)
        return result

    def convert_record(self, raw_record):
        h = self.header
        dig_min, phys_min, gain = self.dig_min, self.phys_min, self.gain
        time = float("nan")
        signals = []
        events = []
        for i, samples in enumerate(raw_record):
            if h["label"][i] == EVENT_CHANNEL:
                ann = tal(samples)
                if ann:
                    time = ann[0][0]
                    events.extend(ann)
            else:
                dig = np.frombuffer(samples, dtype="<i2").astype(np.float32)
                phys = (dig - dig_min[i]) * gain[i] + phys_min[i]
                signals.append(phys)
        return time, signals, events

    def read_record(self):
        return self.convert_record(self.read_raw_record())

    def records(self):
        try:
            while True:
                yield self.read_record()
        except EDFEndOfData:
            return


def load_edf(edffile):
    """Load an EDF/EDF+ file and return a simple tuple-like object."""

    if isinstance(edffile, (str, bytes)):
        with open(edffile, "rb") as f:
            return load_edf(f)

    reader = BaseEDFReader(edffile)
    reader.read_header()
    h = reader.header
    log.debug("EDF header: %s", h)

    nsamp = np.unique([n for (l, n) in zip(h["label"], h["n_samples_per_record"]) if l != EVENT_CHANNEL])
    assert nsamp.size == 1, "Multiple sample rates not supported"
    sample_rate = float(nsamp[0]) / h["record_length"]

    rectime, X, annotations = zip(*reader.records())
    X = np.hstack(X)
    annotations = reduce(operator.add, annotations)
    chan_lab = [lab for lab in reader.header["label"] if lab != EVENT_CHANNEL]

    if reader.header["contiguous"]:
        time = np.arange(X.shape[1]) / sample_rate
    else:
        reclen = reader.header["record_length"]
        within_rec_time = np.linspace(0, reclen, nsamp, endpoint=False)
        time = np.hstack([t + within_rec_time for t in rectime])

    tup = namedtuple("EDF", "X sample_rate chan_lab time annotations")
    return tup(X, sample_rate, chan_lab, time, annotations)
