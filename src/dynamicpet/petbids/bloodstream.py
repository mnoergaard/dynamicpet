from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from .petbidsjson import read_json
from ..temporalobject.temporalmatrix import TemporalMatrix


def load_inputfunction(filename: str | Path) -> Tuple[TemporalMatrix, TemporalMatrix]:
    """Load arterial input function from a bloodstream derivative TSV.

    Parameters
    ----------
    filename : str or Path
        Path to ``*_inputfunction.tsv`` produced by the ``bloodstream`` tool.

    Returns
    -------
    Tuple[TemporalMatrix, TemporalMatrix]
        ``TemporalMatrix`` instances for the metabolite corrected plasma
        activity (``AIF`` column) and whole blood activity
        (``whole_blood_radioactivity`` column). Times are returned in minutes.
    """
    fname = Path(filename)
    jsonfile = fname.with_suffix(".json")
    if jsonfile.exists():
        _ = read_json(jsonfile)  # read for completeness, but not used currently

    # get column indices from header
    with fname.open() as f:
        header = f.readline().strip().split("\t")
    header_lower = [h.lower() for h in header]
    time_idx = header_lower.index("time")
    aif_idx = header_lower.index("aif")
    wb_idx = header_lower.index("whole_blood_radioactivity")

    data = np.genfromtxt(fname, delimiter="\t", skip_header=1)
    times_sec = data[:, time_idx].astype(float)
    aif = data[:, aif_idx].astype(float)
    wb = data[:, wb_idx].astype(float)

    frame_start = times_sec / 60.0
    if len(frame_start) > 1:
        frame_duration = np.diff(
            frame_start, append=frame_start[-1] + (frame_start[-1] - frame_start[-2])
        )
    else:
        frame_duration = np.array([1.0])

    aif_tm = TemporalMatrix(aif, frame_start, frame_duration, [header[aif_idx]])
    wb_tm = TemporalMatrix(wb, frame_start, frame_duration, [header[wb_idx]])
    return aif_tm, wb_tm
