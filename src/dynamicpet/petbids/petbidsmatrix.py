"""PETBIDSMatrix class."""

from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from dynamicpet.petbids.petbidsjson import (
    PetBidsJson,
    get_frametiming_in_mins,
    read_json,
    update_frametiming_from,
    write_json,
)
from dynamicpet.temporalobject.temporalmatrix import TemporalMatrix

from .petbidsobject import PETBIDSObject

if TYPE_CHECKING:
    from os import PathLike

    from dynamicpet.typing_utils import NumpyNumberArray, RealNumber


class PETBIDSMatrix(TemporalMatrix, PETBIDSObject):
    """4-D image data with corresponding PET-BIDS time frame information.

    Args:
        dataobj: vector or k x num_frames matrix
        json_dict: PET-BIDS json dictionary
        elem_names: list of k ROI names

    Attributes:
        _dataobj: 1 x num_frames vector or k x num_frames matrix
        frame_start: vector containing the start times of each frame, in min
        frame_duration: vector containing the durations of each frame, in min
        json_dict: PET-BIDS json dictionary
        elem_names: list of k element names

    """

    def __init__(
        self,
        dataobj: NumpyNumberArray,
        json_dict: PetBidsJson,
        elem_names: list[str] | None = None,
    ) -> None:
        """Matrix with corresponding PET-BIDS time frame information.

        Args:
            dataobj: vector or k x num_frames matrix
            json_dict: PET-BIDS json dictionary
            elem_names: list of k ROI names

        """
        frame_start, frame_duration = get_frametiming_in_mins(json_dict)

        super().__init__(dataobj, frame_start, frame_duration, elem_names)

        # need to make a copy of json_dict before storing
        self.json_dict: PetBidsJson = deepcopy(json_dict)

    def extract(self, start_time: RealNumber, end_time: RealNumber) -> PETBIDSMatrix:
        """Extract a temporally shorter PETBIDSMatrix from a PETBIDSMatrix.

        Args:
            start_time: time (min) at which to begin relative to TimeZero, incl.
            end_time: time (min) at which to stop relative to TimeZero, incl.

        Returns:
            extracted PETBIDSMatrix

        """
        extracted_matrix = super().extract(start_time, end_time)
        json_dict = update_frametiming_from(self.json_dict, extracted_matrix)

        return PETBIDSMatrix(extracted_matrix.dataobj, json_dict)

    def concatenate(self, other: PETBIDSMatrix) -> PETBIDSMatrix:  # type: ignore[override]
        """Concatenate another PETBIDSMatrix at the end (in time).

        Args:
            other: PETBIDSMatrix to concatenate

        Returns:
            concatenated PETBIDSMatrix

        """
        newdecaycorrecttime, original_anchor = self._decay_correct_offset(other)
        other = other.decay_correct(decaycorrecttime=newdecaycorrecttime)

        concat_mat = super().concatenate(other)
        json_dict = update_frametiming_from(self.json_dict, concat_mat)

        concat_res = PETBIDSMatrix(concat_mat.dataobj, json_dict)
        concat_res.set_timezero(anchor=original_anchor)

        return concat_res

    def decay_correct(self, decaycorrecttime: float = 0) -> PETBIDSMatrix:
        """Return decay corrected PETBIDSMatrix.

        Args:
            decaycorrecttime: time to decay correct to, relative to time zero

        Returns:
            decay corrected TACs

        """
        tacs = self.get_decay_corrected_tacs(decaycorrecttime)
        corrected_tacs = np.reshape(tacs, self.shape)

        json_dict = deepcopy(self.json_dict)
        json_dict["ImageDecayCorrected"] = True
        json_dict["ImageDecayCorrectionTime"] = (
            decaycorrecttime + json_dict["ScanStart"] + json_dict["InjectionStart"]
        )

        return PETBIDSMatrix(corrected_tacs, json_dict)

    def decay_uncorrect(self) -> PETBIDSMatrix:
        """Return decay uncorrected PETBIDSMatrix."""
        tacs = self.get_decay_uncorrected_tacs()
        uncorrected_tacs = np.reshape(tacs, self.shape)

        json_dict = deepcopy(self.json_dict)
        json_dict["ImageDecayCorrected"] = False
        # PET-BIDS still requires "ImageDecayCorrectionTime" tag,
        # so we don't do anything about it

        return PETBIDSMatrix(uncorrected_tacs, json_dict)

    def to_filename(
        self,
        filename: str | PathLike[str],
        anchor: Literal["InjectionStart", "ScanStart"] = "InjectionStart",
        *,
        save_json: bool = False,
    ) -> None:
        """Save to file.

        Args:
            filename: file name for the tabular TAC tsv output
            save_json: whether the PET-BIDS json side car should be saved
            anchor: time anchor. The corresponding tag in the PET-BIDS json will
                    be set to zero (with appropriate offsets applied to other
                    tags).

        Raises:
            ValueError: file is not a tsv file

        """
        with Path(filename).open("w") as f:
            tsvwriter = csv.writer(f, delimiter="\t")
            tsvwriter.writerow(self.elem_names)
            for row in self.dataobj.T:
                tsvwriter.writerow(row)

        if save_json:
            self.set_timezero(anchor)

            fname = Path(filename)
            if fname.suffix != ".tsv":
                msg = "output file must be a tsv file"
                raise ValueError(msg)
            jsonfilename = fname.with_suffix(".json")
            write_json(self.json_dict, jsonfilename)


def load(
    filename: str | PathLike[str],
    jsonfilename: str | PathLike[str] | None = None,
) -> PETBIDSMatrix:
    """Read a tsv file containing temporal entries.

    Each column of the tsv file should be a time activity curve.

    Args:
        filename: path to the tsv file
        jsonfilename: path to the PET BIDS json file

    Returns:
        PETBIDSMatrix created from tsv file

    Raises:
        ValueError: file is not a tsv file

    """
    fname = Path(filename)

    if fname.suffix != ".tsv":
        msg = "output file must be a tsv file"
        raise ValueError(msg)

    if jsonfilename is None:
        jsonfilename = fname.with_suffix(".json")
    json_dict = read_json(jsonfilename)

    with fname.open() as f:
        tsvreader = csv.reader(f, delimiter="\t")
        header = next(tsvreader)
    tsv = np.genfromtxt(fname, delimiter="\t", skip_header=1)

    if (
        len(header) >= 2
        and header[0] == "FrameTimesStart"
        and header[1] == "FrameTimesEnd"
    ):
        frame_start = tsv[:, 0]
        frame_end = tsv[:, 1]
        frame_duration = frame_end - frame_start

        if "FrameTimesStart" not in json_dict:
            json_dict["FrameTimesStart"] = frame_start.tolist()
        if "FrameDuration" not in json_dict:
            json_dict["FrameDuration"] = frame_duration.tolist()

        tsv = tsv[:, 2:]
        header = header[2:]

    return PETBIDSMatrix(tsv.T, json_dict, header)


def load_inputfunction(
    filename: str | PathLike[str],
    column: str = "aif",
) -> TemporalMatrix:
    """Load an arterial input function from a tsv file.

    The tsv file should contain a ``time`` column in seconds and one or more
    columns describing the blood activity curves. By default, the column named
    ``AIF`` or the fifth column (zero-index 4) is used. ``column`` can be set to
    ``"plasma"`` or ``"whole_blood"`` to load the corresponding columns if they
    exist.

    Args:
        filename: path to the tsv file
        column: which column to load (``"aif"`` | ``"plasma"`` |
            ``"whole_blood"``)

    Returns:
        TemporalMatrix with times in seconds
    """

    fname = Path(filename)

    with fname.open() as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)

    header_lower = [h.lower() for h in header]
    try:
        time_idx = header_lower.index("time")
    except ValueError as exc:  # pragma: no cover - sanity check
        raise ValueError("time column not found") from exc

    col_name = column.lower()
    col_idx: int
    if col_name in {"aif", "metabolite_corrected_aif"}:
        if "aif" in header_lower:
            col_idx = header_lower.index("aif")
        else:
            col_idx = 4  # default location
    elif col_name in {"plasma", "plasma_radioactivity"}:
        col_idx = header_lower.index("plasma_radioactivity")
    elif col_name in {"whole_blood", "whole_blood_radioactivity"}:
        col_idx = header_lower.index("whole_blood_radioactivity")
    else:  # pragma: no cover - sanity check
        raise ValueError(f"Unrecognized column: {column}")

    tsv = np.genfromtxt(fname, delimiter="\t", skip_header=1)
    times = tsv[:, time_idx].astype(float)
    values = tsv[:, col_idx].astype(float)

    if len(times) > 1:
        frame_duration = np.diff(times, append=times[-1] + (times[-1] - times[-2]))
    else:
        frame_duration = np.array([1.0])

    return TemporalMatrix(values, times, frame_duration, [header[col_idx]])
