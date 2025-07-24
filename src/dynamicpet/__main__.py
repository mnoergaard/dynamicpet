"""Command-line interface."""

from __future__ import annotations

import csv
import re
import warnings
from json import dump as json_dump
from pathlib import Path
from typing import TYPE_CHECKING

import click
import numpy as np
from nibabel.filename_parser import splitext_addext
from nibabel.loadsave import load as nib_load

from dynamicpet import __version__
from dynamicpet.denoise import hypr, nesma
from dynamicpet.kineticmodel.srtm import SRTMLammertsma1996, SRTMZhou2003
from dynamicpet.kineticmodel.suvr import SUVR
from dynamicpet.petbids import PETBIDSImage, PETBIDSMatrix
from dynamicpet.petbids.petbidsimage import load as petbidsimage_load
from dynamicpet.petbids.petbidsmatrix import load as petbidsmatrix_load
from dynamicpet.temporalobject.temporalobject import INTEGRATION_TYPE_OPTS, WEIGHT_OPTS

if TYPE_CHECKING:
    from nibabel.spatialimages import SpatialImage

    from dynamicpet.kineticmodel.kineticmodel import KineticModel
    from dynamicpet.temporalobject import TemporalMatrix
    from dynamicpet.typing_utils import NumpyNumberArray

IMPLEMENTED_KMS = [
    "SUVR",
    "SRTMLammertsma1996",
    "SRTMZhou2003",
    "MA1",
    "LoganBlood",
    "OneTCM",
    "TwoTCM",
]

INTEGRATION_TYPES = (
    str(INTEGRATION_TYPE_OPTS).replace("'", "").split("[")[1].split("]")[0].split(", ")
)
WEIGHTS = str(WEIGHT_OPTS).replace("'", "").split("[")[1].split("]")[0].split(", ")


@click.command()
@click.argument("pet", type=str)
@click.option(
    "--method",
    type=click.Choice(["HYPRLR", "NESMA"], case_sensitive=False),
    required=True,
    help="Name of denoising method",
)
@click.option(
    "--fwhm",
    default=None,
    type=float,
    help="Full width at half max in mm for smoothing (only for HYPR-LR)",
)
@click.option(
    "--mask",
    default=None,
    type=str,
    help=(
        "Binary mask specifying voxels where denoising should be performed. "
        "(only for NESMA)"
    ),
)
@click.option(
    "--window_half_size",
    default=None,
    nargs=3,
    type=int,
    help=(
        "The size of search window centered around a voxel will be "
        "2 * window_half_size + 1 (any part of the window that extends beyond "
        "the image will be truncated). (only for NESMA)"
    ),
)
@click.option(
    "--thresh",
    default=0.05,
    type=click.FloatRange(0, 1),
    help="threshold (only for NESMA)",
)
@click.option(
    "--outputdir",
    default=None,
    type=str,
    help=(
        "Directory in which to save denoised image. "
        "If not provided, it will be saved in the same directory as the PET image."
    ),
)
@click.option("--json", default=None, type=str, help="PET-BIDS json file")
def denoise(  # noqa: PLR0912, PLR0913
    pet: str,
    method: str,
    fwhm: float | None,
    mask: str | None,
    window_half_size: tuple[int, int, int] | None,
    thresh: float | None,
    outputdir: str | None,
    json: str | None,
) -> None:
    """Perform dynamic PET denoising.

    Outputs will have a '_desc-<method>_pet' suffix.

    PET: 4-D PET image
    """
    # load PET
    pet_img = petbidsimage_load(pet, json)

    if method == "HYPRLR":
        doc = hypr.hypr_lr.__doc__

        # parameters not relevant to HYPR-LR
        mask = None
        window_half_size = None
        thresh = None

        if fwhm:
            res = hypr.hypr_lr(pet_img, fwhm)
        else:
            msg = "fwhm must be specified for HYPR-LR"
            raise ValueError(msg)
    elif method == "NESMA":
        doc = nesma.nesma_semiadaptive.__doc__

        # parameter not relevant to NESMA
        fwhm = None

        if mask:
            mask_img: SpatialImage = nib_load(mask)  # type: ignore[assignment]
            # check that mask is in the same space as pet
            if not np.all(pet_img.img.affine == mask_img.affine):
                msg = "PET and mask are not in the same space"
                raise ValueError(msg)
            mask_img_mat: NumpyNumberArray = mask_img.get_fdata().astype("bool")

            if window_half_size:
                if thresh:
                    res, _ = nesma.nesma_semiadaptive(
                        pet_img,
                        mask_img_mat,
                        window_half_size,
                        thresh,
                    )
                else:
                    msg = "thresh must be specified for NESMA"
                    raise ValueError(msg)
            else:
                msg = "window_half_size must be specified for NESMA"
                raise ValueError(msg)
        else:
            msg = "mask must be specified for NESMA"
            raise ValueError(msg)
    else:
        msg = f"Denoising method {method} is not supported"
        raise NotImplementedError(msg)

    froot, ext, addext = splitext_addext(pet)
    if outputdir:
        odir = Path(outputdir)
        odir.mkdir(parents=True, exist_ok=True)
        bname = Path(froot).name
        # if the input file name follows the PET-BIDS convention, it should end
        # with "_pet". Need to move this to the end of the new file name to
        # maintain compatibility with the PET-BIDS Derivatives convention.
        froot = re.sub("_pet$", "", str(odir / bname))
    output = froot + "_desc-" + method.lower() + "_pet" + ext + addext
    output_json = froot + "_desc-" + method.lower() + "_pet.json"
    res.to_filename(output)

    cmd = (
        f"denoise --method {method} "
        + (f"--fwhm {fwhm} " if fwhm else "")
        + (f"--mask {mask} " if mask else "")
        + (f"--window_half_size {window_half_size} " if window_half_size else "")
        + (f"--thresh {thresh} " if thresh else "")
        + (f"--outputdir {outputdir} " if outputdir else "")
        + (f"--json {json} " if json else "")
        + pet
    )
    derivative_json_dict = {
        "Description": (
            re.sub(r"\s+", " ", doc.split("Args:")[0]).strip() if doc else ""
        ),
        # "Sources": [pet],  # noqa: ERA001
        "SoftwareName": "dynamicpet",
        "SoftwareVersion": __version__,
        "CommandLine": cmd,
    }

    with Path(output_json).open("w") as f:
        json_dump(derivative_json_dict, f, indent=4)


@click.command()
@click.argument("pet", type=str)
@click.option(
    "--model",
    type=click.Choice(IMPLEMENTED_KMS, case_sensitive=False),
    required=True,
    help="Name of kinetic model (" + ", ".join(IMPLEMENTED_KMS) + ")",
)
@click.option(
    "--refroi",
    default=None,
    type=str,
    help=(
        "Name of reference region. "
        "Required when PET is specified as a tsv file. "
        "REFROI option cannot be used if PET is specified as an image."
    ),
)
@click.option(
    "--refmask",
    default=None,
    type=str,
    help=(
        "3-D binary mask indicating reference tissue (must be in alignment with PET). "
        "REFMASK option cannot be used if PET is specified as a tsv file."
    ),
)
@click.option(
    "--outputdir",
    default=None,
    type=str,
    help=(
        "Directory in which to save each estimated parametric image. "
        "If not provided, they will be saved in the same directory as the PET image."
    ),
)
@click.option(
    "--json",
    default=None,
    type=str,
    help=(
        "PET-BIDS json file. "
        "If not specified, it is assumed to have the same name as the PET "
        "image file, except with a .json extension."
    ),
)
@click.option(
    "--plasmatac",
    default=None,
    type=str,
    help="Tabular plasma TAC file for blood-based models",
)
@click.option(
    "--bloodtac",
    default=None,
    type=str,
    help="Tabular whole blood TAC file for blood-based models",
)
@click.option(
    "--petmask",
    default=None,
    type=str,
    help=(
        "Binary mask specifying voxels where model should be fitted. "
        "Voxels outside this binary mask in the resulting parametric images "
        "will be set to NA."
    ),
)
@click.option(
    "--start",
    default=None,
    type=float,
    help="Start of time window for model in min",
)
@click.option(
    "--end",
    default=None,
    type=float,
    help="End of time window for model in min",
)
@click.option(
    "--fwhm",
    default=None,
    type=float,
    help="Full width at half max in mm for smoothing, used for some models",
)
@click.option(
    "--weight_by",
    type=click.Choice(WEIGHTS, case_sensitive=False),
    default="frame_duration",
    help="Frame weights used in estimation procedures involving regression.",
)
@click.option(
    "--integration_type",
    type=click.Choice(INTEGRATION_TYPES, case_sensitive=False),
    default="trapz",
    help=(
        "Integration type used in estimation procedures involving "
        "time integrals. "
        "'trapz' is trapezoidal integration and "
        "'rect' is rectangular integration."
    ),
)
@click.option(
    "--tstar",
    default=None,
    type=float,
    help="Time beyond which to assume linearity (for LoganBlood/MA1)",
)
@click.option(
    "--vb-fixed",
    default=None,
    type=float,
    help="Fixed fractional blood volume for compartmental models",
)
@click.option(
    "--fit-end-time",
    default=None,
    type=float,
    help="End time in min over which model is fitted",
)
def kineticmodel(  # noqa: C901, PLR0912, PLR0913, PLR0915
    pet: str,
    model: str,
    refroi: str | None,
    refmask: str | None,
    outputdir: str | None,
    json: str | None,
    plasmatac: str | None,
    bloodtac: str | None,
    petmask: str | None,
    start: float | None,
    end: float | None,
    fwhm: float | None,
    tstar: float | None,
    vb_fixed: float | None,
    fit_end_time: float | None,
    weight_by: WEIGHT_OPTS = "frame_duration",
    integration_type: INTEGRATION_TYPE_OPTS = "trapz",
) -> None:
    """Fit a reference tissue model to a dynamic PET image or TACs.

    Outputs will have a '_model-<model>_meas-<parameter>' suffix.

    PET: 4-D PET image (can be 3-D if model is SUVR) or a 2-D tabular TACs tsv file
    """
    model = model.lower()

    froot, ext, addext = splitext_addext(pet)

    pet_img, reftac, petmask_img_mat, plasma_tac, blood_tac = parse_kineticmodel_inputs(
        pet,
        json,
        refroi,
        refmask,
        petmask,
        plasmatac,
        bloodtac,
    )

    if start is None:
        start = pet_img.start_time
    if end is None:
        end = pet_img.end_time
    if start != pet_img.start_time or end != pet_img.end_time:
        pet_img = pet_img.extract(start, end)
        reftac = reftac.extract(start, end)
        if plasma_tac is not None:
            plasma_tac = plasma_tac.extract(start, end)
        if blood_tac is not None:
            blood_tac = blood_tac.extract(start, end)

    if fwhm and model not in ["srtmzhou2003"]:
        fwhm = None
        warnings.warn(
            "--fwhm argument is not relevant for this model, will be ignored",
            RuntimeWarning,
            stacklevel=2,
        )
    if tstar is not None and model not in ["ma1", "loganblood"]:
        tstar = None
        warnings.warn(
            "--tstar argument is not relevant for this model, will be ignored",
            RuntimeWarning,
            stacklevel=2,
        )
    if vb_fixed is not None and model not in ["onetcm", "twotcm"]:
        vb_fixed = None
        warnings.warn(
            "--vb-fixed argument is not relevant for this model, will be ignored",
            RuntimeWarning,
            stacklevel=2,
        )
    if fit_end_time is not None and model not in ["ma1", "loganblood"]:
        fit_end_time = None
        warnings.warn(
            "--fit-end-time argument is not relevant for this model, will be ignored",
            RuntimeWarning,
            stacklevel=2,
        )

    if model in ["ma1", "loganblood", "onetcm", "twotcm"]:
        if plasma_tac is None:
            msg = "--plasmatac must be specified for plasma input models"
            raise ValueError(msg)
        input_tac = plasma_tac
    else:
        input_tac = reftac

    # fit kinetic model
    km: KineticModel
    match model:
        case "suvr":
            model_abbr = "SUVR"
            km = SUVR(reftac, pet_img)
            km.fit(mask=petmask_img_mat)
        case "srtmlammertsma1996":
            model_abbr = "SRTM"
            km = SRTMLammertsma1996(reftac, pet_img)
            km.fit(mask=petmask_img_mat, weight_by=weight_by)
        case "srtmzhou2003":
            model_abbr = "SRTM"
            km = SRTMZhou2003(reftac, pet_img)
            km.fit(
                mask=petmask_img_mat,
                integration_type=integration_type,
                weight_by=weight_by,
                fwhm=fwhm,
            )
        case "ma1":
            model_abbr = "MA1"
            try:
                from dynamicpet.kineticmodel.ma1 import (
                    MA1,  # type: ignore[import-not-found]
                )
            except Exception as exc:  # pragma: no cover - optional dep
                msg = "MA1 model not available"
                raise NotImplementedError(msg) from exc
            km = MA1(input_tac, pet_img)
            km.fit(
                mask=petmask_img_mat,
                integration_type=integration_type,
                weight_by=weight_by,
                tstar=0.0 if tstar is None else tstar,
            )
        case "loganblood":
            model_abbr = "Logan"
            try:
                from dynamicpet.kineticmodel.logan import (
                    LoganBlood,  # type: ignore[import-not-found]
                )
            except Exception as exc:  # pragma: no cover - optional dep
                msg = "LoganBlood model not available"
                raise NotImplementedError(msg) from exc
            km = LoganBlood(input_tac, pet_img)
            km.fit(
                mask=petmask_img_mat,
                integration_type=integration_type,
                weight_by=weight_by,
                tstar=0.0 if tstar is None else tstar,
            )
        case "onetcm":
            model_abbr = "1TCM"
            try:
                from dynamicpet.kineticmodel.onetcm import (
                    OneTCM,  # type: ignore[import-not-found]
                )
            except Exception as exc:  # pragma: no cover - optional dep
                msg = "OneTCM model not available"
                raise NotImplementedError(msg) from exc
            km = OneTCM(input_tac, pet_img, blood_tac)
            km.fit(
                mask=petmask_img_mat,
                integration_type=integration_type,
                weight_by=weight_by,
                vb_fixed=vb_fixed,
            )
        case "twotcm":
            model_abbr = "2TCM"
            try:
                from dynamicpet.kineticmodel.twotcm import (
                    TwoTCM,  # type: ignore[import-not-found]
                )
            except Exception as exc:  # pragma: no cover - optional dep
                msg = "TwoTCM model not available"
                raise NotImplementedError(msg) from exc
            km = TwoTCM(input_tac, pet_img, blood_tac)
            km.fit(
                mask=petmask_img_mat,
                integration_type=integration_type,
                weight_by=weight_by,
                vb_fixed=vb_fixed,
            )
        case _:
            msg = f"Model {model} is not supported"
            raise ValueError(msg)

    if outputdir:
        odir = Path(outputdir)
        odir.mkdir(parents=True, exist_ok=True)
        bname = Path(froot).name
        froot = str(odir / bname)

    if isinstance(pet_img, PETBIDSMatrix):
        # if the input file name follows the PET-BIDS Derivatives convention,
        # it should end with "_tacs". Need to remove this to maintain
        # compatibility with the PET-BIDS Derivatives convention.
        froot = re.sub("_tacs$", "", froot)

        output = froot + "_model-" + model_abbr + "_kinpar" + ext
        output_json = froot + "_model-" + model_abbr + "_kinpar.json"
        data = np.empty((len(km.parameters), pet_img.num_elements))
        for i, param in enumerate(km.parameters.keys()):
            data[i] = km.get_parameter(param)
        datat = data.T
        with Path(output).open("w") as f:
            tsvwriter = csv.writer(f, delimiter="\t")
            tsvwriter.writerow(["name", *list(km.parameters.keys())])
            for i, elem in enumerate(pet_img.elem_names):
                tsvwriter.writerow([elem, *datat[i].tolist()])
    else:
        # if the input file name follows the PET-BIDS convention, it should end
        # with "_pet". Need to remove this to maintain compatibility with the
        # PET-BIDS Derivatives convention.
        froot = re.sub("_pet$", "", froot)

        # save estimated parameters as image
        for param in km.parameters:
            res_img: SpatialImage = km.get_parameter(param)  # type: ignore[assignment]
            output = (
                froot
                + "_model-"
                + model_abbr
                + "_meas-"
                + param
                + "_mimap"
                + ext
                + addext
            )
            res_img.to_filename(output)
        output_json = froot + "_model-" + model_abbr + "_mimap.json"

    # save json PET BIDS derivative file
    inputvalues = [start, end]
    inputvalueslabels = [
        "Start of time window for model",
        "End of time window for model",
    ]
    inputvaluesunits = ["min", "min"]

    if fwhm:
        inputvalues += [fwhm]
        inputvalueslabels += ["Full width at half max"]
        inputvaluesunits += ["mm"]
    if tstar is not None:
        inputvalues += [tstar]
        inputvalueslabels += ["tstar"]
        inputvaluesunits += ["min"]
    if vb_fixed is not None:
        inputvalues += [vb_fixed]
        inputvalueslabels += ["Fixed blood volume fraction"]
        inputvaluesunits += [""]
    if fit_end_time is not None:
        inputvalues += [fit_end_time]
        inputvalueslabels += ["End time for fitting"]
        inputvaluesunits += ["min"]

    cmd = (
        f"kineticmodel --model {model} "
        + (f"--refroi {refroi} " if refroi else f"--refmask {refmask} ")
        + (f"--outputdir {outputdir} " if outputdir else "")
        + (f"--json {json} " if json else "")
        + (f"--plasmatac {plasmatac} " if plasmatac else "")
        + (f"--bloodtac {bloodtac} " if bloodtac else "")
        + (f"--petmask {petmask} " if petmask else "")
        + f"--start {start} "
        + f"--end {end} "
        + (f"--fwhm {fwhm} " if fwhm else "")
        + (f"--tstar {tstar} " if tstar is not None else "")
        + (f"--vb-fixed {vb_fixed} " if vb_fixed is not None else "")
        + (f"--fit-end-time {fit_end_time} " if fit_end_time is not None else "")
        + f"--weight_by {weight_by} "
        + f"--integration_type {integration_type} "
        + pet
    )
    doc = km.__class__.__doc__
    derivative_json_dict = {
        "Description": re.sub(r"\s+", " ", doc) if doc else "",
        # "Sources": [pet],  # noqa: ERA001
        "ModelName": model_abbr,
        "ReferenceRegion": refroi if refroi else refmask,
        "AdditionalModelDetails": (
            f"Frame weighting by: {weight_by}. Integration type: {integration_type}."
        ),
        "InputValues": inputvalues,
        "InputValuesLabels": inputvalueslabels,
        "InputValuesUnits": inputvaluesunits,
        "SoftwareName": "dynamicpet",
        "SoftwareVersion": __version__,
        "CommandLine": cmd,
    }

    with Path(output_json).open("w") as f:
        json_dump(derivative_json_dict, f, indent=4)


def parse_kineticmodel_inputs(  # noqa: C901, PLR0913, PLR0912, PLR0915
    tac_object_file: str,
    tac_object_json: str | None = None,
    refroi: str | None = None,
    refmask: str | None = None,
    petmask: str | None = None,
    plasmatac: str | None = None,
    bloodtac: str | None = None,
) -> tuple[
    PETBIDSImage | PETBIDSMatrix,
    TemporalMatrix,
    NumpyNumberArray | None,
    TemporalMatrix | None,
    TemporalMatrix | None,
]:
    """Parse kinetic model inputs.

    Args:
        tac_object_file: 3-D or 4-D PET image file or a 2-D tabular TACs tsv file
        tac_object_json: PET-BIDS json accompanying tac_object_file
                         (if not provided, will be assumed to have the same name
                          as tac_object_file, except with a .json extension)
        refroi: Used only if tac_object_file is a tsv file. Name of reference
                region as specified in the tsv file.
        refmask: Used only if tac_object_file is an image file. 3-D binary mask
                 indicating reference tissue (must be in alignment with PET).
        petmask: Used only if tac_object_file is an image file. Binary mask
                 specifying voxels where model should be fitted.
        plasmatac: Plasma input TAC in tsv format for blood-based models
        bloodtac: Whole blood TAC in tsv format for blood-based models

    Returns:
        tac_object: 3-D or 4-D PET image or 2-D TACs
        reftac: reference TAC
        petmask_img_mat: binary PET mask
        plasma_tac: plasma input TAC if provided
        blood_tac: whole blood TAC if provided

    Raises:
        ValueError: refroi or refmask is not specified, or PET and mask are not
                    in the same space

    """
    plasma_tm: PETBIDSMatrix | None = None
    blood_tm: PETBIDSMatrix | None = None
    plasma = None
    blood = None
    if plasmatac is not None:
        plasma_tm = petbidsmatrix_load(plasmatac)
        plasma = plasma_tm.get_elem(plasma_tm.elem_names[0])
    if bloodtac is not None:
        blood_tm = petbidsmatrix_load(bloodtac)
        blood = blood_tm.get_elem(blood_tm.elem_names[0])

    if tac_object_file[-4:] == ".tsv":
        tac_object_tm: PETBIDSMatrix = petbidsmatrix_load(
            tac_object_file,
            tac_object_json,
        )
        if refroi:
            reftac = tac_object_tm.get_elem(refroi)
        elif plasma is not None:
            reftac = plasma
        else:
            msg = "refroi must be specified when tac_object is a tsv file"
            raise ValueError(msg)

        return tac_object_tm, reftac, None, plasma, blood

    # otherwise, read as PETBIDSImage
    if not refmask and plasma is None:
        msg = "refmask must be specified when tac_object is an image"
        raise ValueError(msg)

    tac_object: PETBIDSImage = petbidsimage_load(tac_object_file, tac_object_json)
    refmask_img_mat: NumpyNumberArray | None = None
    if refmask:
        refmask_img: SpatialImage = nib_load(refmask)  # type: ignore[assignment]
        if not np.all(tac_object.img.affine == refmask_img.affine):
            msg = "PET and refmask are not in the same space"
            raise ValueError(msg)
        refmask_img_mat = refmask_img.get_fdata().astype("bool")

    petmask_img_mat: NumpyNumberArray | None
    if petmask:
        petmask_img: SpatialImage = nib_load(petmask)  # type: ignore[assignment]
        # check that petmask is in the same space as pet
        if not np.all(tac_object.img.affine == petmask_img.affine):
            msg = "PET image and petmask are not in the same space"
            raise ValueError(msg)

        petmask_img_mat = petmask_img.get_fdata().astype("bool")
        # check that refmask is fully within petmask
        if not np.all(petmask_img_mat[refmask_img_mat]):
            warnings.warn(
                (
                    "refmask is not fully within petmask. "
                    "Will restrict refmask to within petmask."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            if refmask_img_mat is not None:
                refmask_img_mat = refmask_img_mat * petmask_img_mat
    else:
        petmask_img_mat = None

    # extract average TAC in reference region
    if refmask_img_mat is not None:
        reftac = tac_object.mean_timeseries_in_mask(refmask_img_mat)
    elif plasma is not None:
        reftac = plasma
    else:
        msg = "Unable to determine reference TAC"
        raise ValueError(msg)

    return tac_object, reftac, petmask_img_mat, plasma, blood


if __name__ == "__main__":
    kineticmodel(prog_name="kineticmodel")  # pragma: no cover
