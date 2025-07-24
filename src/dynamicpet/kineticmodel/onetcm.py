"""One-tissue compartment model (1TCM)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import curve_fit  # type: ignore[import-untyped]
from scipy.signal import convolve  # type: ignore[import-untyped]
from tqdm import trange

from dynamicpet.temporalobject.temporalimage import TemporalImage, image_maker
from dynamicpet.temporalobject.temporalmatrix import TemporalMatrix

from .kineticmodel import KineticModel

if TYPE_CHECKING:
    from dynamicpet.typing_utils import NumpyNumberArray
    from dynamicpet.temporalobject.temporalobject import WEIGHT_OPTS


def onetcm_model(
    plasma: TemporalMatrix,
    K1: float,
    k2: float,
    vB: float = 0.0,
    blood: TemporalMatrix | None = None,
) -> NumpyNumberArray:
    """Generate tissue TAC for 1TCM."""
    t = plasma.frame_mid.astype("float")
    t_upsampled, step = np.linspace(np.min(t), np.max(t), 1024, retstep=True)
    cp_upsampled = np.interp(t_upsampled, t, plasma.dataobj.astype(float).flatten())
    conv_res = (
        convolve(cp_upsampled, np.exp(-k2 * t_upsampled), mode="full")[: len(t_upsampled)]
        * step
    )
    tissue = K1 * conv_res
    if blood is not None:
        blood_up = np.interp(t_upsampled, t, blood.dataobj.astype(float).flatten())
        tissue = (1 - vB) * tissue + vB * blood_up
    return np.interp(t, t_upsampled, tissue)


class OneTCM(KineticModel):
    """One-tissue compartment model."""

    def __init__(
        self,
        reftac: TemporalMatrix,
        tacs: TemporalMatrix | TemporalImage,
        blood_tac: TemporalMatrix | None = None,
        vB: float | None = None,
    ) -> None:
        super().__init__(reftac, tacs)
        self.blood_tac = blood_tac
        self.vB = vB

    @classmethod
    def get_param_names(cls) -> list[str]:
        return ["K1", "k2", "vB"]

    def fit(
        self,
        mask: NumpyNumberArray | None = None,
        weight_by: WEIGHT_OPTS | NumpyNumberArray | None = None,
    ) -> None:
        tacs = self.tacs.timeseries_in_mask(mask)
        num_elements = tacs.num_elements
        roitacs = tacs.dataobj.reshape(num_elements, tacs.num_frames)

        weights = tacs.get_weights(weight_by)

        K1 = np.zeros((num_elements, 1))
        k2 = np.zeros((num_elements, 1))
        vB = np.zeros((num_elements, 1))

        for k in trange(num_elements):
            if self.vB is None and self.blood_tac is not None:
                init = (0.5, 0.1, 0.05)
                popt, _ = curve_fit(
                    lambda t, p1, p2, p3: onetcm_model(self.reftac, p1, p2, p3, self.blood_tac),
                    self.reftac.frame_mid,
                    roitacs[k, :].flatten(),
                    init,
                    sigma=weights,
                    bounds=([0, 0, 0], [10, 5, 1]),
                )
                K1[k], k2[k], vB[k] = popt
            else:
                vb_val = 0.0 if self.vB is None else self.vB
                init = (0.5, 0.1)
                popt, _ = curve_fit(
                    lambda t, p1, p2: onetcm_model(self.reftac, p1, p2, vb_val, self.blood_tac),
                    self.reftac.frame_mid,
                    roitacs[k, :].flatten(),
                    init,
                    sigma=weights,
                    bounds=([0, 0], [10, 5]),
                )
                K1[k], k2[k] = popt
                vB[k] = vb_val

        self.set_parameter("K1", K1, mask)
        self.set_parameter("k2", k2, mask)
        self.set_parameter("vB", vB, mask)

    def fitted_tacs(self) -> TemporalMatrix | TemporalImage:
        num_elements = self.tacs.num_elements
        fitted = np.empty_like(self.tacs.dataobj)
        for i in trange(num_elements):
            idx = np.unravel_index(i, self.tacs.shape[:-1])
            K1_val = self.parameters["K1"][*idx]
            k2_val = self.parameters["k2"][*idx]
            vb_val = self.parameters["vB"][*idx]
            if K1_val or k2_val or vb_val:
                fitted[*idx, :] = onetcm_model(
                    self.reftac, K1_val, k2_val, vb_val, self.blood_tac
                )
        if isinstance(self.tacs, TemporalImage):
            img = image_maker(fitted, self.tacs.img)
            return TemporalImage(img, self.tacs.frame_start, self.tacs.frame_duration)
        return TemporalMatrix(fitted, self.tacs.frame_start, self.tacs.frame_duration)
