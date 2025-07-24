"""Two-tissue compartment model (2TCM)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import curve_fit  # type: ignore[import-untyped]
from tqdm import trange

from dynamicpet.temporalobject.temporalimage import TemporalImage, image_maker
from dynamicpet.temporalobject.temporalmatrix import TemporalMatrix

from .kineticmodel import KineticModel

if TYPE_CHECKING:
    from dynamicpet.typing_utils import NumpyNumberArray
    from dynamicpet.temporalobject.temporalobject import WEIGHT_OPTS


def twotcm_model(
    plasma: TemporalMatrix,
    K1: float,
    k2: float,
    k3: float,
    k4: float,
    vB: float = 0.0,
    blood: TemporalMatrix | None = None,
) -> NumpyNumberArray:
    """Generate tissue TAC for 2TCM using Euler integration."""
    t = plasma.frame_mid.astype(float)
    t_upsampled, step = np.linspace(np.min(t), np.max(t), 1024, retstep=True)
    cp = np.interp(t_upsampled, t, plasma.dataobj.astype(float).flatten())
    c1 = np.zeros(len(t_upsampled))
    c2 = np.zeros(len(t_upsampled))
    for i in range(1, len(t_upsampled)):
        dt = step
        c1[i] = c1[i - 1] + dt * (K1 * cp[i - 1] - (k2 + k3) * c1[i - 1] + k4 * c2[i - 1])
        c2[i] = c2[i - 1] + dt * (k3 * c1[i - 1] - k4 * c2[i - 1])
    tissue = c1 + c2
    if blood is not None:
        blood_up = np.interp(t_upsampled, t, blood.dataobj.astype(float).flatten())
        tissue = (1 - vB) * tissue + vB * blood_up
    return np.interp(t, t_upsampled, tissue)


class TwoTCM(KineticModel):
    """Two-tissue compartment model."""

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
        return ["K1", "k2", "k3", "k4", "vB"]

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
        k3 = np.zeros((num_elements, 1))
        k4 = np.zeros((num_elements, 1))
        vB = np.zeros((num_elements, 1))

        for k in trange(num_elements):
            if self.vB is None and self.blood_tac is not None:
                init = (0.5, 0.5, 0.05, 0.05, 0.05)
                popt, _ = curve_fit(
                    lambda t, p1, p2, p3, p4, p5: twotcm_model(
                        self.reftac, p1, p2, p3, p4, p5, self.blood_tac
                    ),
                    self.reftac.frame_mid,
                    roitacs[k, :].flatten(),
                    init,
                    sigma=weights,
                    bounds=([0, 0, 0, 0, 0], [10, 5, 5, 5, 1]),
                )
                K1[k], k2[k], k3[k], k4[k], vB[k] = popt
            else:
                vb_val = 0.0 if self.vB is None else self.vB
                init = (0.5, 0.5, 0.05, 0.05)
                popt, _ = curve_fit(
                    lambda t, p1, p2, p3, p4: twotcm_model(
                        self.reftac, p1, p2, p3, p4, vb_val, self.blood_tac
                    ),
                    self.reftac.frame_mid,
                    roitacs[k, :].flatten(),
                    init,
                    sigma=weights,
                    bounds=([0, 0, 0, 0], [10, 5, 5, 5]),
                )
                K1[k], k2[k], k3[k], k4[k] = popt
                vB[k] = vb_val

        self.set_parameter("K1", K1, mask)
        self.set_parameter("k2", k2, mask)
        self.set_parameter("k3", k3, mask)
        self.set_parameter("k4", k4, mask)
        self.set_parameter("vB", vB, mask)

    def fitted_tacs(self) -> TemporalMatrix | TemporalImage:
        num_elements = self.tacs.num_elements
        fitted = np.empty_like(self.tacs.dataobj)
        for i in trange(num_elements):
            idx = np.unravel_index(i, self.tacs.shape[:-1])
            k1_val = self.parameters["K1"][*idx]
            k2_val = self.parameters["k2"][*idx]
            k3_val = self.parameters["k3"][*idx]
            k4_val = self.parameters["k4"][*idx]
            vb_val = self.parameters["vB"][*idx]
            if k1_val or k2_val or k3_val or k4_val or vb_val:
                fitted[*idx, :] = twotcm_model(
                    self.reftac,
                    k1_val,
                    k2_val,
                    k3_val,
                    k4_val,
                    vb_val,
                    self.blood_tac,
                )
        if isinstance(self.tacs, TemporalImage):
            img = image_maker(fitted, self.tacs.img)
            return TemporalImage(img, self.tacs.frame_start, self.tacs.frame_duration)
        return TemporalMatrix(fitted, self.tacs.frame_start, self.tacs.frame_duration)
