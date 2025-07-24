"""Logan graphical analysis with plasma input."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import solve  # type: ignore[import-untyped]
from tqdm import trange

from .kineticmodel import KineticModel

if TYPE_CHECKING:
    from dynamicpet.temporalobject.temporalimage import TemporalImage
    from dynamicpet.temporalobject.temporalmatrix import TemporalMatrix
    from dynamicpet.temporalobject.temporalobject import (
        INTEGRATION_TYPE_OPTS,
        WEIGHT_OPTS,
    )
    from dynamicpet.typing_utils import NumpyNumberArray


class LoganBlood(KineticModel):
    """Logan plot with plasma input."""

    @classmethod
    def get_param_names(cls) -> list[str]:
        """Get names of kinetic model parameters."""
        return ["DV", "intercept"]

    def fit(
        self,
        mask: NumpyNumberArray | None = None,
        integration_type: INTEGRATION_TYPE_OPTS = "trapz",
        weight_by: WEIGHT_OPTS | NumpyNumberArray | None = "frame_duration",
        tstar: float = 0,
    ) -> None:
        """Estimate model parameters."""
        input_tac: NumpyNumberArray = self.reftac.dataobj.flatten()[:, np.newaxis]
        int_input = self.reftac.cumulative_integral(integration_type).flatten()[:, np.newaxis]

        tacs = self.tacs.timeseries_in_mask(mask)
        num_elements = tacs.num_elements
        tacs_mat = tacs.dataobj
        int_tacs_mat = tacs.cumulative_integral(integration_type)

        t_idx = tacs.frame_start >= tstar
        input_tstar = input_tac[t_idx, :]
        int_input_tstar = int_input[t_idx, :]
        tacs_mat_tstar = tacs_mat[:, t_idx]
        int_tacs_mat_tstar = int_tacs_mat[:, t_idx]

        weights = tacs.get_weights(weight_by)
        w_star = np.diag(weights[t_idx])

        dv = np.zeros((num_elements, 1))
        intercept = np.zeros((num_elements, 1))

        for k in trange(num_elements):
            tac_tstar = tacs_mat_tstar[k, :][:, np.newaxis]
            if np.allclose(tac_tstar, 0):
                continue
            int_tac_tstar = int_tacs_mat_tstar[k, :][:, np.newaxis]

            x = np.column_stack((int_input_tstar / tac_tstar, np.ones_like(tac_tstar)))
            y = int_tac_tstar / tac_tstar
            try:
                b = solve(x.T @ w_star @ x, x.T @ w_star @ y, assume_a="sym")
                dv[k] = b[0]
                intercept[k] = b[1]
            except LinAlgError:
                pass

        self.set_parameter("DV", dv, mask)
        self.set_parameter("intercept", intercept, mask)

    def fitted_tacs(self) -> TemporalMatrix | TemporalImage:
        """Get fitted TACs."""
        return self.tacs
