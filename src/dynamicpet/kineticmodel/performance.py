"""Model performance metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class ModelPerformance:
    """Evaluate performance of a fitted model."""

    observed: ArrayLike
    predicted: ArrayLike
    num_parameters: int

    def __post_init__(self) -> None:
        self.observed = np.asarray(self.observed, dtype=float)
        self.predicted = np.asarray(self.predicted, dtype=float)
        if self.observed.shape != self.predicted.shape:
            msg = "observed and predicted must have the same shape"
            raise ValueError(msg)

    @property
    def residuals(self) -> np.ndarray:
        """Get residuals."""
        return self.observed - self.predicted

    @property
    def n(self) -> int:
        """Number of observations."""
        return int(self.observed.size)

    @property
    def rss(self) -> float:
        """Residual sum of squares."""
        return float(np.sum(self.residuals**2))

    @property
    def mse(self) -> float:
        """Mean squared error."""
        return self.rss / self.n

    @property
    def sigma_squared(self) -> float:
        """Unbiased residual variance estimate."""
        df = self.n - self.num_parameters
        if df <= 0:
            msg = "Degrees of freedom must be positive"
            raise ValueError(msg)
        return self.rss / df

    @property
    def aic(self) -> float:
        """Akaike information criterion."""
        return self.n * float(np.log(self.rss / self.n)) + 2 * self.num_parameters

    @property
    def fpe(self) -> float:
        """Final prediction error."""
        df = self.n - self.num_parameters
        if df <= 0:
            msg = "Degrees of freedom must be positive"
            raise ValueError(msg)
        return self.sigma_squared * (self.n + self.num_parameters) / df

    @property
    def coef_variation(self) -> float:
        """Coefficient of variation of residuals."""
        mean_obs = float(np.mean(self.observed))
        if mean_obs == 0:
            return np.inf
        return float(np.sqrt(self.sigma_squared) / mean_obs)
