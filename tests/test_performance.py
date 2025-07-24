"""Tests for ModelPerformance."""

import numpy as np
from dynamicpet.kineticmodel import ModelPerformance


def make_perf() -> ModelPerformance:
    observed = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.1, 1.9, 2.8])
    return ModelPerformance(observed, predicted, num_parameters=2)


def test_mse() -> None:
    perf = make_perf()
    assert np.isclose(perf.mse, 0.02)


def test_sigma_squared() -> None:
    perf = make_perf()
    assert np.isclose(perf.sigma_squared, 0.06)


def test_aic() -> None:
    perf = make_perf()
    assert np.isclose(perf.aic, -7.736069016284432)


def test_fpe() -> None:
    perf = make_perf()
    assert np.isclose(perf.fpe, 0.3)


def test_coef_variation() -> None:
    perf = make_perf()
    assert np.isclose(perf.coef_variation, 0.12247448713915901)
