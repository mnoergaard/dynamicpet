import numpy as np
import pytest
from dynamicpet.kineticmodel.ma1 import MA1
from dynamicpet.kineticmodel.loganblood import LoganBlood
from dynamicpet.kineticmodel.onetcm import OneTCM
from dynamicpet.kineticmodel.twotcm import TwoTCM
from dynamicpet.temporalobject import TemporalMatrix


@pytest.fixture
def simple_data():
    frame_start = np.array([0, 60, 120], dtype=float)
    frame_duration = np.array([60, 60, 60], dtype=float)
    cp = TemporalMatrix(np.array([10, 10, 10]), frame_start, frame_duration)
    tac_data = np.array([[5, 5, 5], [8, 8, 8]], dtype=float)
    tacs = TemporalMatrix(tac_data, frame_start, frame_duration)
    return cp, tacs


def test_ma1(simple_data):
    cp, tacs = simple_data
    km = MA1(cp, tacs)
    km.fit()
    dv = km.get_parameter("DV")
    assert dv.shape == (2,)


def test_loganblood(simple_data):
    cp, tacs = simple_data
    km = LoganBlood(cp, tacs)
    km.fit()
    dv = km.get_parameter("DV")
    assert dv.shape == (2,)


def test_onetcm(simple_data):
    cp, tacs = simple_data
    km = OneTCM(cp, tacs)
    km.fit()
    k1 = km.get_parameter("K1")
    assert k1.shape == (2,)
    fitted = km.fitted_tacs()
    assert fitted.dataobj.shape == tacs.dataobj.shape


def test_twotcm(simple_data):
    cp, tacs = simple_data
    km = TwoTCM(cp, tacs)
    km.fit()
    k1 = km.get_parameter("K1")
    assert k1.shape == (2,)
    fitted = km.fitted_tacs()
    assert fitted.dataobj.shape == tacs.dataobj.shape
