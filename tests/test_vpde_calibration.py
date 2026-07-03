"""Tests for VPDE calibration validation."""

from volition.constants import DIM4_IRREVERSIBLE
from volition.vpde import validate_calibration
from volition.vpde.calibration import KOREA_TARGET


def test_validate_calibration_usa():
    results = validate_calibration()
    usa = results[0]
    assert usa.target.name == "USA_irreversibility"
    assert usa.residual < 0.5


def test_korea_stays_irreversible():
    results = validate_calibration()
    korea = next(r for r in results if r.target.name == "South_Korea_stable_irreversible")
    assert korea.dim4_final >= DIM4_IRREVERSIBLE