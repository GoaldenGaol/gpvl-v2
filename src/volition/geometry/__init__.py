"""Octonion and G2 geometry for 8D→7D volitional embedding."""

from volition.geometry.calibration import GeometryCalibrationResult, calibrate_geometry
from volition.geometry.g2 import (
    G2_DIM,
    G2Calibration,
    G2Projection,
    g2_cross_product,
    g2_generators_count,
)
from volition.geometry.octonion import Octonion, multiplication_table_latex

__all__ = [
    "G2Calibration",
    "G2Projection",
    "G2_DIM",
    "GeometryCalibrationResult",
    "Octonion",
    "calibrate_geometry",
    "g2_cross_product",
    "g2_generators_count",
    "multiplication_table_latex",
]