"""Tests for octonion algebra and G2 calibration."""

import numpy as np
import pytest

from volition.geometry import (
    G2_DIM,
    calibrate_geometry,
    g2_cross_product,
    g2_generators_count,
    multiplication_table_latex,
)
from volition.geometry.g2 import G2Projection
from volition.geometry.octonion import Octonion
from volition.state import VolitionalVector


def test_octonion_norm():
    o = Octonion.unit(3)
    assert o.norm == 1.0


def test_fano_cycle_e1_e2_e4():
    e1, e2, e4 = Octonion.unit(1), Octonion.unit(2), Octonion.unit(4)
    assert np.allclose((e1 * e2).components, e4.components)
    assert np.allclose((e2 * e1).components, -e4.components)


def test_octonion_non_associative():
    e1, e2, e5 = Octonion.unit(1), Octonion.unit(2), Octonion.unit(5)
    lhs = (e1 * e2) * e5
    rhs = e1 * (e2 * e5)
    assert not np.allclose(lhs.components, rhs.components)


def test_octonion_norm_multiplicative_basis():
    """Normed division algebra: |e_i e_j| = 1 for unit imaginary products."""
    for i in range(1, 8):
        for j in range(1, 8):
            if i == j:
                continue
            prod = Octonion.unit(i) * Octonion.unit(j)
            if prod.norm > 0.5:
                assert prod.norm == pytest.approx(1.0, abs=1e-10)


def test_octonion_inverse():
    a = Octonion(np.array([1.0, 0.3, 0.2, 0.0, 0.1, 0.0, 0.0, 0.0]))
    identity = a * a.inverse()
    assert identity.components[0] == pytest.approx(1.0, abs=1e-10)
    assert np.allclose(identity.components[1:], 0.0, atol=1e-10)


def test_g2_roundtrip_uncalibrated():
    vec = VolitionalVector(dim4=1.2, dim1=0.1)
    proj = G2Projection()
    oct = proj.from_volitional_vector(vec)
    recovered = proj.to_volitional_vector(oct)
    assert abs(recovered.dim4 - 1.2) < 1e-10
    assert abs(recovered.dim1 - 0.1) < 1e-10


def test_g2_generators_dimension():
    assert g2_generators_count() == G2_DIM == 14


def test_g2_cross_product_standard():
    i = np.array([1.0, 0.0, 0.0])
    j = np.array([0.0, 1.0, 0.0])
    k = g2_cross_product(i, j)
    assert np.allclose(k, [0.0, 0.0, 1.0], atol=1e-10)


def test_calibrate_geometry_roundtrip():
    proj, result = calibrate_geometry()
    assert result.n_samples == 195
    assert result.roundtrip_max_error < 1e-10
    assert result.rank_correlation_preserved
    assert result.spearman_r == pytest.approx(1.0)


def test_embed_extract_dim4_calibrated():
    proj, _ = calibrate_geometry()
    for d in [0.5, 0.94, 1.2, 1.41]:
        oct = proj.embed_dim4(d)
        assert proj.extract_dim4(oct) == pytest.approx(d, rel=1e-9)


def test_multiplication_table_latex():
    tex = multiplication_table_latex()
    assert r"e_1 e_2" in tex
    assert r"\end{align}" in tex