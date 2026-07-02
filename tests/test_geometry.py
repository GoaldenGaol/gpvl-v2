"""Tests for octonion/G2 geometry stubs."""

from volition.geometry.g2 import G2Projection
from volition.geometry.octonion import Octonion
from volition.state import VolitionalVector


def test_octonion_norm():
    o = Octonion.unit(3)
    assert o.norm == 1.0


def test_g2_roundtrip():
    vec = VolitionalVector(dim4=1.2, dim1=0.1)
    oct = G2Projection.from_volitional_vector(vec)
    recovered = G2Projection.to_volitional_vector(oct)
    assert abs(recovered.dim4 - 1.2) < 1e-10
    assert abs(recovered.dim1 - 0.1) < 1e-10