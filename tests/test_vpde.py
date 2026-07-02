"""Tests for VPDE 1D solver."""

import numpy as np

from volition.vpde import solve_vpde


def test_vpde_returns_trajectory():
    t, dim4 = solve_vpde(0.85)
    assert len(t) == 200
    assert len(dim4) == 200
    assert np.all(np.isfinite(dim4))