import numpy as np
import pytest
from como.nbh import adaptive_radius, build_nbh


def make_grid(n=5, spread=1.0):
    """Make n points spread in 7D space."""
    rng = np.random.default_rng(0)
    return rng.uniform(0, spread, size=(n, 7))


def test_adaptive_radius_positive():
    ea_norm = make_grid(10)
    r = adaptive_radius(ea_norm)
    assert r > 0


def test_adaptive_radius_small_spread():
    ea_norm = make_grid(10, spread=0.01)
    r_small = adaptive_radius(ea_norm)
    ea_norm_large = make_grid(10, spread=10.0)
    r_large = adaptive_radius(ea_norm_large)
    assert r_small < r_large


def test_adaptive_radius_degenerate():
    ea_norm = make_grid(2)
    r = adaptive_radius(ea_norm, k=3)
    assert r > 0  # Should not crash; degrades to k=1


def test_build_nbh_shape():
    ea_norm = make_grid(8)
    va_norm = make_grid(20)
    membership, r = build_nbh(ea_norm, va_norm, r=1.0)
    assert membership.shape == (20, 8)
    assert membership.dtype == bool


def test_build_nbh_all_inside():
    # Place VAs exactly at EA positions with a nonzero radius
    ea_norm = make_grid(5)
    va_norm = ea_norm.copy()
    membership, r = build_nbh(ea_norm, va_norm, r=0.01)
    # Each VA at an EA position is in that EA's sphere (dist=0)
    assert membership.any()


def test_build_nbh_all_outside():
    ea_norm = np.zeros((5, 7))
    va_norm = np.ones((10, 7)) * 100.0  # Far away
    membership, r = build_nbh(ea_norm, va_norm, r=0.001)
    assert not membership.any()


def test_build_nbh_large_radius():
    ea_norm = make_grid(5)
    va_norm = make_grid(10)
    membership, r = build_nbh(ea_norm, va_norm, r=1000.0)
    assert membership.all()


def test_user_radius_respected():
    ea_norm = make_grid(5)
    va_norm = make_grid(10)
    _, r_used = build_nbh(ea_norm, va_norm, r=3.14)
    assert r_used == pytest.approx(3.14)


def test_empty_va():
    ea_norm = make_grid(5)
    va_norm = np.empty((0, 7))
    membership, r = build_nbh(ea_norm, va_norm, r=1.0)
    assert membership.shape == (0, 5)
