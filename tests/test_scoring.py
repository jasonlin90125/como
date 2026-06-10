"""Unit tests for COMO score functions (Equations 1–6)."""

import numpy as np
import pytest
from como.scoring import c_score, d_score, s_score, p_score, assign_stage


def make_membership(n_va, n_ea, covered_va=None, overlap_pattern=None):
    """Make a boolean membership array.
    covered_va: list of VA indices that are in at least one NBH
    overlap_pattern: dict {va_idx: [ea_idx, ...]}
    """
    m = np.zeros((n_va, n_ea), dtype=bool)
    if covered_va is not None:
        for vi in covered_va:
            m[vi, 0] = True
    if overlap_pattern is not None:
        for vi, ea_list in overlap_pattern.items():
            for ei in ea_list:
                m[vi, ei] = True
    return m


# --- C score tests ---

def test_c_score_formula():
    # 4 of 10 VAs in at least one NBH
    m = make_membership(10, 3, covered_va=[0, 3, 5, 7])
    assert c_score(m) == pytest.approx(0.4)


def test_c_score_all_covered():
    m = np.ones((5, 3), dtype=bool)
    assert c_score(m) == pytest.approx(1.0)


def test_c_score_none_covered():
    m = np.zeros((10, 3), dtype=bool)
    assert c_score(m) == pytest.approx(0.0)


def test_c_score_empty():
    m = np.empty((0, 5), dtype=bool)
    assert c_score(m) == 0.0


# --- D score tests ---

def test_d_score_no_overlap():
    # All covered VAs in exactly one NBH -> d_mean=1 -> D=0
    m = np.zeros((6, 3), dtype=bool)
    m[0, 0] = True
    m[1, 1] = True
    m[2, 2] = True
    D, d_mean = d_score(m)
    assert d_mean == pytest.approx(1.0)
    assert D == pytest.approx(0.0)


def test_d_score_high_overlap():
    # 4 covered VAs each in 2 NBHs -> d_mean=2 -> D=0.5
    m = make_membership(4, 3, overlap_pattern={
        0: [0, 1], 1: [0, 1], 2: [1, 2], 3: [0, 2]
    })
    D, d_mean = d_score(m)
    assert d_mean == pytest.approx(2.0)
    assert D == pytest.approx(0.5)


def test_d_score_none_covered():
    m = np.zeros((5, 3), dtype=bool)
    D, d_mean = d_score(m)
    assert D == 0.0
    assert d_mean == 1.0


def test_d_score_empty():
    m = np.empty((0, 3), dtype=bool)
    D, d_mean = d_score(m)
    assert D == 0.0


# --- S score tests ---

def test_s_score_harmonic_mean():
    C, D = 0.43, 0.90
    expected = 2 * C * D / (C + D)
    assert s_score(C, D) == pytest.approx(expected)


def test_s_score_zero_when_d_zero():
    assert s_score(1.0, 0.0) == pytest.approx(0.0)


def test_s_score_zero_when_c_zero():
    assert s_score(0.0, 0.9) == pytest.approx(0.0)


def test_s_score_both_zero():
    assert s_score(0.0, 0.0) == 0.0


def test_s_score_paper_table1_as1_approx():
    # Paper Table 1: S≈0.58, C≈0.43, D≈0.90
    assert s_score(0.43, 0.90) == pytest.approx(0.578, abs=0.01)


# --- P score tests ---

def test_p_score_empty():
    m = np.empty((0, 3), dtype=bool)
    assert p_score(m, np.array([7.0, 6.5, 8.0])) == 0.0


def test_p_score_no_overlap():
    # No VA in >=2 NBHs
    m = make_membership(5, 3, covered_va=[0, 1, 2])
    acts = np.array([7.0, 6.5, 8.0])
    assert p_score(m, acts) == 0.0


def test_p_score_zero_uniform_activity():
    # All EAs have same pActivity -> all |pot_a - pot_b| = 0 -> P=0
    m = make_membership(3, 3, overlap_pattern={0: [0, 1], 1: [1, 2], 2: [0, 1, 2]})
    acts = np.array([7.0, 7.0, 7.0])
    assert p_score(m, acts) == pytest.approx(0.0)


def test_p_score_known_value():
    # VA j=0 in NBHs of EA 0 and EA 1 with pActs 7.0 and 8.0
    # m_j=2, delta_j = (2/(2*1)) * |7-8| = 1.0, w_j = 0.5
    # P = (0.5 * 1.0) / 0.5 = 1.0
    m = np.zeros((2, 2), dtype=bool)
    m[0, 0] = True
    m[0, 1] = True
    # VA 1 has no overlap
    acts = np.array([7.0, 8.0])
    assert p_score(m, acts) == pytest.approx(1.0)


def test_p_score_three_way_overlap():
    # VA in NBHs of 3 EAs with pActs 6, 7, 8
    # m_j=3, C(3,2)=3 pairs: |6-7|=1, |6-8|=2, |7-8|=1
    # delta_j = (2/(3*2)) * (1+2+1) = (2/6)*4 = 4/3
    # w_j = 1/3
    # P = (1/3 * 4/3) / (1/3) = 4/3
    m = np.zeros((1, 3), dtype=bool)
    m[0, :] = True
    acts = np.array([6.0, 7.0, 8.0])
    expected = (2.0 / (3 * 2)) * (1 + 2 + 1)
    assert p_score(m, acts) == pytest.approx(expected)


# --- Stage assignment tests ---

def test_assign_stage_early():
    assert assign_stage(0.1, 0.2) == "early"


def test_assign_stage_early_mid():
    assert assign_stage(0.1, 0.8) == "early_mid"


def test_assign_stage_mid():
    assert assign_stage(0.7, 0.2) == "mid"


def test_assign_stage_late():
    assert assign_stage(0.7, 0.8) == "late"


def test_assign_stage_at_threshold():
    # Exactly at threshold: >= means high
    assert assign_stage(0.4, 0.5) == "late"
    assert assign_stage(0.39, 0.49) == "early"
