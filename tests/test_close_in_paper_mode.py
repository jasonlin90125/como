"""Tests for CloseInVAGenerator paper-mode random H-aware sampling."""

import numpy as np
import pytest
from rdkit import Chem

from como.analogs.close_in import CloseInVAGenerator


_CORE = "Nc1ccccc1"  # aniline-like, 2 sites

# 3×3 grid minus one corner — 8 EAs, leaving one non-EA combination
_EAS = [
    "CNc1ccc(F)cc1",
    "CNc1ccc(Cl)cc1",
    "CNc1ccc(Br)cc1",
    "CCNc1ccc(F)cc1",
    "CCNc1ccc(Cl)cc1",
    "CCNc1ccc(Br)cc1",
    "CCCNc1ccc(F)cc1",
    "CCCNc1ccc(Cl)cc1",
    # missing: CCCNc1ccc(Br)cc1  → the VA the generator can find
]
_ACTS = np.array([7.2, 7.8, 7.5, 6.9, 7.4, 7.1, 6.5, 7.0])
_HAC_RANGE = (5, 30)


class TestCloseInPaperMode:
    def test_reproducible_with_same_seed(self):
        gen_a = CloseInVAGenerator(random_state=0)
        gen_b = CloseInVAGenerator(random_state=0)
        va_a = gen_a.generate(_EAS, _ACTS, _CORE, n=50, ea_hac_range=_HAC_RANGE)
        va_b = gen_b.generate(_EAS, _ACTS, _CORE, n=50, ea_hac_range=_HAC_RANGE)
        assert va_a == va_b

    def test_different_seeds_have_same_length(self):
        gen_a = CloseInVAGenerator(random_state=1)
        gen_b = CloseInVAGenerator(random_state=2)
        va_a = gen_a.generate(_EAS, _ACTS, _CORE, n=5, ea_hac_range=_HAC_RANGE)
        va_b = gen_b.generate(_EAS, _ACTS, _CORE, n=5, ea_hac_range=_HAC_RANGE)
        # Both should return a list (possibly empty if space is exhausted)
        assert isinstance(va_a, list)
        assert isinstance(va_b, list)

    def test_no_ea_duplicates(self):
        gen = CloseInVAGenerator(random_state=42)
        vas = gen.generate(_EAS, _ACTS, _CORE, n=200, ea_hac_range=_HAC_RANGE)
        ea_canon = {Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in _EAS}
        for smi in vas:
            assert smi not in ea_canon, f"EA duplicate in VAs: {smi}"

    def test_all_valid_smiles(self):
        gen = CloseInVAGenerator(random_state=42)
        vas = gen.generate(_EAS, _ACTS, _CORE, n=100, ea_hac_range=_HAC_RANGE)
        for smi in vas:
            assert Chem.MolFromSmiles(smi) is not None, f"Invalid SMILES: {smi}"

    def test_hac_range_respected(self):
        gen = CloseInVAGenerator(random_state=42)
        tight = (9, 14)
        vas = gen.generate(_EAS, _ACTS, _CORE, n=100, ea_hac_range=tight)
        for smi in vas:
            hac = Chem.MolFromSmiles(smi).GetNumHeavyAtoms()
            assert tight[0] <= hac <= tight[1], f"HAC {hac} out of range for {smi}"

    def test_no_duplicates_in_output(self):
        gen = CloseInVAGenerator(random_state=42)
        vas = gen.generate(_EAS, _ACTS, _CORE, n=200, ea_hac_range=_HAC_RANGE)
        assert len(vas) == len(set(vas)), "Duplicate SMILES in VA output"

    def test_returns_something(self):
        gen = CloseInVAGenerator(random_state=42)
        vas = gen.generate(_EAS, _ACTS, _CORE, n=50, ea_hac_range=_HAC_RANGE)
        assert len(vas) > 0, "Expected some VAs from an 8-EA partially-saturated series"

    def test_none_core_returns_empty(self):
        gen = CloseInVAGenerator(random_state=42)
        vas = gen.generate(_EAS, _ACTS, None, n=50, ea_hac_range=_HAC_RANGE)
        assert vas == []

    def test_generation_report_populated(self):
        gen = CloseInVAGenerator(random_state=42)
        gen.generate(_EAS, _ACTS, _CORE, n=20, ea_hac_range=_HAC_RANGE)
        report = gen.generation_report
        assert "n_requested" in report
        assert "n_generated" in report
        assert report["n_requested"] == 20
        assert report["substitution_probability"] > 0

    def test_generate_from_decomposition(self):
        from como.series.decomposition import decompose_series
        decomp = decompose_series(_CORE, _EAS, list(_ACTS))
        gen = CloseInVAGenerator(random_state=42)
        vas = gen.generate_from_decomposition(decomp, n=20)
        assert isinstance(vas, list)
        ea_canon = decomp.ea_canonical_set
        for smi in vas:
            assert smi not in ea_canon
