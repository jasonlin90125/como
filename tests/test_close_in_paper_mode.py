"""Tests for CloseInVAGenerator paper-mode random H-aware sampling."""

import numpy as np
import pytest
from rdkit import Chem

from como.analogs.close_in import CloseInVAGenerator


_CORE = "Nc1ccccc1"  # aniline, 2 sites
_EAS = [
    "CNc1ccc(F)cc1",
    "CNc1ccc(Cl)cc1",
    "CCNc1ccc(F)cc1",
    "CCNc1ccc(Cl)cc1",
]
_ACTS = np.array([7.2, 7.8, 6.9, 7.5])
_HAC_RANGE = (5, 30)


class TestCloseInPaperMode:
    def test_reproducible_with_same_seed(self):
        gen_a = CloseInVAGenerator(paper_mode=True, random_state=0)
        gen_b = CloseInVAGenerator(paper_mode=True, random_state=0)
        va_a = gen_a.generate(_EAS, _ACTS, _CORE, n=50, ea_hac_range=_HAC_RANGE)
        va_b = gen_b.generate(_EAS, _ACTS, _CORE, n=50, ea_hac_range=_HAC_RANGE)
        assert va_a == va_b

    def test_different_seeds_produce_different_results(self):
        gen_a = CloseInVAGenerator(paper_mode=True, random_state=1)
        gen_b = CloseInVAGenerator(paper_mode=True, random_state=2)
        va_a = gen_a.generate(_EAS, _ACTS, _CORE, n=100, ea_hac_range=_HAC_RANGE)
        va_b = gen_b.generate(_EAS, _ACTS, _CORE, n=100, ea_hac_range=_HAC_RANGE)
        # With 100 VAs and different seeds they should not be identical
        assert set(va_a) != set(va_b) or len(va_a) == 0

    def test_no_ea_duplicates(self):
        gen = CloseInVAGenerator(paper_mode=True, random_state=42)
        vas = gen.generate(_EAS, _ACTS, _CORE, n=200, ea_hac_range=_HAC_RANGE)
        ea_canon = {Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in _EAS}
        for smi in vas:
            assert smi not in ea_canon, f"EA duplicate in VAs: {smi}"

    def test_all_valid_smiles(self):
        gen = CloseInVAGenerator(paper_mode=True, random_state=42)
        vas = gen.generate(_EAS, _ACTS, _CORE, n=100, ea_hac_range=_HAC_RANGE)
        for smi in vas:
            assert Chem.MolFromSmiles(smi) is not None, f"Invalid SMILES: {smi}"

    def test_hac_range_respected(self):
        gen = CloseInVAGenerator(paper_mode=True, random_state=42)
        tight = (9, 12)
        vas = gen.generate(_EAS, _ACTS, _CORE, n=100, ea_hac_range=tight)
        for smi in vas:
            hac = Chem.MolFromSmiles(smi).GetNumHeavyAtoms()
            assert tight[0] <= hac <= tight[1], f"HAC {hac} out of range for {smi}"

    def test_no_duplicates_in_output(self):
        gen = CloseInVAGenerator(paper_mode=True, random_state=42)
        vas = gen.generate(_EAS, _ACTS, _CORE, n=200, ea_hac_range=_HAC_RANGE)
        assert len(vas) == len(set(vas)), "Duplicate SMILES in VA output"

    def test_returns_something(self):
        gen = CloseInVAGenerator(paper_mode=True, random_state=42)
        vas = gen.generate(_EAS, _ACTS, _CORE, n=50, ea_hac_range=_HAC_RANGE)
        assert len(vas) > 0, "Expected some VAs from a 2-site 4-EA series"

    def test_legacy_mode_unchanged(self):
        gen = CloseInVAGenerator(paper_mode=False)
        vas = gen.generate(_EAS, _ACTS, _CORE, n=200, ea_hac_range=_HAC_RANGE)
        # Legacy mode should still produce valid VAs
        assert len(vas) > 0
        for smi in vas:
            assert Chem.MolFromSmiles(smi) is not None

    def test_legacy_mode_deterministic(self):
        gen_a = CloseInVAGenerator(paper_mode=False)
        gen_b = CloseInVAGenerator(paper_mode=False)
        va_a = gen_a.generate(_EAS, _ACTS, _CORE, n=200, ea_hac_range=_HAC_RANGE)
        va_b = gen_b.generate(_EAS, _ACTS, _CORE, n=200, ea_hac_range=_HAC_RANGE)
        assert va_a == va_b

    def test_none_core_returns_empty(self):
        gen = CloseInVAGenerator(paper_mode=True, random_state=42)
        vas = gen.generate(_EAS, _ACTS, None, n=50, ea_hac_range=_HAC_RANGE)
        assert vas == []
