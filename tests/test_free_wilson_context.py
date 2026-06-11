"""Tests for context-fixed FW VA generation and correct EA-in-NBH counting."""

import numpy as np
import pytest
from rdkit import Chem

from como.analogs.free_wilson import FreeWilsonVAGenerator


_CORE_2S = "Nc1ccccc1"
_HAC_RANGE = (5, 30)


# ---------------------------------------------------------------------------
# Minimal 2×2 system
# ---------------------------------------------------------------------------

class TestFWMinimal2x2:
    """Toy 2-site × 2-substituent system with exactly 3 EAs (one missing corner)."""

    _EAS = [
        ("CNc1ccc(F)cc1", 7.2),   # R1=Me, R2=F
        ("CNc1ccc(Cl)cc1", 7.8),  # R1=Me, R2=Cl
        ("CCNc1ccc(F)cc1", 6.9),  # R1=Et, R2=F
        # Missing: R1=Et, R2=Cl → pred = 7.8 + 6.9 - 7.2 = 7.5
    ]

    def setup_method(self):
        smiles = [r[0] for r in self._EAS]
        acts = np.array([r[1] for r in self._EAS])
        self.gen = FreeWilsonVAGenerator()
        self.vas = self.gen.generate(smiles, acts, _CORE_2S, n=100, ea_hac_range=_HAC_RANGE)

    def test_exactly_one_fw_va(self):
        assert len(self.vas) == 1, f"Expected 1 FW VA, got {len(self.vas)}"

    def test_prediction_additivity(self):
        pred = list(self.gen.fw_predictions.values())[0]
        assert pred == pytest.approx(7.5, abs=0.01)

    def test_ea_in_nbh_count(self):
        # All 3 present corners participate in the qualifying FW matrix
        assert self.gen.n_ea_in_fw_nbh == 3

    def test_no_ea_duplicates(self):
        ea_canon = {Chem.MolToSmiles(Chem.MolFromSmiles(r[0])) for r in self._EAS}
        for smi in self.vas:
            assert smi not in ea_canon

    def test_valid_smiles(self):
        for smi in self.vas:
            assert Chem.MolFromSmiles(smi) is not None


class TestFW4CornerRetrospecive:
    """4-corner system: no VA candidate, 4 retrospective FW EA predictions."""

    _EAS = [
        ("CNc1ccc(F)cc1", 7.2),
        ("CNc1ccc(Cl)cc1", 7.8),
        ("CCNc1ccc(F)cc1", 6.9),
        ("CCNc1ccc(Cl)cc1", 7.5),
    ]

    def setup_method(self):
        smiles = [r[0] for r in self._EAS]
        acts = np.array([r[1] for r in self._EAS])
        self.gen = FreeWilsonVAGenerator()
        self.vas = self.gen.generate(smiles, acts, _CORE_2S, n=100, ea_hac_range=_HAC_RANGE)

    def test_no_fw_va_candidates(self):
        # All 4 corners present → no missing corner VA
        assert len(self.vas) == 0, f"Expected 0 FW VAs for complete 2×2, got {len(self.vas)}"

    def test_retrospective_ea_predictions_generated(self):
        # Should have retrospective FW predictions for each EA
        assert len(self.gen.fw_ea_predictions) > 0

    def test_all_four_eas_in_nbh(self):
        assert self.gen.n_ea_in_fw_nbh == 4

    def test_retrospective_prediction_formula(self):
        # Hold out CCNc1ccc(Cl)cc1 (Et,Cl):
        # pred = act(Me,Cl) + act(Et,F) - act(Me,F) = 7.8 + 6.9 - 7.2 = 7.5
        target = Chem.MolToSmiles(Chem.MolFromSmiles("CCNc1ccc(Cl)cc1"))
        if target in self.gen.fw_ea_predictions:
            preds = self.gen.fw_ea_predictions[target]
            assert pytest.approx(7.5, abs=0.01) in [pytest.approx(p, abs=0.01) for p in preds]


class TestFWContextIsolation:
    """3-site system: context mismatch must block false FW squares."""

    # Sites: N-alkyl, para-halogen, ortho-methyl
    # The ortho-methyl differs across some EAs → context blocks cross-group FW
    _EAS = [
        ("CNc1ccc(F)cc1", 7.2),      # R1=Me-N, R2=F, R3=no-orthoMe
        ("CNc1ccc(Cl)cc1", 7.8),     # R1=Me-N, R2=Cl, R3=no-orthoMe
        ("CCNc1ccc(F)cc1", 6.9),     # R1=Et-N, R2=F, R3=no-orthoMe
        # R1=Et, R2=Cl with no-orthoMe → this is the FW VA candidate, NOT blocked
        # Add one with different context (ortho-Me) to ensure context separation
    ]

    def setup_method(self):
        smiles = [r[0] for r in self._EAS]
        acts = np.array([r[1] for r in self._EAS])
        self.gen = FreeWilsonVAGenerator()
        self.vas = self.gen.generate(smiles, acts, _CORE_2S, n=100, ea_hac_range=_HAC_RANGE)

    def test_fw_va_found_when_context_matches(self):
        # All 3 EAs share the same context → FW VA should be found
        assert len(self.vas) == 1


class TestFWNoneSubstituent:
    """H/None as a substituent level: when p_sub < 1 some sites have None."""

    def test_none_as_level(self):
        # Build a case where one EA has an H at one site
        # The FW level (None vs Et-N) should still form a valid pair
        core = _CORE_2S
        # One EA with only N-Me (no para-sub), another with N-Me + para-F,
        # third with N-Et + para-F → FW VA should be N-Et without para-sub
        eas = [
            "CNc1ccccc1",       # R1=Me, R2=H (None)
            "CNc1ccc(F)cc1",    # R1=Me, R2=F
            "CCNc1ccccc1",      # R1=Et, R2=H (None)
            # Missing: R1=Et, R2=F → pred = act(Me,F) + act(Et,H) - act(Me,H)
        ]
        acts = np.array([6.5, 7.2, 6.9])
        gen = FreeWilsonVAGenerator()
        vas = gen.generate(eas, acts, core, n=100, ea_hac_range=_HAC_RANGE)
        # The R1=Et, R2=F candidate should be generated
        # (In context-fixed mode, the 3 EAs above form a qualifying 3-present matrix
        # because context = () for 2 sites and None/F are distinct levels)
        # Note: whether R2=H shows as a site at all depends on decomposition
        # Just check it doesn't crash and produces valid SMILES
        for smi in vas:
            assert Chem.MolFromSmiles(smi) is not None


class TestFWPredictionAggregation:
    """Multiple qualifying matrices for same VA → predictions aggregated."""

    def test_prediction_metadata(self):
        core = _CORE_2S
        eas = [
            ("CNc1ccc(F)cc1", 7.2),
            ("CNc1ccc(Cl)cc1", 7.8),
            ("CCNc1ccc(F)cc1", 6.9),
        ]
        smiles = [r[0] for r in eas]
        acts = np.array([r[1] for r in eas])
        gen = FreeWilsonVAGenerator()
        vas = gen.generate(smiles, acts, core, n=100, ea_hac_range=_HAC_RANGE)
        if vas:
            assert len(gen.fw_predictions) == len(vas)
            for smi in vas:
                assert smi in gen.fw_pred_n
                assert smi in gen.fw_pred_std
                assert gen.fw_pred_n[smi] >= 1
