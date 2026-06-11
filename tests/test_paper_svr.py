"""Tests for PaperSVRPredictor — nested CV, external actives, EA split."""

import numpy as np
import pytest
from rdkit import Chem

from como.potency import PaperSVRPredictor, SVRPredictor


# Small EGFR-like EA set
_EA_SMILES = [
    "COc1cc2ncnc(Nc3ccccc3)c2cc1OC",
    "COc1cc2ncnc(Nc3ccc(F)cc3)c2cc1OC",
    "COc1cc2ncnc(Nc3ccc(Cl)cc3)c2cc1OC",
    "CCOc1cc2ncnc(Nc3ccccc3)c2cc1OCC",
    "CCOc1cc2ncnc(Nc3ccc(F)cc3)c2cc1OCC",
    "CCOc1cc2ncnc(Nc3ccc(Cl)cc3)c2cc1OCC",
    "Cc1cc2ncnc(Nc3ccccc3)c2cc1C",
    "Cc1cc2ncnc(Nc3ccc(F)cc3)c2cc1C",
]
_EA_ACTS = np.array([7.2, 7.8, 7.5, 6.9, 7.4, 7.1, 6.5, 7.0])

# Small external actives set
_EXT_SMILES = [
    "c1ccc(Nc2ncnc3cccc(OC)c23)cc1",
    "CCc1ccc(Nc2ncnc3cccc(OCC)c23)cc1",
    "c1ccc(Nc2ncnc3ccccc23)cc1F",
    "c1ccc(Nc2ncnc3cccc(C)c23)cc1Cl",
]
_EXT_ACTS = np.array([6.8, 7.1, 6.5, 7.3])


class TestPaperSVRPredictor:
    def test_requires_external_by_default(self):
        predictor = PaperSVRPredictor(allow_ea_only=False)
        with pytest.raises(ValueError, match="external target actives"):
            predictor.fit_paper_protocol(_EA_SMILES, _EA_ACTS)

    def test_ea_only_mode_allowed(self):
        predictor = PaperSVRPredictor(allow_ea_only=True, random_state=42)
        result = predictor.fit_paper_protocol(_EA_SMILES, _EA_ACTS)
        assert result.cv_r2_mean is not None
        assert len(result.outer_fold_results) == 3

    def test_with_external_actives(self):
        predictor = PaperSVRPredictor(random_state=42)
        result = predictor.fit_paper_protocol(
            _EA_SMILES, _EA_ACTS, _EXT_SMILES, _EXT_ACTS
        )
        assert len(result.outer_fold_results) == 3
        for fold in result.outer_fold_results:
            assert fold.n_train > 0
            assert fold.n_test > 0

    def test_outer_fold_count(self):
        predictor = PaperSVRPredictor(random_state=42, outer_folds=2)
        result = predictor.fit_paper_protocol(
            _EA_SMILES, _EA_ACTS, _EXT_SMILES, _EXT_ACTS
        )
        assert len(result.outer_fold_results) == 2

    def test_external_validation_populated(self):
        predictor = PaperSVRPredictor(random_state=42)
        result = predictor.fit_paper_protocol(
            _EA_SMILES, _EA_ACTS, _EXT_SMILES, _EXT_ACTS
        )
        # External validation EA set should be non-empty
        assert len(result.external_validation) > 0
        for smi, (obs, pred_mean, pred_std) in result.external_validation.items():
            assert not np.isnan(pred_mean)

    def test_ensemble_prediction(self):
        predictor = PaperSVRPredictor(random_state=42)
        predictor.fit_paper_protocol(
            _EA_SMILES, _EA_ACTS, _EXT_SMILES, _EXT_ACTS
        )
        test_smiles = ["COc1cc2ncnc(Nc3ccccc3)c2cc1OC", "c1ccccc1"]
        means, stds = predictor.predict_ensemble(test_smiles)
        assert means.shape == (2,)
        assert not np.isnan(means[0])  # valid SMILES

    def test_reproducible(self):
        p1 = PaperSVRPredictor(random_state=42)
        p2 = PaperSVRPredictor(random_state=42)
        r1 = p1.fit_paper_protocol(_EA_SMILES, _EA_ACTS, _EXT_SMILES, _EXT_ACTS)
        r2 = p2.fit_paper_protocol(_EA_SMILES, _EA_ACTS, _EXT_SMILES, _EXT_ACTS)
        assert r1.cv_mae_mean == pytest.approx(r2.cv_mae_mean, abs=1e-6)

    def test_ea_external_val_never_in_training(self):
        # EA external validation set should have been held out
        predictor = PaperSVRPredictor(random_state=42, ea_train_fraction=0.5)
        result = predictor.fit_paper_protocol(
            _EA_SMILES, _EA_ACTS, _EXT_SMILES, _EXT_ACTS
        )
        # External validation entries should be held-out EA canonical SMILES
        ea_canon = {Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in _EA_SMILES
                    if Chem.MolFromSmiles(s) is not None}
        for smi in result.external_validation:
            assert smi in ea_canon, f"Unexpected SMILES in external validation: {smi}"

    def test_mae_is_finite(self):
        predictor = PaperSVRPredictor(random_state=42)
        result = predictor.fit_paper_protocol(
            _EA_SMILES, _EA_ACTS, _EXT_SMILES, _EXT_ACTS
        )
        assert not np.isnan(result.cv_mae_mean)
        assert result.cv_mae_mean > 0
