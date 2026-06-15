"""Unit tests for VA generator strategies."""

import csv
import numpy as np
import pytest
from rdkit import Chem

from como.analogs.close_in import CloseInVAGenerator
from como.analogs.free_wilson import FreeWilsonVAGenerator
from como.analogs.diverse import DiverseVAGenerator
from como.analogs.csv_plugin import CSVPluginVAGenerator


# 2-site core with 3×3 grid minus one corner — close-in can find the 9th combo
_CORE_1R = "Nc1ccccc1"  # aniline, 2 sites
_EAS_1R = [
    "CNc1ccc(F)cc1",
    "CNc1ccc(Cl)cc1",
    "CNc1ccc(Br)cc1",
    "CCNc1ccc(F)cc1",
    "CCNc1ccc(Cl)cc1",
    "CCNc1ccc(Br)cc1",
    "CCCNc1ccc(F)cc1",
    "CCCNc1ccc(Cl)cc1",
    # missing: CCCNc1ccc(Br)cc1 -- close-in can generate it
]
_ACTS_1R = np.array([7.2, 7.8, 7.5, 6.9, 7.4, 7.1, 6.5, 7.0])

# 2-site scaffold for FW tests: N-substituted para-substituted anilines
# Missing corner: N-Et + Cl (FW VA candidate, pred = 7.8 + 6.9 - 7.2 = 7.5)
_CORE_2R = "Nc1ccccc1"  # aniline
_EAS_2R = [
    ("CNc1ccc(F)cc1", 7.2),    # R1=Me, R2=F
    ("CNc1ccc(Cl)cc1", 7.8),   # R1=Me, R2=Cl
    ("CCNc1ccc(F)cc1", 6.9),   # R1=Et, R2=F
    # R1=Et, R2=Cl is intentionally MISSING (FW VA candidate)
]
_SMILES_2R = [row[0] for row in _EAS_2R]
_ACTS_2R = np.array([row[1] for row in _EAS_2R])

_HAC_RANGE = (10, 50)


class TestCloseInVAGenerator:
    def test_returns_valid_smiles(self):
        gen = CloseInVAGenerator(random_state=42)
        vas = gen.generate(_EAS_1R, _ACTS_1R, _CORE_1R, n=10, ea_hac_range=_HAC_RANGE)
        for smi in vas:
            assert Chem.MolFromSmiles(smi) is not None, f"Invalid SMILES: {smi}"

    def test_no_ea_duplicates(self):
        gen = CloseInVAGenerator(random_state=42)
        vas = gen.generate(_EAS_1R, _ACTS_1R, _CORE_1R, n=10, ea_hac_range=_HAC_RANGE)
        ea_canon = {Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in _EAS_1R}
        for smi in vas:
            assert smi not in ea_canon, f"EA duplicate found in VAs: {smi}"

    def test_size_filter(self):
        gen = CloseInVAGenerator(random_state=42)
        tight_range = (12, 16)
        vas = gen.generate(_EAS_1R, _ACTS_1R, _CORE_1R, n=10, ea_hac_range=tight_range)
        for smi in vas:
            mol = Chem.MolFromSmiles(smi)
            hac = mol.GetNumHeavyAtoms()
            assert tight_range[0] <= hac <= tight_range[1], f"HAC {hac} out of range for {smi}"

    def test_no_core_returns_empty(self):
        gen = CloseInVAGenerator(random_state=42)
        vas = gen.generate(_EAS_1R, _ACTS_1R, None, n=100, ea_hac_range=_HAC_RANGE)
        assert vas == []

    def test_deduplication(self):
        gen = CloseInVAGenerator(random_state=42)
        vas = gen.generate(_EAS_1R, _ACTS_1R, _CORE_1R, n=5, ea_hac_range=_HAC_RANGE)
        assert len(vas) == len(set(vas)), "Duplicate SMILES in VA output"


class TestDiverseVAGenerator:
    def test_returns_valid_smiles(self):
        gen = DiverseVAGenerator()
        vas = gen.generate(_EAS_1R, _ACTS_1R, _CORE_1R, n=50, ea_hac_range=_HAC_RANGE)
        for smi in vas:
            assert Chem.MolFromSmiles(smi) is not None

    def test_bundled_fragments_loaded(self):
        gen = DiverseVAGenerator()
        frags = gen._load_fragments()
        assert len(frags) >= 100, f"Too few bundled fragments: {len(frags)}"

    def test_custom_fragment_lib(self, tmp_path):
        frag_file = tmp_path / "frags.smi"
        frag_file.write_text("F\nCl\nCC\nCCO\nc1ccccc1\n")
        gen = DiverseVAGenerator(fragment_lib_path=frag_file)
        frags = gen._load_fragments()
        assert len(frags) >= 4

    def test_no_ea_duplicates(self):
        gen = DiverseVAGenerator()
        vas = gen.generate(_EAS_1R, _ACTS_1R, _CORE_1R, n=50, ea_hac_range=_HAC_RANGE)
        ea_canon = {Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in _EAS_1R}
        for smi in vas:
            assert smi not in ea_canon


class TestFreeWilsonVAGenerator:
    def test_finds_missing_corner(self):
        gen = FreeWilsonVAGenerator()
        vas = gen.generate(_SMILES_2R, _ACTS_2R, _CORE_2R, n=100, ea_hac_range=_HAC_RANGE)
        assert len(vas) >= 1

    def test_fw_predictions_populated(self):
        gen = FreeWilsonVAGenerator()
        vas = gen.generate(_SMILES_2R, _ACTS_2R, _CORE_2R, n=100, ea_hac_range=_HAC_RANGE)
        if vas:
            assert len(gen.fw_predictions) > 0

    def test_fw_potency_additivity(self):
        gen = FreeWilsonVAGenerator()
        vas = gen.generate(_SMILES_2R, _ACTS_2R, _CORE_2R, n=100, ea_hac_range=_HAC_RANGE)
        # FW prediction: pAct(Et,Cl) ≈ pAct(Me,Cl) + pAct(Et,F) - pAct(Me,F)
        #                             = 7.8 + 6.9 - 7.2 = 7.5
        if gen.fw_predictions:
            for pred in gen.fw_predictions.values():
                assert pytest.approx(pred, abs=0.01) == 7.5

    def test_no_ea_duplicates(self):
        gen = FreeWilsonVAGenerator()
        vas = gen.generate(_SMILES_2R, _ACTS_2R, _CORE_2R, n=100, ea_hac_range=_HAC_RANGE)
        ea_canon = {Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in _SMILES_2R
                    if Chem.MolFromSmiles(s) is not None}
        for smi in vas:
            assert smi not in ea_canon

    def test_no_core_returns_empty(self):
        gen = FreeWilsonVAGenerator()
        vas = gen.generate(_SMILES_2R, _ACTS_2R, None, n=100, ea_hac_range=_HAC_RANGE)
        assert vas == []


class TestCSVPluginVAGenerator:
    def test_passthrough_smiles(self, tmp_path):
        csv_file = tmp_path / "plugin.csv"
        rows = [
            ("c1ccc(CC(=O)N)cc1",),
            ("c1ccc(CC(=O)NC)cc1",),
            ("c1ccc(CC(=O)NCC)cc1",),
        ]
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["smiles"])
            writer.writerows(rows)

        gen = CSVPluginVAGenerator(csv_file)
        vas = gen.generate(_EAS_1R, _ACTS_1R, None, n=100, ea_hac_range=_HAC_RANGE)
        assert len(vas) == 3

    def test_activity_column_parsed(self, tmp_path):
        csv_file = tmp_path / "plugin_act.csv"
        rows = [
            ("c1ccc(CC(=O)N)cc1", "7.5"),
            ("c1ccc(CC(=O)NC)cc1", "8.0"),
        ]
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["smiles", "pred_pAct"])
            writer.writerows(rows)

        gen = CSVPluginVAGenerator(csv_file, activity_col="pred_pAct")
        vas = gen.generate([], np.array([]), None, n=100, ea_hac_range=(5, 100))
        assert len(gen.external_activities) == 2
        for v in gen.external_activities.values():
            assert isinstance(v, float)

    def test_invalid_smiles_skipped(self, tmp_path):
        csv_file = tmp_path / "plugin_bad.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["smiles"])
            writer.writerows([("NOT_VALID",), ("c1ccccc1",)])

        gen = CSVPluginVAGenerator(csv_file)
        vas = gen.generate([], np.array([]), None, n=100, ea_hac_range=(5, 100))
        assert len(vas) == 1
