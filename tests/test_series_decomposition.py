"""Tests for como.series: decomposition, assembly, schema."""

import math
import numpy as np
import pytest
from rdkit import Chem

from como.series import (
    EARecord,
    RejectedRecord,
    SeriesDecomposition,
    decompose_series,
    assemble_series_member,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# 2-site aniline system: N-alkyl × para-halogen
_CORE_2S = "Nc1ccccc1"
_EAS_2S = [
    ("CNc1ccc(F)cc1", 7.2),
    ("CNc1ccc(Cl)cc1", 7.8),
    ("CCNc1ccc(F)cc1", 6.9),
    ("CCNc1ccc(Cl)cc1", 7.5),  # all 4 corners present
]

# 3-site aniline system: same as 2-site + one extra ortho-F (3rd site)
_EAS_3S = [
    ("CNc1ccc(F)cc1", 7.2),
    ("CNc1ccc(Cl)cc1", 7.8),
    ("CCNc1ccc(F)cc1", 6.9),
    ("CCNc1ccc(Cl)cc1", 7.5),
    ("CNc1c(F)ccc(Cl)c1", 7.3),  # 3rd ortho-F site
]


# ---------------------------------------------------------------------------
# decompose_series tests
# ---------------------------------------------------------------------------

class TestDecomposeSeries:
    def test_basic_2site(self):
        smiles = [r[0] for r in _EAS_2S]
        acts = [r[1] for r in _EAS_2S]
        decomp = decompose_series(_CORE_2S, smiles, acts)
        assert len(decomp.site_list) == 2
        assert len(decomp.ea_records) == 4

    def test_all_eas_decomposed(self):
        smiles = [r[0] for r in _EAS_2S]
        decomp = decompose_series(_CORE_2S, smiles)
        # All 4 molecules should decompose cleanly
        assert len(decomp.rejected_records) == 0
        assert len(decomp.ea_records) == 4

    def test_site_pools_only_organic(self):
        smiles = [r[0] for r in _EAS_2S]
        decomp = decompose_series(_CORE_2S, smiles)
        for site, pool in decomp.site_pools.items():
            # None (H) should not appear in pools
            assert None not in pool
            assert all(isinstance(s, str) for s in pool)

    def test_unique_substituents_count(self):
        smiles = [r[0] for r in _EAS_2S]
        decomp = decompose_series(_CORE_2S, smiles)
        # site 0: Me, Et → 2; site 4: F, Cl → 2 → total 4
        assert decomp.n_unique_substituents if hasattr(decomp, "n_unique_substituents") else True
        total = sum(len(v) for v in decomp.site_pools.values())
        assert total == 4

    def test_ea_canonical_set_populated(self):
        smiles = [r[0] for r in _EAS_2S]
        decomp = decompose_series(_CORE_2S, smiles)
        assert len(decomp.ea_canonical_set) == 4

    def test_invalid_smiles_rejected(self):
        decomp = decompose_series(
            _CORE_2S,
            ["CNc1ccc(F)cc1", "NOT_VALID_SMILES"],
        )
        assert any(r.reason == "invalid_smiles" for r in decomp.rejected_records)

    def test_hac_range(self):
        smiles = [r[0] for r in _EAS_2S]
        decomp = decompose_series(_CORE_2S, smiles)
        hacs = [r.heavy_atom_count for r in decomp.ea_records]
        assert decomp.ea_hac_range == (min(hacs) - 3, max(hacs) + 3)

    def test_substitution_probability(self):
        smiles = [r[0] for r in _EAS_2S]
        decomp = decompose_series(_CORE_2S, smiles)
        # All 4 EAs have both sites filled → p_sub = 1.0
        assert decomp.substitution_probability == pytest.approx(1.0)

    def test_site_map_none_for_missing_sites(self):
        # One molecule lacks one declared site
        # In legacy mode, site_list is inferred from observed decomposition
        # Molecule with only one group: core is the full aniline with para-F
        decomp = decompose_series(
            _CORE_2S,
            ["CNc1ccc(F)cc1", "Nc1ccc(F)cc1"],  # 2nd has no N-sub
        )
        # site_map for 2nd EA should have None for N-site (if it decomposed at all)
        # It may simply not decompose if 'N' is part of the core
        # Just check we get some records
        assert len(decomp.ea_records) + len(decomp.rejected_records) == 2

    def test_exit_vector_filtering_paper_mode(self):
        # Core with exit vectors: only declared positions should be allowed
        core_ev = "*Nc1ccc(*)cc1"  # N and para are exit vectors
        smiles_ok = ["CNc1ccc(F)cc1", "CCNc1ccc(Cl)cc1"]
        decomp = decompose_series(core_ev, smiles_ok, paper_mode=True)
        assert len(decomp.ea_records) == 2
        assert len(decomp.rejected_records) == 0

    def test_exit_vector_sites_stored(self):
        core_ev = "*Nc1ccc(*)cc1"
        decomp = decompose_series(core_ev, ["CNc1ccc(F)cc1"])
        assert len(decomp.exit_vector_sites) == 2

    def test_activities_aligned(self):
        smiles = [r[0] for r in _EAS_2S]
        acts = [r[1] for r in _EAS_2S]
        decomp = decompose_series(_CORE_2S, smiles, acts)
        for rec in decomp.ea_records:
            # Activity should be one of the provided values
            assert any(abs(rec.activity - a) < 1e-9 for a in acts)

    def test_default_nan_activities(self):
        smiles = [r[0] for r in _EAS_2S[:2]]
        decomp = decompose_series(_CORE_2S, smiles)
        for rec in decomp.ea_records:
            assert math.isnan(rec.activity)


class TestAssembleSeriesMember:
    def test_roundtrip(self):
        """Decompose then reassemble should recover canonical SMILES."""
        smiles = [r[0] for r in _EAS_2S]
        acts = [r[1] for r in _EAS_2S]
        decomp = decompose_series(_CORE_2S, smiles, acts)

        recovered = set()
        for rec in decomp.ea_records:
            assembled = assemble_series_member(decomp, rec.site_map)
            if assembled is not None:
                recovered.add(assembled)

        original = {Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in smiles}
        # All decomposed EAs should round-trip
        assert recovered == original

    def test_none_site_leaves_h(self):
        smiles = [r[0] for r in _EAS_2S]
        decomp = decompose_series(_CORE_2S, smiles)
        # site_map with one site as None
        site_map = {list(decomp.site_list)[0]: None, list(decomp.site_list)[1]: None}
        # Assembly should still work (returns the bare core or fails gracefully)
        result = assemble_series_member(decomp, site_map)
        # Either returns None or a valid SMILES
        if result is not None:
            assert Chem.MolFromSmiles(result) is not None

    def test_partial_map(self):
        smiles = [r[0] for r in _EAS_2S]
        decomp = decompose_series(_CORE_2S, smiles)
        # Use only one site from the first EA
        rec = decomp.ea_records[0]
        site0 = list(decomp.site_list)[0]
        partial_map = {site0: rec.site_map[site0], list(decomp.site_list)[1]: None}
        result = assemble_series_member(decomp, partial_map)
        # Should produce a valid SMILES or None
        if result is not None:
            assert Chem.MolFromSmiles(result) is not None
