"""Free-Wilson VA generator: enumerate missing substituent combinations."""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from rdkit import Chem

from .base import VAGenerator
from .close_in import _assemble_from_site_frags


@dataclass
class FWCandidate:
    """A Free-Wilson VA candidate discovered from one or more 2×2 matrices."""
    site_frags: dict[int, str]       # organic sites only (passed to assembler)
    fw_preds: list[float] = field(default_factory=list)
    supporting_ea_indices: list[tuple[int, int, int]] = field(default_factory=list)
    varying_sites_list: list[tuple[int, int]] = field(default_factory=list)

    @property
    def fw_pred_mean(self) -> float:
        return float(np.mean(self.fw_preds)) if self.fw_preds else float("nan")

    @property
    def fw_pred_std(self) -> float:
        return float(np.std(self.fw_preds)) if len(self.fw_preds) > 1 else 0.0

    @property
    def fw_pred_n(self) -> int:
        return len(self.fw_preds)


def _mean_activity(idx_list: list[int], activities: np.ndarray) -> float:
    """Return the mean activity for the given row indices, ignoring NaN."""
    vals = activities[idx_list]
    finite = vals[np.isfinite(vals)]
    if len(finite) == 0:
        return float("nan")
    return float(finite.mean())


def _discover_fw_candidates(
    row_site_maps: list[dict[int, str | None]],
    row_activities: np.ndarray,
    site_list: list[int],
    rows_for_canon: list[str],
) -> tuple[dict[tuple, FWCandidate], set[int], dict[str, list[float]]]:
    """Core FW matrix discovery algorithm.

    Returns:
        candidates_by_key: deduped FW VA candidates
        ea_in_nbh_indices: row indices that participate in ≥1 qualifying matrix
        fw_ea_predictions: {canonical_smiles: [predicted_pActivity, ...]}
    """
    candidates_by_key: dict[tuple, FWCandidate] = {}
    ea_in_nbh_indices: set[int] = set()
    fw_ea_predictions: dict[str, list[float]] = {}

    for site_a, site_b in combinations(site_list, 2):
        rest_sites = [s for s in site_list if s not in (site_a, site_b)]
        matrices: dict[tuple, dict[tuple, list[int]]] = defaultdict(lambda: defaultdict(list))

        for ea_idx, sm in enumerate(row_site_maps):
            ra = sm.get(site_a)
            rb = sm.get(site_b)
            context = tuple(sm.get(s) for s in rest_sites)
            matrices[context][(ra, rb)].append(ea_idx)

        for context, matrix in matrices.items():
            ra_vals = sorted(
                {k[0] for k in matrix},
                key=lambda x: (x is None, x or ""),
            )
            rb_vals = sorted(
                {k[1] for k in matrix},
                key=lambda x: (x is None, x or ""),
            )

            for ra1, ra2 in combinations(ra_vals, 2):
                for rb1, rb2 in combinations(rb_vals, 2):
                    corners = [(ra1, rb1), (ra1, rb2), (ra2, rb1), (ra2, rb2)]
                    present = [(c, matrix[c]) for c in corners if c in matrix]
                    missing = [c for c in corners if c not in matrix]

                    if len(present) == 4:
                        # All-four-corner: retrospective FW EA validation
                        # Hold each corner out; predict from the other three.
                        for held_c, held_rows in present:
                            same_a = [idx for c, rows_list in present
                                      for idx in rows_list
                                      if c[0] == held_c[0] and c != held_c]
                            same_b = [idx for c, rows_list in present
                                      for idx in rows_list
                                      if c[1] == held_c[1] and c != held_c]
                            base = [idx for c, rows_list in present
                                    for idx in rows_list
                                    if c[0] != held_c[0] and c[1] != held_c[1]]
                            if not same_a or not same_b or not base:
                                continue
                            # Average activities across duplicate corners
                            act_a = _mean_activity(same_a, row_activities)
                            act_b = _mean_activity(same_b, row_activities)
                            act_base = _mean_activity(base, row_activities)
                            if any(np.isnan([act_a, act_b, act_base])):
                                continue
                            pred = float(act_a + act_b - act_base)
                            for held_idx in held_rows:
                                canon_smi = _row_to_canon(rows_for_canon[held_idx])
                                if canon_smi:
                                    fw_ea_predictions.setdefault(canon_smi, []).append(pred)
                        for _, idx_list in present:
                            ea_in_nbh_indices.update(idx_list)
                        continue

                    if len(present) != 3 or len(missing) != 1:
                        continue

                    missing_ra, missing_rb = missing[0]

                    same_a_rows = same_b_rows = base_rows = None
                    for (r_a, r_b), idx_list in present:
                        if r_a == missing_ra:
                            same_a_rows = idx_list
                        elif r_b == missing_rb:
                            same_b_rows = idx_list
                        else:
                            base_rows = idx_list

                    if same_a_rows is None or same_b_rows is None or base_rows is None:
                        continue

                    # Average activities across duplicate corners (fixes arbitrary first-row)
                    act_a = _mean_activity(same_a_rows, row_activities)
                    act_b = _mean_activity(same_b_rows, row_activities)
                    act_base = _mean_activity(base_rows, row_activities)
                    if any(np.isnan([act_a, act_b, act_base])):
                        continue
                    fw_pred = float(act_a + act_b - act_base)

                    # Build site_frags from the same_a corner's site map
                    base_frags = {k: v for k, v in row_site_maps[same_a_rows[0]].items()
                                  if v is not None}
                    if missing_ra is not None:
                        base_frags[site_a] = missing_ra
                    elif site_a in base_frags:
                        del base_frags[site_a]
                    if missing_rb is not None:
                        base_frags[site_b] = missing_rb
                    elif site_b in base_frags:
                        del base_frags[site_b]

                    dedup_key = tuple(base_frags.get(s) for s in sorted(site_list))
                    if dedup_key not in candidates_by_key:
                        candidates_by_key[dedup_key] = FWCandidate(site_frags=base_frags)
                    candidates_by_key[dedup_key].fw_preds.append(fw_pred)
                    candidates_by_key[dedup_key].supporting_ea_indices.append(
                        (same_a_rows[0], same_b_rows[0], base_rows[0])
                    )
                    candidates_by_key[dedup_key].varying_sites_list.append((site_a, site_b))
                    for _, idx_list in present:
                        ea_in_nbh_indices.update(idx_list)

    return candidates_by_key, ea_in_nbh_indices, fw_ea_predictions


class FreeWilsonVAGenerator(VAGenerator):
    """Free-Wilson VA generator.

    Implements context-fixed 2×2 substituent matrices as described in the
    DeepCOMO paper.  For each pair of substitution sites and each fixed
    context of all remaining sites, a missing 4th corner is a FW VA candidate
    predicted by additivity.

    Duplicate corner entries use mean activity across all EAs at that corner,
    not the first-row arbitrary pick.

    EAs are counted as being in a FW neighborhood only when they participate
    in at least one qualifying 2×2 matrix (3 present + 1 missing, or all 4
    present for retrospective validation) — NOT by MMP graph degree alone.
    """

    strategy_name = "free_wilson"

    def __init__(self) -> None:
        self.fw_predictions: dict[str, float] = {}
        self.fw_pred_std: dict[str, float] = {}
        self.fw_pred_n: dict[str, int] = {}
        self.n_ea_in_fw_nbh: int = 0
        self.n_sites: int = 0
        self.n_unique_substituents: int = 0
        self.fw_ea_predictions: dict[str, list[float]] = {}
        self.fw_neighborhood_records: list[dict] = []

    def generate(
        self,
        ea_smiles: list[str],
        ea_activities: np.ndarray,
        core_smiles: str | None,
        n: int,
        ea_hac_range: tuple[int, int],
    ) -> list[str]:
        """VAGenerator interface shim: decompose and delegate to generate_from_decomposition."""
        if core_smiles is None:
            warnings.warn("FreeWilsonVAGenerator requires a core SMILES. Skipping.")
            return []

        from ..series.decomposition import decompose_series
        import dataclasses

        decomp = decompose_series(core_smiles, ea_smiles,
                                  ea_activities=list(ea_activities))
        # Apply externally-provided HAC range
        decomp = dataclasses.replace(decomp, ea_hac_range=ea_hac_range)
        return self.generate_from_decomposition(decomp)

    def generate_from_decomposition(
        self,
        decomp,
        n: int | None = None,
        hac_range: tuple[int, int] | None = None,
        paper_mode: bool = False,
    ) -> list[str]:
        """Generate FW VAs from a pre-computed SeriesDecomposition.

        When paper_mode=True, n is ignored (count is determined by the series).
        """
        from ..series.assembly import _assemble_core_plus_frags

        self._reset_state()

        if paper_mode and n is not None:
            import sys
            print(
                "[COMO] FW paper_mode: n is ignored — FW VA count is series-determined.",
                file=sys.stderr,
            )

        if not decomp.ea_records or not decomp.site_list:
            return []

        site_list = list(decomp.site_list)
        if len(site_list) < 2:
            return []

        self.n_sites = len(site_list)
        self.n_unique_substituents = sum(len(pool) for pool in decomp.site_pools.values())

        # Build row_site_maps and activities from the decomp records
        row_site_maps = [dict(r.site_map) for r in decomp.ea_records]
        row_activities = np.array([r.activity for r in decomp.ea_records], dtype=np.float64)
        rows_for_canon = [r.canonical_smiles for r in decomp.ea_records]
        ea_set = decomp.ea_canonical_set

        candidates_by_key, ea_in_nbh_indices, fw_ea_preds = _discover_fw_candidates(
            row_site_maps, row_activities, site_list, rows_for_canon,
        )

        self.n_ea_in_fw_nbh = len(ea_in_nbh_indices)
        self.fw_ea_predictions = fw_ea_preds

        min_hac, max_hac = hac_range if hac_range is not None else decomp.ea_hac_range

        return self._assemble_candidates(
            candidates_by_key, decomp.core_mol, ea_set, min_hac, max_hac, site_list,
        )

    def _reset_state(self) -> None:
        self.fw_predictions = {}
        self.fw_pred_std = {}
        self.fw_pred_n = {}
        self.n_ea_in_fw_nbh = 0
        self.n_sites = 0
        self.n_unique_substituents = 0
        self.fw_ea_predictions = {}
        self.fw_neighborhood_records = []

    def _assemble_candidates(
        self,
        candidates_by_key: dict,
        core_mol,
        ea_set: frozenset[str] | set[str],
        min_hac: int,
        max_hac: int,
        site_list: list[int],
    ) -> list[str]:
        results: list[str] = []
        seen: set[str] = set()

        for cand in candidates_by_key.values():
            smi = _assemble_from_site_frags(core_mol, cand.site_frags)
            if smi is None:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            hac = mol.GetNumHeavyAtoms()
            if hac < min_hac or hac > max_hac:
                continue
            canon = Chem.MolToSmiles(mol)
            if canon in seen or canon in ea_set:
                continue
            seen.add(canon)
            results.append(canon)
            self.fw_predictions[canon] = cand.fw_pred_mean
            self.fw_pred_std[canon] = cand.fw_pred_std
            self.fw_pred_n[canon] = cand.fw_pred_n

        return results



def _row_to_canon(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else None
