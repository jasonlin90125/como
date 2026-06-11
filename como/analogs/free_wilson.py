"""Free-Wilson VA generator: enumerate missing substituent combinations."""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

import networkx as nx
import numpy as np
from rdkit import Chem

from .base import VAGenerator
from .close_in import _assemble_from_site_frags, _decompose_replacecore


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


class FreeWilsonVAGenerator(VAGenerator):
    """Free-Wilson VA generator.

    Implements context-fixed 2×2 substituent matrices as described in the
    DeepCOMO paper.  For each pair of substitution sites and each fixed
    context of all remaining sites, a missing 4th corner is a FW VA candidate
    predicted by additivity.

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
        self._mmp_graph: nx.Graph | None = None
        # Retrospective FW EA predictions {canonical_smiles: list[float]}
        self.fw_ea_predictions: dict[str, list[float]] = {}

    def generate(
        self,
        ea_smiles: list[str],
        ea_activities: np.ndarray,
        core_smiles: str | None,
        n: int,
        ea_hac_range: tuple[int, int],
    ) -> list[str]:
        self.fw_predictions = {}
        self.fw_pred_std = {}
        self.fw_pred_n = {}
        self.n_ea_in_fw_nbh = 0
        self.n_sites = 0
        self.n_unique_substituents = 0
        self.fw_ea_predictions = {}

        if core_smiles is None:
            warnings.warn("FreeWilsonVAGenerator requires a core SMILES. Skipping.")
            return []

        core_mol, rows = _decompose_replacecore(core_smiles, ea_smiles)
        if core_mol is None or not rows:
            return []

        # Build unified site_list (organic sites only, like legacy)
        all_sites: set[int] = set()
        for _, site_map in rows:
            all_sites.update(k for k, v in site_map.items() if v is not None)
        site_list = sorted(all_sites)

        if len(site_list) < 2:
            return []

        # Compute stats consumed by scoring.py
        unique_subs_per_site = {
            s: {sm.get(s) for _, sm in rows if sm.get(s) is not None}
            for s in site_list
        }
        self.n_sites = len(site_list)
        self.n_unique_substituents = sum(len(v) for v in unique_subs_per_site.values())

        ea_set: set[str] = set()
        for s in ea_smiles:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                ea_set.add(Chem.MolToSmiles(mol))

        # Align activities with decomposed rows
        row_site_maps: list[dict[int, str | None]] = [sm for _, sm in rows]
        row_activities = self._align_activities(rows, ea_smiles, ea_activities)

        # MMP graph (for diagnostics; not used for EA-in-NBH counting)
        self._mmp_graph = self._build_mmp_graph(row_site_maps, site_list)

        # Context-fixed FW discovery
        candidates_by_key: dict[tuple, FWCandidate] = {}
        ea_in_nbh_indices: set[int] = set()

        for site_a, site_b in combinations(site_list, 2):
            rest_sites = [s for s in site_list if s not in (site_a, site_b)]
            # group rows by fixed context of all other sites
            matrices: dict[tuple, dict[tuple, list[int]]] = defaultdict(lambda: defaultdict(list))

            for ea_idx, sm in enumerate(row_site_maps):
                ra = sm.get(site_a)  # may be None (H)
                rb = sm.get(site_b)  # may be None (H)
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
                            # Retrospective FW EA: hold each corner out once
                            for held_c, held_rows in present:
                                same_a_rows = [idx for c, rows_list in present
                                               for idx in rows_list
                                               if c[0] == held_c[0] and c != held_c]
                                same_b_rows = [idx for c, rows_list in present
                                               for idx in rows_list
                                               if c[1] == held_c[1] and c != held_c]
                                base_rows = [idx for c, rows_list in present
                                             for idx in rows_list
                                             if c[0] != held_c[0] and c[1] != held_c[1]]
                                if not same_a_rows or not same_b_rows or not base_rows:
                                    continue
                                acts = row_activities[[same_a_rows[0], same_b_rows[0], base_rows[0]]]
                                if np.any(np.isnan(acts)):
                                    continue
                                pred = float(acts[0] + acts[1] - acts[2])
                                for held_idx in held_rows:
                                    canon_smi = _row_to_canon(rows[held_idx][0])
                                    if canon_smi:
                                        self.fw_ea_predictions.setdefault(canon_smi, []).append(pred)
                            # All 4 corners participate in qualifying FW neighborhoods
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

                        acts = row_activities[[same_a_rows[0], same_b_rows[0], base_rows[0]]]
                        if np.any(np.isnan(acts)):
                            continue
                        fw_pred = float(acts[0] + acts[1] - acts[2])

                        # Build site_frags for this candidate from same_a's context
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
                        # All 3 present corners participate
                        for _, idx_list in present:
                            ea_in_nbh_indices.update(idx_list)

        self.n_ea_in_fw_nbh = len(ea_in_nbh_indices)

        min_hac, max_hac = ea_hac_range
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

    @staticmethod
    def _align_activities(
        rows: list[tuple[str, dict]],
        ea_smiles: list[str],
        ea_activities: np.ndarray,
    ) -> np.ndarray:
        canon_to_act: dict[str, float] = {}
        for smi, act in zip(ea_smiles, ea_activities):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                canon_to_act[Chem.MolToSmiles(mol)] = float(act)

        aligned = []
        for orig_smi, _ in rows:
            mol = Chem.MolFromSmiles(orig_smi)
            canon = Chem.MolToSmiles(mol) if mol else orig_smi
            aligned.append(canon_to_act.get(canon, float("nan")))
        return np.array(aligned)

    def _build_mmp_graph(
        self,
        row_site_maps: list[dict[int, str | None]],
        site_list: list[int],
    ) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(range(len(row_site_maps)))
        for i, j in combinations(range(len(row_site_maps)), 2):
            diff_sites = [
                s for s in site_list
                if row_site_maps[i].get(s) != row_site_maps[j].get(s)
            ]
            if len(diff_sites) == 1:
                g.add_edge(i, j, site=diff_sites[0])
        return g


def _row_to_canon(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else None
