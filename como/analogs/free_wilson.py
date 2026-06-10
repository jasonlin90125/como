"""Free-Wilson VA generator: enumerate missing substituent combinations."""

from __future__ import annotations

import warnings
from collections import defaultdict
from itertools import combinations

import networkx as nx
import numpy as np
from rdkit import Chem

from .base import VAGenerator
from .close_in import _assemble_from_site_frags, _decompose_replacecore


class FreeWilsonVAGenerator(VAGenerator):
    strategy_name = "free_wilson"

    def __init__(self) -> None:
        self.fw_predictions: dict[str, float] = {}
        self.n_ea_in_fw_nbh: int = 0
        self.n_sites: int = 0
        self.n_unique_substituents: int = 0
        self._mmp_graph: nx.Graph | None = None

    def generate(
        self,
        ea_smiles: list[str],
        ea_activities: np.ndarray,
        core_smiles: str | None,
        n: int,
        ea_hac_range: tuple[int, int],
    ) -> list[str]:
        self.fw_predictions = {}
        self.n_ea_in_fw_nbh = 0
        self.n_sites = 0
        self.n_unique_substituents = 0

        if core_smiles is None:
            warnings.warn("FreeWilsonVAGenerator requires a core SMILES. Skipping.")
            return []

        core_mol, rows = _decompose_replacecore(core_smiles, ea_smiles)
        if core_mol is None or not rows:
            return []

        # Gather all substitution sites present across EAs
        all_sites: set[int] = set()
        for _, site_map in rows:
            all_sites.update(site_map.keys())
        site_list = sorted(all_sites)

        if len(site_list) < 2:
            return []

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

        # row_data: list of {core_attach_idx: frag_smi}, aligned with ea_activities
        # (rows may be a subset of ea_smiles if some EAs don't match the core;
        #  we need activities aligned with rows, not ea_smiles)
        row_site_maps: list[dict[int, str]] = [sm for _, sm in rows]
        row_activities = self._align_activities(rows, ea_smiles, ea_activities)

        # Build MMP graph
        self._mmp_graph = self._build_mmp_graph(row_site_maps, site_list)
        ea_in_nbh = {node for node, deg in self._mmp_graph.degree() if deg > 0}
        self.n_ea_in_fw_nbh = len(ea_in_nbh)

        # Find FW VA candidates
        candidates = self._find_fw_candidates(row_site_maps, site_list, row_activities)

        min_hac, max_hac = ea_hac_range
        results: list[str] = []
        seen: set[str] = set()

        for cand in candidates:
            smi = _assemble_from_site_frags(core_mol, cand["site_frags"])
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

            preds = cand["fw_preds"]
            if preds:
                self.fw_predictions[canon] = float(np.mean(preds))

        return results

    @staticmethod
    def _align_activities(
        rows: list[tuple[str, dict]],
        ea_smiles: list[str],
        ea_activities: np.ndarray,
    ) -> np.ndarray:
        """Map each decomposed row back to its EA activity by canonical SMILES."""
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
        row_site_maps: list[dict[int, str]],
        site_list: list[int],
    ) -> nx.Graph:
        """Build MMP graph: edges connect EAs differing at exactly one substitution site."""
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

    def _find_fw_candidates(
        self,
        row_site_maps: list[dict[int, str]],
        site_list: list[int],
        row_activities: np.ndarray,
    ) -> list[dict]:
        """Find missing corners in 2×2 substituent submatrices.

        For each pair of sites (site_a, site_b), map (frag_a, frag_b) -> row index.
        If exactly 3 of 4 corners of a 2×2 submatrix are filled, the missing
        corner is a FW VA candidate with a predicted activity from additivity.
        """
        candidates: dict[tuple, dict] = {}

        for site_a, site_b in combinations(site_list, 2):
            # Build matrix: (frag_a, frag_b) -> list of row indices
            matrix: dict[tuple[str, str], list[int]] = defaultdict(list)
            for idx, sm in enumerate(row_site_maps):
                r_a = sm.get(site_a)
                r_b = sm.get(site_b)
                if r_a and r_b:
                    matrix[(r_a, r_b)].append(idx)

            r_a_vals = sorted({k[0] for k in matrix})
            r_b_vals = sorted({k[1] for k in matrix})

            for r_a1, r_a2 in combinations(r_a_vals, 2):
                for r_b1, r_b2 in combinations(r_b_vals, 2):
                    corners = [
                        (r_a1, r_b1),
                        (r_a1, r_b2),
                        (r_a2, r_b1),
                        (r_a2, r_b2),
                    ]
                    present = [(c, matrix[c]) for c in corners if c in matrix]
                    missing = [c for c in corners if c not in matrix]

                    if len(present) != 3 or len(missing) != 1:
                        continue

                    missing_r_a, missing_r_b = missing[0]

                    # Identify corners: same_a shares r_a, same_b shares r_b, base shares neither
                    same_a = same_b = base = None
                    for (r_a, r_b), idx_list in present:
                        if r_a == missing_r_a:
                            same_a = idx_list[0]
                        elif r_b == missing_r_b:
                            same_b = idx_list[0]
                        else:
                            base = idx_list[0]

                    if same_a is None or same_b is None or base is None:
                        continue

                    acts = row_activities[[same_a, same_b, base]]
                    if np.any(np.isnan(acts)):
                        continue

                    fw_pred = float(acts[0] + acts[1] - acts[2])

                    # Build full site_frags for this candidate:
                    # start from same_a's assignments, override site_a and site_b
                    base_frags = dict(row_site_maps[same_a])
                    base_frags[site_a] = missing_r_a
                    base_frags[site_b] = missing_r_b

                    dedup_key = tuple(
                        base_frags.get(s, "") for s in sorted(site_list)
                    )
                    if dedup_key not in candidates:
                        candidates[dedup_key] = {
                            "site_frags": base_frags,
                            "fw_preds": [],
                        }
                    candidates[dedup_key]["fw_preds"].append(fw_pred)

        return list(candidates.values())
