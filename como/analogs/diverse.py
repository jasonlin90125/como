"""Diverse VA generator: enumerate substituent combinations from a fragment library."""

from __future__ import annotations

import importlib.resources
import itertools
import random
import warnings
from pathlib import Path

import numpy as np
from rdkit import Chem

from .base import VAGenerator
from .close_in import _assemble_from_site_frags, _decompose_replacecore, _canonical


def _make_attachable_fragment(frag_canon: str) -> str | None:
    """Mark atom 0 of a fragment SMILES with map :1 for assembly.

    _assemble_from_site_frags expects fragment SMILES where exactly one atom
    carries AtomMapNum=1, which is the attachment point.
    """
    mol = Chem.MolFromSmiles(frag_canon)
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    rw.GetAtomWithIdx(0).SetAtomMapNum(1)
    try:
        Chem.SanitizeMol(rw)
        return Chem.MolToSmiles(rw.GetMol())
    except Exception:
        return None


class DiverseVAGenerator(VAGenerator):
    strategy_name = "diverse"

    def __init__(
        self,
        fragment_lib_path: str | Path | None = None,
        rng_seed: int = 42,
    ) -> None:
        self._fragment_lib_path = fragment_lib_path
        self._rng_seed = rng_seed

    def _load_fragments(self) -> list[str]:
        """Load and validate fragment SMILES from file or bundled library.

        Returns canonical SMILES (no attachment marks) for each valid fragment.
        """
        if self._fragment_lib_path is not None:
            lines = Path(self._fragment_lib_path).read_text().splitlines()
        else:
            try:
                ref = importlib.resources.files(__package__).joinpath("fragments.smi")
                lines = ref.read_text().splitlines()
            except Exception:
                here = Path(__file__).parent
                lines = (here / "fragments.smi").read_text().splitlines()

        fragments: list[str] = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            smi = line.split()[0]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            if mol.GetNumHeavyAtoms() > 13:
                continue
            canon = Chem.MolToSmiles(mol)
            fragments.append(canon)

        return list(dict.fromkeys(fragments))  # deduplicate preserving order

    def generate(
        self,
        ea_smiles: list[str],
        ea_activities: np.ndarray,
        core_smiles: str | None,
        n: int,
        ea_hac_range: tuple[int, int],
    ) -> list[str]:
        if core_smiles is None:
            warnings.warn("DiverseVAGenerator requires a core SMILES. Skipping.")
            return []

        core_mol, rows = _decompose_replacecore(core_smiles, ea_smiles)
        if core_mol is None or not rows:
            return []

        # Identify all substitution sites
        all_sites: set[int] = set()
        for _, site_map in rows:
            all_sites.update(site_map.keys())
        site_list = sorted(all_sites)

        if not site_list:
            return []

        fragments = self._load_fragments()
        if not fragments:
            warnings.warn("No valid fragments loaded for DiverseVAGenerator.")
            return []

        # Prepare attachable versions of each fragment (mark atom 0 with :1)
        attachable: list[str] = []
        for frag in fragments:
            af = _make_attachable_fragment(frag)
            if af is not None:
                attachable.append(af)

        if not attachable:
            warnings.warn("Could not prepare any attachable fragments.")
            return []

        ea_set = {c for s in ea_smiles if (c := _canonical(s)) is not None}
        min_hac, max_hac = ea_hac_range
        rng = random.Random(self._rng_seed)

        # Each site gets the same pool; build pool-per-site list for itertools.product
        pool_per_site = [attachable for _ in site_list]

        results: list[str] = []
        seen: set[str] = set()

        total_size = len(attachable) ** len(site_list)

        if total_size <= n * 5:
            for combo in itertools.product(*pool_per_site):
                site_frags = dict(zip(site_list, combo))
                smi = _assemble_from_site_frags(core_mol, site_frags)
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
                if len(results) >= n:
                    break
        else:
            attempts = 0
            max_attempts = n * 20
            while len(results) < n and attempts < max_attempts:
                combo = tuple(rng.choice(attachable) for _ in site_list)
                site_frags = dict(zip(site_list, combo))
                smi = _assemble_from_site_frags(core_mol, site_frags)
                attempts += 1
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

        return results
