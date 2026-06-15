"""Close-in VA generator: H-aware random sampling over EA-derived substituents."""

from __future__ import annotations

import warnings

import numpy as np
from rdkit import Chem
from rdkit.Chem import ReplaceCore, GetMolFrags, RWMol

from .base import VAGenerator
from ..series.assembly import _assemble_core_plus_frags


# ---------------------------------------------------------------------------
# Core decomposition helpers (shared with free_wilson.py and diverse.py)
# ---------------------------------------------------------------------------

def _prep_fragment(frag_mol: Chem.Mol) -> tuple[str | None, int]:
    """Clean a fragment from GetMolFrags for storage and reassembly.

    Each fragment has one dummy atom (*) whose isotope = the core atom index
    it was attached to (set by ReplaceCore(..., labelByIndex=True)).

    Returns:
        fragment_smiles: SMILES with map :1 on the attachment neighbor (or None)
        core_attach_idx: the core atom index this fragment connects to (-1 on failure)
    """
    rw = RWMol(frag_mol)
    attach_idx = -1
    dummy_idx = -1

    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 0:        # dummy atom
            attach_idx = atom.GetIsotope()  # = core atom index
            dummy_idx = atom.GetIdx()
            for nbr in atom.GetNeighbors():
                nbr.SetAtomMapNum(1)         # mark attachment neighbor with :1
            break

    if dummy_idx < 0 or attach_idx < 0:
        return None, -1

    rw.RemoveAtom(dummy_idx)
    try:
        Chem.SanitizeMol(rw)
        return Chem.MolToSmiles(rw.GetMol()), attach_idx
    except Exception:
        return None, -1


def _has_fused_match(mol: Chem.Mol, core_mol: Chem.Mol) -> bool:
    """Return True if every match of core_mol onto mol cuts at least one ring bond."""
    core_query = Chem.MolFromSmarts(Chem.MolToSmarts(core_mol))
    if core_query is None:
        core_query = core_mol

    ri = mol.GetRingInfo()
    bond_ring_set: set[int] = set()
    for ring in ri.BondRings():
        bond_ring_set.update(ring)

    matches = mol.GetSubstructMatches(core_query)
    if not matches:
        return False

    for match in matches:
        match_atom_set = set(match)
        cuts_ring_bond = any(
            bond.GetIdx() in bond_ring_set
            for bond in mol.GetBonds()
            if ((bond.GetBeginAtomIdx() in match_atom_set)
                != (bond.GetEndAtomIdx() in match_atom_set))
        )
        if not cuts_ring_bond:
            return False
    return True


def _strip_exit_vectors(
    core_mol: Chem.Mol,
) -> tuple[Chem.Mol | None, frozenset[int]]:
    """Strip * (dummy) atoms from a core and return constrained-matching info."""
    dummy_indices = frozenset(
        a.GetIdx() for a in core_mol.GetAtoms() if a.GetAtomicNum() == 0
    )
    if not dummy_indices:
        return None, frozenset()

    ev_neighbors_orig: set[int] = set()
    for a in core_mol.GetAtoms():
        if a.GetAtomicNum() == 0:
            for nbr in a.GetNeighbors():
                ev_neighbors_orig.add(nbr.GetIdx())

    rw = Chem.RWMol(core_mol)
    for idx in sorted(dummy_indices, reverse=True):
        rw.RemoveAtom(idx)
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        return None, frozenset()
    stripped_mol = rw.GetMol()

    orig_to_stripped: dict[int, int] = {}
    shift = 0
    for orig_idx in range(core_mol.GetNumAtoms()):
        if orig_idx in dummy_indices:
            shift += 1
        else:
            orig_to_stripped[orig_idx] = orig_idx - shift

    valid_sites = frozenset(
        orig_to_stripped[i] for i in ev_neighbors_orig if i in orig_to_stripped
    )
    return stripped_mol, valid_sites


def _decompose_replacecore(
    core_smiles: str,
    ea_smiles: list[str],
) -> tuple[Chem.Mol | None, list[tuple[str, dict[int, str]]]]:
    """Decompose EAs into per-site fragments using ReplaceCore + GetMolFrags.

    Used only internally by FreeWilsonVAGenerator.generate() legacy shim.
    """
    core_mol = Chem.MolFromSmiles(core_smiles)
    if core_mol is None:
        core_mol = Chem.MolFromSmarts(core_smiles)
    if core_mol is None:
        warnings.warn(f"Could not parse core SMILES: {core_smiles!r}")
        return None, []

    stripped_mol, ev_sites = _strip_exit_vectors(core_mol)
    match_core = stripped_mol if stripped_mol is not None else core_mol
    has_ev = stripped_mol is not None

    rows: list[tuple[str, dict[int, str]]] = []
    n_fused_skipped = 0
    n_ev_rejected = 0
    for smi in ea_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        if _has_fused_match(mol, match_core):
            n_fused_skipped += 1
            continue

        sidechain_mol = ReplaceCore(mol, match_core, labelByIndex=True)
        if sidechain_mol is None:
            continue

        frags = GetMolFrags(sidechain_mol, asMols=True)
        if not frags:
            continue

        site_map: dict[int, str] = {}
        for frag in frags:
            frag_smi, attach_idx = _prep_fragment(frag)
            if frag_smi is not None and attach_idx >= 0:
                site_map[attach_idx] = frag_smi

        if not site_map:
            continue

        if has_ev and not site_map.keys() <= ev_sites:
            n_ev_rejected += 1
            continue

        rows.append((smi, site_map))

    if n_fused_skipped:
        import sys
        print(
            f"[COMO] Warning: {n_fused_skipped} EA(s) skipped — core match cuts a "
            f"fused ring bond (core too small or molecule has extra fused rings). "
            f"Consider using a larger core that includes the full fused system.",
            file=sys.stderr,
        )
    if n_ev_rejected:
        import sys
        print(
            f"[COMO] Warning: {n_ev_rejected} EA(s) rejected — substituents found "
            f"outside the declared exit vector (*) positions in the core.",
            file=sys.stderr,
        )

    return match_core, rows


def _assemble_from_site_frags(
    core_mol: Chem.Mol,
    site_frags: dict[int, str],
) -> str | None:
    """Reassemble a molecule from a core and per-site fragment SMILES."""
    try:
        rw = RWMol(core_mol)

        for core_attach_idx, frag_smi in site_frags.items():
            frag_mol = Chem.MolFromSmiles(frag_smi)
            if frag_mol is None:
                return None

            combined = Chem.CombineMols(rw, frag_mol)
            rw = RWMol(combined)

            end_atm = -1
            for atom in rw.GetAtoms():
                if atom.GetAtomMapNum() == 1:
                    end_atm = atom.GetIdx()
                    break
            if end_atm < 0:
                return None

            rw.GetAtomWithIdx(end_atm).SetNumExplicitHs(0)
            rw.GetAtomWithIdx(end_atm).SetNoImplicit(False)

            rw.AddBond(core_attach_idx, end_atm, Chem.BondType.SINGLE)
            for atom in rw.GetAtoms():
                atom.SetAtomMapNum(0)

        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _canonical(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol is not None else None


# ---------------------------------------------------------------------------
# CloseInVAGenerator
# ---------------------------------------------------------------------------

class CloseInVAGenerator(VAGenerator):
    """Close-in VA generator: random H-aware sampling over EA-derived substituents.

    Decorates all substitution sites on the core with randomly selected
    substituents extracted from the analog series. At each site, substituents
    are chosen with probability p_sub (AS-specific global substitution
    probability); sites are left as H/no-substituent with probability 1-p_sub.

    Uses a SeriesDecomposition as its primary input. The `generate()` method
    accepts raw SMILES and activities for drop-in compatibility with other
    VAGenerator subclasses, but internally calls decompose_series first.
    """

    strategy_name = "close_in"

    def __init__(
        self,
        random_state: int | None = None,
        max_attempts: int = 100_000,
        min_organic_substituents: int = 1,
    ) -> None:
        self.random_state = random_state
        self.max_attempts = max_attempts
        self.min_organic_substituents = min_organic_substituents
        self.generation_report: dict = {}

    def generate(
        self,
        ea_smiles: list[str],
        ea_activities: np.ndarray,
        core_smiles: str | None,
        n: int,
        ea_hac_range: tuple[int, int],
    ) -> list[str]:
        if core_smiles is None:
            warnings.warn("CloseInVAGenerator requires a core SMILES. Skipping.")
            return []

        from ..series.decomposition import decompose_series
        import dataclasses

        decomp = decompose_series(core_smiles, ea_smiles,
                                  ea_activities=list(ea_activities))
        if not decomp.ea_records or not decomp.site_list:
            return []

        # Override decomp's HAC range with the externally-provided one
        decomp = dataclasses.replace(decomp, ea_hac_range=ea_hac_range)
        return self._sample_from_decomp(decomp, n, self.random_state)

    def generate_from_decomposition(
        self,
        decomp,
        n: int = 1000,
        random_state: int | None = None,
    ) -> list[str]:
        """Generate close-in VAs from a pre-computed SeriesDecomposition.

        Preferred over `generate()` when the same decomp is shared with FW
        and scoring to avoid redundant decomposition.
        """
        rng_seed = random_state if random_state is not None else self.random_state

        if not decomp.ea_records or not decomp.site_list:
            self.generation_report = {
                "n_requested": n, "n_generated": 0,
                "n_attempts": 0, "n_invalid": 0,
                "n_duplicate": 0, "n_existing_analog": 0,
                "n_outside_hac_range": 0,
                "substitution_probability": decomp.substitution_probability,
                "probability_mode": "global",
                "random_state": rng_seed,
            }
            return []

        return self._sample_from_decomp(decomp, n, rng_seed)

    def _sample_from_decomp(self, decomp, n: int, random_state) -> list[str]:
        """H-aware random sampling from a SeriesDecomposition."""
        from ..series.assembly import assemble_series_member

        rng = np.random.default_rng(random_state)
        p_sub = decomp.substitution_probability
        site_list = decomp.site_list
        pool_arrays = {s: sorted(decomp.site_pools[s]) for s in site_list}
        ea_set = decomp.ea_canonical_set
        min_hac, max_hac = decomp.ea_hac_range

        results: list[str] = []
        seen: set[str] = set()
        attempts = 0
        n_invalid = 0
        n_duplicate = 0
        n_existing = 0
        n_outside_hac = 0
        stagnation = 0
        stagnation_limit = max(1000, n * 10)

        while len(results) < n and attempts < self.max_attempts and stagnation < stagnation_limit:
            attempts += 1
            site_map: dict[int, str | None] = {}
            for s in site_list:
                pool = pool_arrays[s]
                if pool and rng.random() < p_sub:
                    site_map[s] = pool[int(rng.integers(len(pool)))]
                else:
                    site_map[s] = None

            organic_count = sum(1 for v in site_map.values() if v is not None)
            if organic_count < self.min_organic_substituents:
                continue

            smi = assemble_series_member(decomp, site_map)
            if smi is None:
                n_invalid += 1
                stagnation += 1
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                n_invalid += 1
                stagnation += 1
                continue
            hac = mol.GetNumHeavyAtoms()
            if hac < min_hac or hac > max_hac:
                n_outside_hac += 1
                stagnation += 1
                continue
            if smi in ea_set:
                n_existing += 1
                stagnation += 1
                continue
            if smi in seen:
                n_duplicate += 1
                stagnation += 1
                continue
            seen.add(smi)
            results.append(smi)
            stagnation = 0

        self.generation_report = {
            "n_requested": n,
            "n_generated": len(results),
            "n_attempts": attempts,
            "n_invalid": n_invalid,
            "n_duplicate": n_duplicate,
            "n_existing_analog": n_existing,
            "n_outside_hac_range": n_outside_hac,
            "substitution_probability": p_sub,
            "probability_mode": "global",
            "random_state": random_state,
        }
        return results
