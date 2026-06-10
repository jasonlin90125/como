"""Close-in VA generator: enumerate substituent combinations observed in EAs."""

from __future__ import annotations

import itertools
import warnings

import numpy as np
from rdkit import Chem
from rdkit.Chem import ReplaceCore, GetMolFrags, RWMol

from .base import VAGenerator


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
    """Return True if every match of core_mol onto mol cuts at least one ring bond.

    Uses the same SMARTS-style matching that ReplaceCore uses internally.
    A cut bond is one where exactly one endpoint is in the core match; it is a
    ring bond if it appears in the molecule's SSSR bond list.  If all matches
    cut a ring bond the molecule cannot be cleanly R-group decomposed with this
    core — the caller should skip it and report.
    """
    # ReplaceCore matches with useChirality=False, useQueryQueryMatches=False;
    # replicate that here so we see the same matches it would use.
    core_query = Chem.MolFromSmarts(Chem.MolToSmarts(core_mol))
    if core_query is None:
        core_query = core_mol

    ri = mol.GetRingInfo()
    bond_ring_set: set[int] = set()
    for ring in ri.BondRings():
        bond_ring_set.update(ring)

    matches = mol.GetSubstructMatches(core_query)
    if not matches:
        return False  # no match at all — ReplaceCore will return None, handled upstream

    for match in matches:
        match_atom_set = set(match)
        cuts_ring_bond = any(
            bond.GetIdx() in bond_ring_set
            for bond in mol.GetBonds()
            if ((bond.GetBeginAtomIdx() in match_atom_set)
                != (bond.GetEndAtomIdx() in match_atom_set))
        )
        if not cuts_ring_bond:
            return False  # at least one clean match exists
    return True  # every match cuts a ring bond


def _strip_exit_vectors(
    core_mol: Chem.Mol,
) -> tuple[Chem.Mol | None, frozenset[int]]:
    """Strip * (dummy) atoms from a core and return the constrained-matching info.

    When exit vectors (*) are present in the core:
      - The * atoms are removed to produce a plain core suitable for ReplaceCore.
      - The returned frozenset contains the atom indices *in the stripped core*
        that neighbored the removed * atoms — these are the only valid substitution
        sites.  A molecule with fragments at any other site is rejected.

    When no * atoms are present, returns (None, frozenset()) — the caller should
    use the original core unchanged with no site-filtering.
    """
    dummy_indices = frozenset(
        a.GetIdx() for a in core_mol.GetAtoms() if a.GetAtomicNum() == 0
    )
    if not dummy_indices:
        return None, frozenset()

    # Neighbors of the * atoms are the declared exit-vector attachment points.
    ev_neighbors_orig: set[int] = set()
    for a in core_mol.GetAtoms():
        if a.GetAtomicNum() == 0:
            for nbr in a.GetNeighbors():
                ev_neighbors_orig.add(nbr.GetIdx())

    # Remove * atoms (highest index first to keep lower indices stable).
    rw = Chem.RWMol(core_mol)
    for idx in sorted(dummy_indices, reverse=True):
        rw.RemoveAtom(idx)
    try:
        Chem.SanitizeMol(rw)
    except Exception:
        return None, frozenset()
    stripped_mol = rw.GetMol()

    # Build orig_idx → stripped_idx mapping (every removed atom shifts later ones).
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

    Uses ReplaceCore(..., labelByIndex=True) so each dummy atom's isotope
    encodes the core atom index, giving consistent site labeling across all EAs.

    EAs where every substructure match of the core would cut a ring bond (fused
    ring mismatch) are skipped with a warning rather than crashing.

    When the core contains explicit exit vectors (*), the * atoms are stripped
    before matching so ReplaceCore sees a plain scaffold.  Only molecules whose
    substituents fall exclusively at the declared exit-vector positions are kept;
    any molecule with a substituent at a non-exit-vector core position is rejected.

    Returns:
        core_mol: parsed core molecule, stripped of * atoms if present (None on failure)
        rows: list of (original_smiles, {core_attach_idx: fragment_smi})
              one entry per successfully decomposed EA
    """
    core_mol = Chem.MolFromSmiles(core_smiles)
    if core_mol is None:
        core_mol = Chem.MolFromSmarts(core_smiles)
    if core_mol is None:
        warnings.warn(f"Could not parse core SMILES: {core_smiles!r}")
        return None, []

    # Strip exit vectors if present; use stripped core for all matching.
    stripped_mol, ev_sites = _strip_exit_vectors(core_mol)
    match_core = stripped_mol if stripped_mol is not None else core_mol
    has_ev = stripped_mol is not None  # True = apply site-filtering

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

        # When exit vectors are declared, reject molecules with substituents
        # at positions outside the declared * positions.
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
    """Reassemble a molecule from a core and per-site fragment SMILES.

    site_frags: {core_atom_idx: fragment_smi_with_:1_map_on_attachment_neighbor}

    Follows the blog.walters approach:
      for each (core_attach_idx, frag):
        CombineMols(running, frag)
        find atom with map :1 → end_atm
        AddBond(core_attach_idx, end_atm)
        clear all map numbers
    """
    try:
        rw = RWMol(core_mol)

        for core_attach_idx, frag_smi in site_frags.items():
            frag_mol = Chem.MolFromSmiles(frag_smi)
            if frag_mol is None:
                return None

            combined = Chem.CombineMols(rw, frag_mol)
            rw = RWMol(combined)

            # Find the fragment attachment atom (map :1)
            end_atm = -1
            for atom in rw.GetAtoms():
                if atom.GetAtomMapNum() == 1:
                    end_atm = atom.GetIdx()
                    break
            if end_atm < 0:
                return None

            # Clear explicit Hs before adding bond to avoid valence overflow
            # (ReplaceCore leaves an explicit H on heteroatom attachment neighbors)
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
    strategy_name = "close_in"

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

        core_mol, rows = _decompose_replacecore(core_smiles, ea_smiles)
        if core_mol is None or not rows:
            return []

        # Identify all substitution sites (core atom indices present in ≥1 EA)
        all_sites: set[int] = set()
        for _, site_map in rows:
            all_sites.update(site_map.keys())
        site_list = sorted(all_sites)

        if not site_list:
            return []

        # Build per-site pools from observed EA fragments
        pools: dict[int, set[str]] = {s: set() for s in site_list}
        for _, site_map in rows:
            for site, frag_smi in site_map.items():
                pools[site].add(frag_smi)

        ea_set = {c for s in ea_smiles if (c := _canonical(s)) is not None}
        min_hac, max_hac = ea_hac_range

        results: list[str] = []
        seen: set[str] = set()

        pool_lists = [sorted(pools[s]) for s in site_list]

        for combo in itertools.product(*pool_lists):
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

        return results
