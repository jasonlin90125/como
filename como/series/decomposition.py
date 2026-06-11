"""Decompose an analog series into a SeriesDecomposition object.

This is the canonical implementation consumed by all VA generators and the
scoring protocol. It replaces the ad-hoc _decompose_replacecore logic that
was scattered across close_in.py and free_wilson.py.
"""

from __future__ import annotations

import warnings

from rdkit import Chem
from rdkit.Chem import GetMolFrags, ReplaceCore, RWMol

from .schema import EARecord, RejectedRecord, SeriesDecomposition
from .standardize import canonical_smiles


# ---------------------------------------------------------------------------
# Internal helpers (also used by the legacy analogs/ shim)
# ---------------------------------------------------------------------------

def _strip_exit_vectors(
    core_mol: Chem.Mol,
) -> tuple[Chem.Mol | None, frozenset[int]]:
    """Strip * dummy atoms from a core and return constrained-matching info.

    Returns (stripped_mol, ev_sites) where ev_sites are the atom indices in
    the STRIPPED core that neighbored the removed * atoms.  Returns
    (None, frozenset()) when no * atoms are present.
    """
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

    rw = RWMol(core_mol)
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


def _has_fused_match(mol: Chem.Mol, core_mol: Chem.Mol) -> bool:
    """Return True when every substructure match of core_mol cuts a ring bond."""
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


def _extract_site_map(
    mol: Chem.Mol,
    match_core: Chem.Mol,
    all_sites: tuple[int, ...] | None,
) -> dict[int, str | None] | None:
    """Run ReplaceCore and return {site_id: fragment_smi | None}.

    Returns None if the molecule cannot be cleanly decomposed.
    all_sites: when provided, missing sites get None (H/no-substituent).
    """
    if _has_fused_match(mol, match_core):
        return None

    sidechain_mol = ReplaceCore(mol, match_core, labelByIndex=True)
    if sidechain_mol is None:
        return None

    frags = GetMolFrags(sidechain_mol, asMols=True)
    if not frags:
        return None

    site_map: dict[int, str | None] = {}
    for frag in frags:
        rw = RWMol(frag)
        attach_idx = -1
        dummy_idx = -1
        for atom in rw.GetAtoms():
            if atom.GetAtomicNum() == 0:
                attach_idx = atom.GetIsotope()
                dummy_idx = atom.GetIdx()
                for nbr in atom.GetNeighbors():
                    nbr.SetAtomMapNum(1)
                break
        if dummy_idx < 0 or attach_idx < 0:
            continue
        rw.RemoveAtom(dummy_idx)
        try:
            Chem.SanitizeMol(rw)
            frag_smi = Chem.MolToSmiles(rw.GetMol())
            site_map[attach_idx] = frag_smi
        except Exception:
            continue

    if not site_map:
        return None

    # Fill declared sites that are absent (H / no substituent) with None
    if all_sites:
        for s in all_sites:
            if s not in site_map:
                site_map[s] = None

    return site_map


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decompose_series(
    core_smiles: str,
    ea_smiles: list[str],
    ea_activities: list[float] | None = None,
    paper_mode: bool = False,
) -> SeriesDecomposition:
    """Decompose an analog series against a core scaffold.

    Parameters
    ----------
    core_smiles:
        Scaffold SMILES.  May contain * dummy atoms to declare exit vectors.
    ea_smiles:
        Input EA SMILES (will be canonicalized internally).
    ea_activities:
        Aligned pActivity values.  If None, all activities stored as nan.
    paper_mode:
        When True, site_list is the declared exit-vector positions only and
        molecules with off-exit-vector substituents are rejected.
        When False (legacy), site_list is inferred from observed decomposition.
    """
    import math

    if ea_activities is None:
        ea_activities = [math.nan] * len(ea_smiles)

    # --- Parse core ---
    core_mol = Chem.MolFromSmiles(core_smiles)
    if core_mol is None:
        core_mol = Chem.MolFromSmarts(core_smiles)
    if core_mol is None:
        raise ValueError(f"Cannot parse core SMILES: {core_smiles!r}")

    stripped_mol, ev_sites = _strip_exit_vectors(core_mol)
    match_core = stripped_mol if stripped_mol is not None else core_mol
    has_ev = stripped_mol is not None

    # In paper mode with exit vectors, site_list is the declared sites.
    # We'll finalize site_list after decomposition if not paper_mode.
    declared_sites: tuple[int, ...] | None = tuple(sorted(ev_sites)) if has_ev else None

    rejected: list[RejectedRecord] = []
    records: list[EARecord] = []

    for smi, act in zip(ea_smiles, ea_activities):
        canon = canonical_smiles(smi)
        if canon is None:
            rejected.append(RejectedRecord(input_smiles=smi, reason="invalid_smiles"))
            continue

        mol = Chem.MolFromSmiles(canon)

        if _has_fused_match(mol, match_core):
            rejected.append(RejectedRecord(input_smiles=smi, reason="fused_ring_mismatch"))
            continue

        site_map = _extract_site_map(mol, match_core, declared_sites)
        if site_map is None:
            rejected.append(RejectedRecord(input_smiles=smi, reason="core_no_match"))
            continue

        # In paper mode, reject molecules with substituents outside exit vectors
        if paper_mode and has_ev:
            organic_sites = {k for k, v in site_map.items() if v is not None}
            if not organic_sites <= ev_sites:
                rejected.append(RejectedRecord(input_smiles=smi, reason="off_exit_vector"))
                continue

        records.append(EARecord(
            input_smiles=smi,
            canonical_smiles=canon,
            activity=float(act),
            site_map=dict(site_map),
            heavy_atom_count=mol.GetNumHeavyAtoms(),
        ))

    # --- Determine site_list ---
    if declared_sites is not None:
        site_list = declared_sites
    else:
        all_observed: set[int] = set()
        for r in records:
            all_observed.update(k for k, v in r.site_map.items() if v is not None)
        site_list = tuple(sorted(all_observed))

    # --- Build site pools ---
    site_pools: dict[int, frozenset[str]] = {s: frozenset() for s in site_list}
    for r in records:
        for s in site_list:
            v = r.site_map.get(s)
            if v is not None:
                site_pools[s] = site_pools[s] | {v}

    unique_substituents: frozenset[str] = frozenset(
        frag for pool in site_pools.values() for frag in pool
    )

    # --- Substitution probabilities ---
    n_ea = len(records)
    n_sites = len(site_list)
    if n_ea > 0 and n_sites > 0:
        total_slots = n_ea * n_sites
        filled = sum(
            1 for r in records for s in site_list if r.site_map.get(s) is not None
        )
        p_sub_global = filled / total_slots
        p_sub_per_site: dict[int, float] = {}
        for s in site_list:
            filled_s = sum(1 for r in records if r.site_map.get(s) is not None)
            p_sub_per_site[s] = filled_s / n_ea
    else:
        p_sub_global = 1.0
        p_sub_per_site = {s: 1.0 for s in site_list}

    # --- HAC range ---
    hacs = [r.heavy_atom_count for r in records]
    ea_hac_range = (min(hacs) - 3, max(hacs) + 3) if hacs else (0, 100)

    # Build canonical set from core-only members (no rejected)
    ea_canonical_set = frozenset(r.canonical_smiles for r in records)

    # Warn about rejected molecules
    n_ev_rejected = sum(1 for r in rejected if r.reason == "off_exit_vector")
    n_fused = sum(1 for r in rejected if r.reason == "fused_ring_mismatch")
    n_no_match = sum(1 for r in rejected if r.reason == "core_no_match")
    import sys
    if n_fused:
        print(
            f"[COMO] Warning: {n_fused} EA(s) skipped — core match cuts a fused ring bond.",
            file=sys.stderr,
        )
    if n_ev_rejected:
        print(
            f"[COMO] Warning: {n_ev_rejected} EA(s) rejected — substituents outside declared exit vectors.",
            file=sys.stderr,
        )
    if n_no_match and paper_mode:
        print(
            f"[COMO] Warning: {n_no_match} EA(s) do not contain the core scaffold.",
            file=sys.stderr,
        )

    return SeriesDecomposition(
        core_smiles=Chem.MolToSmiles(match_core),
        core_mol=match_core,
        site_list=site_list,
        ea_records=tuple(records),
        ea_canonical_set=ea_canonical_set,
        site_pools=site_pools,
        unique_substituents=unique_substituents,
        substitution_probability=p_sub_global,
        site_substitution_probability=p_sub_per_site,
        ea_hac_range=ea_hac_range,
        rejected_records=tuple(rejected),
        exit_vector_sites=ev_sites,
    )
