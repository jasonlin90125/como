"""Assemble a molecule from a core and a site_map.

Works with both the new SeriesDecomposition-based workflow and the legacy
_assemble_from_site_frags used inside the analogs/ generators.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import RWMol

from .schema import SeriesDecomposition


def assemble_series_member(
    decomp: SeriesDecomposition,
    site_map: dict[int, str | None],
) -> str | None:
    """Assemble a molecule from a core and per-site fragments.

    Parameters
    ----------
    decomp:
        The parent SeriesDecomposition (provides core_mol and site_list).
    site_map:
        {site_id -> fragment_smiles | None}
        None means leave that site as H (core atom gets implicit H).

    Returns
    -------
    Canonical SMILES, or None on any chemistry failure.
    """
    organic_sites = {k: v for k, v in site_map.items() if v is not None}
    return _assemble_core_plus_frags(decomp.core_mol, organic_sites)


def _assemble_core_plus_frags(
    core_mol: Chem.Mol,
    site_frags: dict[int, str],
) -> str | None:
    """Low-level: combine core_mol with fragment SMILES that carry :1 attachment markers.

    site_frags: {core_atom_idx: frag_smi_with_:1_on_attachment_neighbor}
    """
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
