"""Molecule standardization and activity alignment."""

from __future__ import annotations

from rdkit import Chem


def canonical_smiles(smi: str) -> str | None:
    """Return RDKit canonical SMILES or None if invalid."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def heavy_atom_count(smi: str) -> int | None:
    mol = Chem.MolFromSmiles(smi)
    return mol.GetNumHeavyAtoms() if mol is not None else None
