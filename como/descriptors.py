import warnings
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.preprocessing import StandardScaler

DESCRIPTOR_NAMES: tuple[str, ...] = (
    "MW", "LogP", "TPSA", "HBD", "HBA", "RotBonds", "Rings"
)


def compute_descriptors(
    smiles_list: list[str],
) -> tuple[np.ndarray, list[int]]:
    """Compute seven LO-relevant physicochemical descriptors for each SMILES.

    Returns:
        raw_matrix: float64 array of shape (n_valid, 7)
        valid_indices: original indices of successfully parsed SMILES
    """
    rows: list[list[float]] = []
    valid_indices: list[int] = []

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            warnings.warn(f"Invalid SMILES at index {i}: {smi!r}", stacklevel=2)
            continue
        rows.append([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcNumRings(mol),
        ])
        valid_indices.append(i)

    if not rows:
        return np.empty((0, 7), dtype=np.float64), valid_indices

    return np.array(rows, dtype=np.float64), valid_indices


def normalize_descriptors(
    ea_raw: np.ndarray,
    va_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on combined EA+VA population, return normalized arrays.

    The scaler is fitted on the combined population as per the COMO methodology,
    so that both populations share the same normalization reference.
    """
    if va_raw.shape[0] == 0:
        combined = ea_raw
    else:
        combined = np.vstack([ea_raw, va_raw])

    scaler = StandardScaler()
    combined_norm = scaler.fit_transform(combined)

    n_ea = ea_raw.shape[0]
    ea_norm = combined_norm[:n_ea]
    va_norm = combined_norm[n_ea:] if va_raw.shape[0] > 0 else np.empty((0, ea_raw.shape[1]))

    return ea_norm, va_norm, scaler
