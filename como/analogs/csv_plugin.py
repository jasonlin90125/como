"""CSV plug-in VA generator: pass-through for generative model output."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import polars as pl
from rdkit import Chem

from .base import VAGenerator


class CSVPluginVAGenerator(VAGenerator):
    """Accept externally generated VAs from a CSV file.

    The CSV must contain a SMILES column. An optional activity column
    (e.g. predicted pActivity from a generative model) is also supported.

    This is the generative model hook: any model (RNN, diffusion, etc.) that
    produces a CSV of SMILES can be plugged in without any code changes.
    """

    strategy_name = "csv_plugin"

    def __init__(
        self,
        csv_path: str | Path,
        smiles_col: str = "smiles",
        activity_col: str | None = None,
    ) -> None:
        self._csv_path = Path(csv_path)
        self._smiles_col = smiles_col
        self._activity_col = activity_col
        self.external_activities: dict[str, float] = {}

    def generate(
        self,
        ea_smiles: list[str],
        ea_activities: np.ndarray,
        core_smiles: str | None,
        n: int,
        ea_hac_range: tuple[int, int],
    ) -> list[str]:
        self.external_activities = {}

        df = pl.read_csv(self._csv_path)
        if self._smiles_col not in df.columns:
            warnings.warn(
                f"CSVPluginVAGenerator: column {self._smiles_col!r} not found in "
                f"{self._csv_path}. Skipping."
            )
            return []

        raw_smiles = df[self._smiles_col].to_list()
        activities_raw: list[float | None] = []
        if self._activity_col and self._activity_col in df.columns:
            activities_raw = df[self._activity_col].to_list()
        else:
            activities_raw = [None] * len(raw_smiles)

        # Build EA canonical set for exclusion
        ea_set: set[str] = set()
        for s in ea_smiles:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                ea_set.add(Chem.MolToSmiles(mol))

        results: list[str] = []
        seen: set[str] = set()

        for smi, act in zip(raw_smiles, activities_raw):
            if smi is None:
                continue
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                continue
            hac = mol.GetNumHeavyAtoms()
            # Loose sanity filter only (not the tight EA-range filter)
            if hac < 5 or hac > 100:
                continue
            canon = Chem.MolToSmiles(mol)
            if canon in seen or canon in ea_set:
                continue
            seen.add(canon)
            results.append(canon)
            if act is not None:
                try:
                    self.external_activities[canon] = float(act)
                except (TypeError, ValueError):
                    pass

        return results[:n]
