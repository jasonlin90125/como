from abc import ABC, abstractmethod

import numpy as np


class VAGenerator(ABC):
    """Abstract base class for all virtual analog (VA) generators."""

    strategy_name: str  # class-level label used as source column in output CSV

    @abstractmethod
    def generate(
        self,
        ea_smiles: list[str],
        ea_activities: np.ndarray,
        core_smiles: str | None,
        n: int,
        ea_hac_range: tuple[int, int],
    ) -> list[str]:
        """Return a list of canonical SMILES for the VA population.

        Args:
            ea_smiles: canonical SMILES of existing analogs (EAs)
            ea_activities: pActivity values aligned with ea_smiles
            core_smiles: scaffold SMILES; may be None if not detectable
            n: soft upper limit on returned VA count
            ea_hac_range: (min_hac - 3, max_hac + 3) size filter bounds

        Returns:
            List of valid, canonical, deduplicated SMILES that are not exact EAs.
        """
