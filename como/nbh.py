import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


def adaptive_radius(ea_norm: np.ndarray, k: int = 3) -> float:
    """Compute adaptive NBH radius as the median k-NN distance among EAs.

    Operates in the already-normalized descriptor space. k=3 matches the
    paper's default for defining series-relevant chemical spacing.
    """
    n = ea_norm.shape[0]
    if n <= k:
        # Degenerate case: fall back to using all available neighbors
        k = max(1, n - 1)

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(ea_norm)
    dists, _ = nbrs.kneighbors(ea_norm)
    # dists[:, 0] == 0 (self); dists[:, 1:k+1] are the k nearest neighbors
    return float(np.median(dists[:, 1 : k + 1]))


def build_nbh(
    ea_norm: np.ndarray,
    va_norm: np.ndarray,
    r: float | None = None,
    k: int = 3,
) -> tuple[np.ndarray, float]:
    """Build the VA-to-EA neighborhood membership matrix.

    Returns:
        membership: bool array (n_va, n_ea) — True if va_j is within EA_i's sphere
        radius_used: the radius actually applied
    """
    if r is None:
        r = adaptive_radius(ea_norm, k)

    if va_norm.shape[0] == 0:
        return np.empty((0, ea_norm.shape[0]), dtype=bool), r

    dist_matrix = cdist(va_norm, ea_norm, metric="euclidean")  # (n_va, n_ea)
    membership = dist_matrix <= r
    return membership, r
