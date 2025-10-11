import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Optional

def min_pair_distance(
    points: np.ndarray,
    *,
    p: float = 2.0,
    eps: float = 0.0,
    ignore_zeros: bool = False,
) -> Tuple[float, Tuple[int, int]]:
    """
    Compute the minimum pairwise distance in a point cloud and the indices
    of the pair achieving it, using a KD-tree (SciPy cKDTree).

    Args:
        points: (N, d) array of points.
        p: Minkowski p-norm (1=Manhattan, 2=Euclidean, np.inf=Chebyshev).
        eps: Approximation parameter; k-NN distances within (1+eps) of true.
        ignore_zeros: If True, ignore 0 distances (e.g., exact duplicates).

    Returns:
        min_dist: The smallest distance found.
        (i_min, j_min): Indices in `points` forming the closest pair.

    Notes:
        Uses a single self-query with k=2 (self & nearest neighbor).
        For very high dimensions, KD-Tree efficiency may decrease (curse of dimensionality). :contentReference[oaicite:1]{index=1}
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2:
        raise ValueError("`points` must be (N, d) with N >= 2")

    tree = cKDTree(pts)                       # build: O(N log N)
    dist, idx = tree.query(pts, k=2, p=p, eps=eps)
    # dist[:,0] 是 0（自身），dist[:,1] 是最近邻距离
    nn_dist = dist[:, 1]
    nn_idx  = idx[:, 1]

    if ignore_zeros:
        mask = nn_dist > 0
        if not np.any(mask):
            # 所有点都是重复点
            return 0.0, (0, int(nn_idx[0]))
        candidates = np.where(mask)[0]
        j = candidates[np.argmin(nn_dist[mask])]
    else:
        j = int(np.argmin(nn_dist))

    i_min = j
    j_min = int(nn_idx[i_min])
    min_dist = float(nn_dist[i_min])

    # 规范化输出：较小索引在前
    if j_min < i_min:
        i_min, j_min = j_min, i_min

    return min_dist, (i_min, j_min)
