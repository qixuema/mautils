import heapq
import logging
from typing import Optional

import numpy as np
from einops import rearrange, repeat
from scipy.interpolate import splprep, splev, interp1d
from scipy.ndimage import gaussian_filter1d

from qixuema.np_utils import check_nan_inf, normalize, safe_norm
from qixuema.o3d_utils import get_vertices_obb

logger = logging.getLogger(__name__)

_EPS = 1e-6

START_END = np.array(
    [[0.0, 0.0, 0.0],
     [0.54020254, -0.77711392, 0.32291667]]
)

START_END_R = np.array([
    [ 0.54020282, -0.77711348,  0.32291649],
    [ 0.77711348,  0.60790503,  0.1629285 ],
    [-0.32291649,  0.1629285 ,  0.93229779]
])


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _bbox_center(points: np.ndarray) -> np.ndarray:
    return (np.min(points, axis=0) + np.max(points, axis=0)) / 2


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _rotation_to_align(
    src_unit: np.ndarray,
    tgt_unit: np.ndarray,
    handle_collinear: bool = True,
    eps: float = _EPS,
):
    """Axis/angle that rotates src_unit onto tgt_unit. Returns None on collinear
    input when handle_collinear is False."""
    dot_product = float(np.dot(src_unit, tgt_unit))
    angle = float(np.arccos(np.clip(dot_product, -1.0, 1.0)))

    if np.abs(dot_product + 1) < eps:
        if not handle_collinear:
            return None
        aux = np.array([1.0, 0.0, 0.0])
        if np.allclose(src_unit, aux) or np.allclose(src_unit, -aux):
            aux = np.array([0.0, 1.0, 0.0])
        axis = np.cross(src_unit, aux)
        axis = axis / (np.linalg.norm(axis) + eps)
        angle = np.pi
    elif np.abs(dot_product - 1) < eps:
        if not handle_collinear:
            return None
        axis = np.array([0.0, 0.0, 1.0])
        angle = 0.0
    else:
        axis = np.cross(src_unit, tgt_unit)
        axis = axis / (np.linalg.norm(axis) + eps)

    return axis, angle


# ---------------------------------------------------------------------------
# basic geometry
# ---------------------------------------------------------------------------

def calculate_cosine_with_z_axis(vertex1, vertex2, vertex3):
    """Cosine between the plane normal (through three points) and the +Z axis."""
    A, B, C = np.asarray(vertex1), np.asarray(vertex2), np.asarray(vertex3)
    normal = np.cross(B - A, C - A)
    return float(np.dot(normalize(normal, _EPS), np.array([0.0, 0.0, 1.0])))


def is_counter_clockwise(points):
    dx = np.diff(points[:, 0])
    dy = np.diff(points[:, 1])
    cross_sum = np.sum(dx[:-1] * dy[1:] - dx[1:] * dy[:-1])
    return cross_sum > 0


def normalize_point_cloud(points, scale=2.0):
    """Fit points into a cube of side `scale`, preserving aspect ratio."""
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    range_max = np.max(max_vals - min_vals)
    mid_points = (max_vals + min_vals) / 2
    return (points - mid_points) * scale / range_max


def rotate_around_z(vertices, theta):
    """Rotate `vertices` around the Z axis by `theta` radians."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ])
    return vertices @ R.T


def remove_self_loops(edges):
    edges = np.asarray(edges)
    return edges[edges[:, 0] != edges[:, 1]]


def find_closed_loop(edges):
    """Trace a single closed loop given edges. Returns [] if the graph is not a
    single simple cycle (every vertex must have degree exactly 2)."""
    edges = remove_self_loops(edges)
    if len(edges) == 0:
        return []

    adj = {}
    for a, b in edges:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))

    if any(len(v) != 2 for v in adj.values()):
        return []

    start = int(edges[0, 0])
    prev, curr = start, int(edges[0, 1])
    loop = [start]
    while curr != start:
        loop.append(curr)
        a, b = adj[curr]
        nxt = b if a == prev else a
        prev, curr = curr, nxt
    return loop


# ---------------------------------------------------------------------------
# translation / scaling
# ---------------------------------------------------------------------------

def translate_to_origin(points):
    """Shift points so the bounding-box center sits at the origin."""
    return points - _bbox_center(points)


def move_vertices_to_coord(vertices, target_coord=0.01, axis=2):
    """Shift vertices so the min along `axis` equals `target_coord`. Does not
    mutate the input."""
    vertices = vertices.copy()
    vertices[:, axis] += target_coord - np.min(vertices[:, axis])
    return vertices


def scale_vertices_in_sphere(vertices, target_radius=1.0, center=None,
                             return_scale_factor_and_center=False):
    """Scale vertices to fit in a sphere of `target_radius`, keeping the center
    in place."""
    if center is None:
        center = _bbox_center(vertices)
    max_distance = np.max(np.linalg.norm(vertices - center, axis=1))
    scale_factor = target_radius / max_distance
    scaled_vertices = center + (vertices - center) * scale_factor
    if return_scale_factor_and_center:
        return scaled_vertices, (scale_factor, center)
    return scaled_vertices


def normalize_vertices_in_sphere(vertices, target_radius=1.0, center=None):
    """Like scale_vertices_in_sphere but re-centers the result at the origin."""
    if center is None:
        center = _bbox_center(vertices)
    max_distance = np.max(np.linalg.norm(vertices - center, axis=1))
    return (vertices - center) * (target_radius / max_distance)


def fit_vertices_to_unit_sphere(vertices):
    """Center on centroid (mean, not bbox) and scale to fit in the unit sphere."""
    centered = vertices - np.mean(vertices, axis=0)
    return centered / np.max(np.linalg.norm(centered, axis=1))


def transform_wf_to_ground(vertices, center=None):
    vertices = translate_to_origin(vertices)
    vertices = move_vertices_to_coord(vertices, 0.01)
    return scale_vertices_in_sphere(vertices, 1.0, center)


def judge_flattened_objects(points, aspect_ratio_threshold=5.0, return_indices=False):
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    dimensions = max_vals - min_vals
    longest_edge = np.max(dimensions)
    shortest_edge = np.min(dimensions)
    aspect_ratio = longest_edge / shortest_edge

    if return_indices:
        return aspect_ratio, np.argmax(dimensions), np.argmin(dimensions)
    return aspect_ratio < aspect_ratio_threshold


# ---------------------------------------------------------------------------
# line-set cleanup
# ---------------------------------------------------------------------------

def remove_duplicate_vertices_and_lines(lineset: dict, return_indices=False,
                                        tolerance=0.0001, return_rows_changed=False):
    """Collapse near-duplicate vertices (tolerance-rounded) and duplicate lines.
    Line order is preserved via argsort of the first-occurrence indices."""
    vertices, lines = lineset['vertices'], lineset['lines']

    adjusted_points = np.round(vertices / tolerance) * tolerance
    unique_points, inverse_indices = np.unique(adjusted_points, axis=0, return_inverse=True)

    updated_lines = inverse_indices[lines]
    # Detect rows whose endpoints get swapped by the sort below.
    changed_rows = updated_lines[:, 0] > updated_lines[:, 1]
    updated_lines = np.sort(updated_lines, axis=1)

    unique_lines, indices = np.unique(updated_lines, axis=0, return_index=True)
    sorted_indices = np.argsort(indices)
    unique_lines = unique_lines[sorted_indices]

    lineset['vertices'] = unique_points
    lineset['lines'] = unique_lines

    if return_rows_changed:
        return lineset, indices, sorted_indices, changed_rows
    if return_indices:
        return lineset, indices, sorted_indices
    return lineset


def create_cube_lineset():
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float32)
    lines = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ])
    return {'vertices': vertices, 'lines': lines}


def remove_unused_vertices(lineset):
    """Drop vertices not referenced by any edge and re-index edges."""
    vertices = lineset['vertices']
    edges = lineset['lines']

    used, inverse = np.unique(edges, return_inverse=True)
    updated_edges = inverse.reshape(edges.shape)
    return {'vertices': vertices[used], 'lines': updated_edges}


# ---------------------------------------------------------------------------
# segment subdivision
# ---------------------------------------------------------------------------

def calculate_length(segment):
    return np.linalg.norm(segment[0] - segment[1])


def subdivide_segment(segment):
    midpoint = (segment[0] + segment[1]) / 2
    return np.array([segment[0], midpoint]), np.array([midpoint, segment[1]])


def subdivide_longest(segments, max_length=256):
    """Repeatedly bisect the longest segment until reaching `max_length`."""
    pq = [(-calculate_length(seg), i) for i, seg in enumerate(segments)]
    heapq.heapify(pq)

    active_segments = list(segments)
    active_mask = [True] * len(segments)
    active_count = len(segments)

    while active_count < max_length:
        _, longest_idx = heapq.heappop(pq)
        if not active_mask[longest_idx]:
            continue
        active_mask[longest_idx] = False
        active_count -= 1

        for seg in subdivide_segment(active_segments[longest_idx]):
            idx = len(active_segments)
            heapq.heappush(pq, (-calculate_length(seg), idx))
            active_segments.append(seg)
            active_mask.append(True)
            active_count += 1

    final_segments = [seg for seg, is_active in zip(active_segments, active_mask) if is_active]
    return np.stack(final_segments, axis=0)


# ---------------------------------------------------------------------------
# polyline transforms (single & batched)
# ---------------------------------------------------------------------------

def transform_polyline(points, handleCollinear=True):
    """Normalize a polyline so first and last points are [-1, 0, 0] and [1, 0, 0]."""
    out = transform_polyline_to_start_and_end(
        points,
        np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        handleCollinear=handleCollinear,
    )
    if out is None:
        return None

    # Snap endpoints to cancel accumulated numerical drift.
    out[:, 1] -= out[-1, 1]
    out[:, 2] -= out[-1, 2]
    out[0] = np.array([-1.0, 0.0, 0.0])
    out[-1] = np.array([1.0, 0.0, 0.0])
    return out


def transform_polyline_to_start_and_end(points, start_end, handleCollinear=True, eps=_EPS):
    start_point, end_point = start_end[0], start_end[1]

    translated_points = points - points[0]
    original_vector = translated_points[-1]
    target_vector = end_point - start_point

    if np.linalg.norm(original_vector) == 0:
        raise ValueError("Polyline start and end coincide; direction undefined.")

    original_norm = normalize(original_vector, eps)
    target_norm = normalize(target_vector, eps)

    rot = _rotation_to_align(original_norm, target_norm, handleCollinear, eps)
    if rot is None:
        return None
    axis, angle = rot

    R = _rodrigues(axis, angle)
    rotated_points = translated_points @ R.T

    original_length = float(np.linalg.norm(rotated_points[-1]) + eps)
    target_length = float(np.linalg.norm(target_vector) + eps)
    scaled_points = rotated_points * (target_length / original_length)

    return scaled_points + start_point


def inverse_transform_multi_polyline(transformed_polylines, start_ends, handleCollinear=True):
    edge_points = []
    for i, start_ends_i in enumerate(start_ends):
        if np.linalg.norm(start_ends_i[0] - start_ends_i[1]) == 0:
            continue

        edge_points_i = inverse_transform_polyline(transformed_polylines[i], start_ends_i, handleCollinear)
        if edge_points_i is None:
            return None
        edge_points.append(edge_points_i)

    return np.array(edge_points)


def inverse_transform_polyline(transformed_points, start_and_end, handleCollinear=True, epsilon=_EPS):
    tgt_start, tgt_end = start_and_end

    original_span = float(np.linalg.norm(transformed_points[-1] - transformed_points[0]))
    translated = transformed_points - transformed_points[0:1]

    tgt_direction = tgt_end - tgt_start
    target_scale = float(np.linalg.norm(tgt_direction))
    if target_scale == 0:
        raise ValueError("The start and end points are the same, so the scale factor cannot be determined.")

    scaled_back_points = translated * target_scale / (original_span + epsilon)

    pn_prime = scaled_back_points[-1]
    if np.linalg.norm(pn_prime) == 0:
        raise ValueError("The transformed polyline's end point is at the origin, so the direction cannot be determined.")

    pn_prime_norm = normalize(pn_prime, epsilon)
    target_norm = normalize(tgt_direction, epsilon)

    rot = _rotation_to_align(pn_prime_norm, target_norm, handleCollinear, epsilon)
    if rot is None:
        return None
    axis, angle = rot

    R = _rodrigues(axis, angle)
    rotated_back_points = scaled_back_points @ R.T

    return rotated_back_points + tgt_start


def denorm_curves(
    norm_curves: np.ndarray,
    corners: np.ndarray,
) -> Optional[np.ndarray]:
    """Use the given corners to denormalize the curves."""
    curves = []
    for i, corner in enumerate(corners):
        if np.linalg.norm(corner[0] - corner[1]) == 0:
            logger.warning(f"Corner {i} has zero length.")
            continue

        curve_i_temp = inverse_transform_polyline(norm_curves[i], start_and_end=START_END)
        curve_i = inverse_transform_polyline(curve_i_temp, start_and_end=corner)

        if curve_i is None:
            logger.warning(f"Curve {i} is None.")
            continue

        curves.append(curve_i)

    if curves:
        return np.stack(curves, axis=0)
    return None


# ---------------------------------------------------------------------------
# bounding boxes
# ---------------------------------------------------------------------------

def points_within_bounding_box(points, bbox):
    """Per-point boolean mask indicating membership in the axis-aligned `bbox`."""
    min_point, max_point = bbox
    within_min_bounds = np.all(points >= min_point, axis=1)
    within_max_bounds = np.all(points <= max_point, axis=1)
    return within_min_bounds & within_max_bounds


def get_bbox(points):
    """Tightest axis-aligned 3D bounding box for `points`."""
    return np.min(points, axis=0), np.max(points, axis=0)


def check_order_single_point(point1, point2):
    """Lexicographic (z, y, x) order. True when point1 is at-or-before point2."""
    if point1[2] < point2[2]:
        return True
    if point1[2] > point2[2]:
        return False
    if point1[1] < point2[1]:
        return True
    if point1[1] > point2[1]:
        return False
    return point1[0] <= point2[0]


# ---------------------------------------------------------------------------
# line-graph edges
# ---------------------------------------------------------------------------

def derive_line_edges_from_lines_np(lines):
    """Treat each line segment as a graph node and produce (i, j) pairs of
    segments that share exactly one vertex."""
    max_num_lines = lines.shape[0]
    line_edges_vertices_threshold = 1

    all_edges = np.stack(np.meshgrid(
        np.arange(max_num_lines),
        np.arange(max_num_lines),
        indexing='ij'), axis=-1)

    shared_vertices = rearrange(lines, 'i c -> i 1 c 1') == rearrange(lines, 'j c -> 1 j 1 c')
    num_shared_vertices = shared_vertices.any(axis=-1).sum(axis=-1)
    is_neighbor_line = num_shared_vertices == line_edges_vertices_threshold

    line_edge = all_edges[is_neighbor_line]
    line_edge = np.sort(line_edge, axis=1)
    line_edge, _ = np.unique(line_edge, return_inverse=True, axis=0)

    return line_edge


def edge_points_to_lineset(edge_points):
    sampled_points = edge_points

    num_points = sampled_points.shape[1]
    points_index = np.arange(num_points)
    lines = np.column_stack((points_index, np.roll(points_index, -1)))[:-1]
    bs = sampled_points.shape[0]
    lines = repeat(lines, 'nl c -> b nl c', b=bs)

    offset = np.arange(bs) * num_points
    offsets = rearrange(offset, 'b -> b 1 1')
    lines = lines + offsets
    lines = lines.reshape(-1, 2)
    vertices = sampled_points.reshape(-1, 3)

    return vertices, lines


# ---------------------------------------------------------------------------
# polyline lengths / sampling
# ---------------------------------------------------------------------------

# Picked so a (batch, 4, 3) float32 input stays under ~180 MB of working memory.
_POLYLINE_LENGTH_BATCH_SIZE = 15_000_000


def calculate_polyline_lengths(points: np.ndarray, batch_size: int = _POLYLINE_LENGTH_BATCH_SIZE) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)

    total_batches = points.shape[0] // batch_size + int(points.shape[0] % batch_size > 0)
    results = []
    for i in range(total_batches):
        batch_points = points[i * batch_size : (i + 1) * batch_size]
        results.append(calculate_polyline_lengths_single_batch(batch_points))
    return np.concatenate(results)


def calculate_polyline_lengths_single_batch(points: np.ndarray) -> np.ndarray:
    """Total length of each polyline in a batch of shape (B, N, 3)."""
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("Input must be a 3-D array of shape (batch_size, num_points, 3).")

    diffs = points[:, 1:, :] - points[:, :-1, :]
    distances = np.linalg.norm(diffs, axis=2)
    return distances.sum(axis=1)


# ---------------------------------------------------------------------------
# batched curve normalization (loop wrappers)
# ---------------------------------------------------------------------------

def normalize_edges_points(edge_points, handleCollinear=True, check_bbox=False):
    if check_nan_inf(edge_points):
        return None

    new_edge_points = np.zeros_like(edge_points)
    for i, edge_points_i in enumerate(edge_points):
        norm_edge_points_i = transform_polyline(edge_points_i, handleCollinear=handleCollinear)
        if check_nan_inf(norm_edge_points_i):
            return None

        if check_bbox:
            _, extent, _ = get_vertices_obb(norm_edge_points_i)
            if np.min(extent) > 0.7:
                return None

        new_edge_points[i] = norm_edge_points_i

    if check_nan_inf(new_edge_points):
        return None
    return new_edge_points


def transform_multi_edges_points(edge_points, start_end=None, handleCollinear=True, check_bbox=False):
    if start_end is None:
        return normalize_edges_points(edge_points, handleCollinear, check_bbox)

    if check_nan_inf(edge_points):
        return None

    new_edge_points = np.zeros_like(edge_points)
    for i, edge_points_i in enumerate(edge_points):
        new_edge_points_i = transform_polyline_to_start_and_end(edge_points_i, start_end[i], handleCollinear)
        if new_edge_points_i is None:
            return None
        if check_nan_inf(new_edge_points_i):
            return None
        new_edge_points[i] = new_edge_points_i

    if check_nan_inf(new_edge_points):
        return None
    return new_edge_points


def normalize_edge_points_two_stage(edge_points, check_bbox=False):
    num_lines = edge_points.shape[0]
    tgt_start_ends = repeat(START_END, 'nv d -> b nv d', b=num_lines)
    edge_points_middle_status = transform_multi_edges_points(edge_points, tgt_start_ends, handleCollinear=False)
    if edge_points_middle_status is None:
        return None
    return normalize_edges_points(edge_points_middle_status, handleCollinear=False, check_bbox=check_bbox)


# ---------------------------------------------------------------------------
# vectorized curve normalization
# ---------------------------------------------------------------------------

def normalize_curves_to_start_end(edge_points, start_end, eps=_EPS):
    translated_points = edge_points - edge_points[:, :1, :]

    original_vec = translated_points[:, -1, :]
    target_vec = start_end[:, 1, :] - start_end[:, 0, :]

    original_length = safe_norm(original_vec, eps)
    original_norm = original_vec / original_length
    target_length = safe_norm(target_vec, eps)
    target_norm = target_vec / target_length

    dot_product = np.einsum('ij,ij->i', original_norm, target_norm).clip(-1, 1)
    angle = np.arccos(dot_product)

    axis = np.cross(original_norm, target_norm)
    axis /= safe_norm(axis, eps)

    mask_reverse = np.abs(dot_product + 1) < eps
    mask_same = np.abs(dot_product - 1) < eps

    axis[mask_same] = np.array([0, 0, 1])
    axis[mask_reverse] = np.cross(original_norm[mask_reverse], np.array([1, 0, 0]))
    axis[mask_reverse] /= safe_norm(axis[mask_reverse], eps)
    angle[mask_reverse] = np.pi
    angle[mask_same] = 0

    # Batched Rodrigues.
    K = np.zeros((len(edge_points), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -axis[:, 2], axis[:, 1]
    K[:, 1, 0], K[:, 1, 2] = axis[:, 2], -axis[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -axis[:, 1], axis[:, 0]

    R = np.eye(3) + np.sin(angle)[:, None, None] * K + (1 - np.cos(angle))[:, None, None] * np.matmul(K, K)
    rotated_points = np.einsum('bij,bkj->bki', R, translated_points)

    scale = target_length / original_length
    scaled_points = rotated_points * scale[:, :, None]

    return scaled_points + start_end[:, :1, :]


def _normalize_curves_step_2(edge_points):
    """Assumes input is already normalized to START_END; transforms to [-1,0,0]→[1,0,0]."""
    R = repeat(START_END_R, 'n c -> b n c', b=edge_points.shape[0])
    rotated_points = np.einsum('bij,bkj->bki', R, edge_points)

    scaled_points = rotated_points * 2

    scaled_points[:, :, 1:] -= scaled_points[:, -1:, 1:]
    scaled_points -= np.array([1, 0, 0])
    scaled_points[:, 0] = [-1, 0, 0]
    scaled_points[:, -1] = [1, 0, 0]

    return scaled_points


def normalize_curves(edge_points):
    assert edge_points.ndim == 3
    num_curves = edge_points.shape[0]
    tgt_start_ends = np.tile(START_END, (num_curves, 1, 1))
    edge_points_middle_status = normalize_curves_to_start_end(edge_points, tgt_start_ends)
    return _normalize_curves_step_2(edge_points_middle_status)


# ---------------------------------------------------------------------------
# polyline sampling
# ---------------------------------------------------------------------------

def cumulative_polyline_length(polyline):
    """Per-vertex cumulative arc length of an open polyline of shape (N, 3)."""
    deltas = np.diff(polyline, axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    return np.concatenate(([0], np.cumsum(seg_lengths)))


def sample_polyline(polyline, step=0.02):
    """Resample a polyline of shape (N, 3) at arc-length spacing `step`."""
    cum_length = cumulative_polyline_length(polyline)
    total_length = cum_length[-1]

    num_samples = int(np.floor(total_length / step))
    sample_distances = np.linspace(0, num_samples * step, num_samples + 1)

    interp_fn = interp1d(cum_length, polyline, axis=0, kind='linear')
    return interp_fn(sample_distances)


def sample_polylines(polylines, step=0.02, max_points=2048):
    """Sample many polylines at arc-length `step` and return a (max_points, 3) cloud."""
    sampled_points_list = [sample_polyline(polylines[i], step) for i in range(polylines.shape[0])]
    all_sampled_points = np.vstack(sampled_points_list)

    valid_rows = ~np.isnan(all_sampled_points).any(axis=1)
    all_sampled_points = all_sampled_points[valid_rows]

    replace = all_sampled_points.shape[0] <= max_points
    random_idx = np.random.choice(all_sampled_points.shape[0], max_points, replace=replace)
    return all_sampled_points[random_idx]


def sample_points_from_mesh(mesh, num_samples=10000):
    points, face_indices = mesh.sample(num_samples, return_index=True)
    normals = mesh.face_normals[face_indices]
    return {'points': points, 'normals': normals}


def fit_bspline(points, n_samples=64, degree=3, smoothing=0.0, arc_length=False, fine_factor=10):
    """Fit a B-spline through `points` and sample `n_samples` points on it.

    Args:
    - points: (N, 3) input points
    - n_samples: number of output samples (including start and end)
    - degree: B-spline degree (default 3)
    - smoothing: smoothing factor s; 0 means interpolation
    - arc_length: sample by arc length rather than parameter-space
    - fine_factor: when arc_length=True, the subdivision factor (N * fine_factor)

    Returns:
    - samples: (n_samples, 3) sampled points
    """
    tck, _ = splprep(points.T, s=smoothing, k=degree)

    if arc_length:
        N = len(points)
        u_fine = np.linspace(0, 1, N * fine_factor)
        coords = np.vstack(splev(u_fine, tck)).T
        seg = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        cumlen = np.concatenate(([0], np.cumsum(seg)))
        u_new = np.interp(np.linspace(0, cumlen[-1], n_samples), cumlen, u_fine)
    else:
        u_new = np.linspace(0, 1, n_samples)

    return np.vstack(splev(u_new, tck)).T


def gaussian_smooth_curve(points, sigma=1.0):
    points = np.asarray(points, dtype=np.float32)
    points_smoothed = gaussian_filter1d(points, sigma=sigma, axis=-2)

    # Preserve original endpoints.
    points_smoothed[..., 0, :] = points[..., 0, :]
    points_smoothed[..., -1, :] = points[..., -1, :]
    return points_smoothed
