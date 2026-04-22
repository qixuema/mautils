"""
Adapted from https://github.com/thusiyuan/cooperative_scene_parsing/blob/master/utils/sunrgbd_utils.py
"""
import itertools

import networkx as nx
import numpy as np
import open3d as o3d
from einops import rearrange

from qixuema.geo_utils import (
    derive_line_edges_from_lines_np,
    find_closed_loop,
    remove_duplicate_vertices_and_lines,
    remove_unused_vertices,
    subdivide_longest,
)
from qixuema.helpers import get_rotaion_matrix_3d, get_rotation_matrix_2d
from qixuema.np_utils import is_close, normalize
from qixuema.line_utils import has_unreferenced_vertices, sort_line_set_vertices_and_lines


# ---------------------------------------------------------------------------
# camera / bbox projection
# ---------------------------------------------------------------------------

def parse_camera_info(camera_info, height, width):
    """Extract intrinsic and extrinsic matrices from a flat camera descriptor."""
    lookat = normalize(camera_info[3:6])
    up = normalize(camera_info[6:9])

    W = lookat
    U = np.cross(W, up)
    V = -np.cross(W, U)

    rot = np.vstack((U, V, W))
    trans = camera_info[:3]

    xfov = camera_info[9]
    yfov = camera_info[10]

    K = np.diag([1, 1, 1])
    K[0, 2] = width / 2
    K[1, 2] = height / 2
    K[0, 0] = K[0, 2] / np.tan(xfov)
    K[1, 1] = K[1, 2] / np.tan(yfov)

    return rot, trans, K


def flip_towards_viewer(normals, points):
    points = points / np.linalg.norm(points)
    proj = points.dot(normals[:2, :].T)
    flip = np.where(proj > 0)
    normals[flip, :] = -normals[flip, :]
    return normals


# Signed offsets (in basis-space) of the 8 corners of an axis-aligned box.
_BB3D_CORNER_SIGNS = np.array([
    [-1,  1,  1],
    [ 1,  1,  1],
    [ 1, -1,  1],
    [-1, -1,  1],
    [-1,  1, -1],
    [ 1,  1, -1],
    [ 1, -1, -1],
    [-1, -1, -1],
], dtype=np.float64)


def get_corners_of_bb3d(basis, coeffs, centroid):
    # Reorder basis so rows go X, Y, Z, then flip basis vectors towards viewer.
    index = np.argsort(np.abs(basis[:, 0]))[::-1]
    if index[2] != 2:
        index[1:] = index[1:][::-1]
    basis = basis[index, :]
    coeffs = coeffs[index]

    basis = flip_towards_viewer(basis, centroid)
    return get_corners_of_bb3d_no_index(basis, coeffs, centroid)


def get_corners_of_bb3d_no_index(basis, coeffs, centroid):
    scaled = basis * np.abs(coeffs)[:, None]
    return _BB3D_CORNER_SIGNS @ scaled + centroid


def project_3d_points_to_2d(points3d, R_ex, K):
    """Project 3d points from camera-centered coordinates to the 2D image plane."""
    points3d = R_ex.dot(points3d.T).T
    x3 = points3d[:, 0]
    y3 = -points3d[:, 1]
    z3 = np.abs(points3d[:, 2])
    xx = x3 * K[0, 0] / z3 + K[0, 2]
    yy = y3 * K[1, 1] / z3 + K[1, 2]
    return np.vstack((xx, yy))


def project_struct_bdb_to_2d(basis, coeffs, center, R_ex, K):
    """Project a 3D bounding box to a 2D axis-aligned image-space bbox dict."""
    corners3d = get_corners_of_bb3d(basis, coeffs, center)
    corners = project_3d_points_to_2d(corners3d, R_ex, K)
    return {
        'x1': int(max(np.min(corners[0, :]), 1)),
        'y1': int(max(np.min(corners[1, :]), 1)),
        'x2': int(min(np.max(corners[0, :]), 2 * K[0, 2])),
        'y2': int(min(np.max(corners[1, :]), 2 * K[1, 2])),
    }


# ---------------------------------------------------------------------------
# BFS over line-graphs
# ---------------------------------------------------------------------------

def bfs_on_each_component(graph, return_list=False):
    components = [sorted(s) for s in nx.connected_components(graph)]
    components = sorted(components, key=lambda s: s[0])

    node_seq = []
    node_seq_list = []
    for component in components:
        # nx.bfs_tree only reaches the node's own component, so no subgraph is needed.
        bfs_nodes = list(nx.bfs_tree(graph, component[0]).nodes())
        if return_list:
            node_seq_list.append(bfs_nodes)
        node_seq.extend(bfs_nodes)

    return node_seq_list if return_list else node_seq


def bfs_(edges, return_list=False):
    G = nx.Graph()
    G.add_edges_from(edges)
    return bfs_on_each_component(G, return_list=return_list)


# ---------------------------------------------------------------------------
# line-set dedup / merge (o3d LineSet API)
# ---------------------------------------------------------------------------

def _dedupe_o3d_lineset(line_set, sync_colors: bool, return_indices: bool):
    points = np.asarray(line_set.points)
    lines = np.asarray(line_set.lines)

    unique_points, inverse_indices = np.unique(points, axis=0, return_inverse=True)
    line_set.points = o3d.utility.Vector3dVector(unique_points)

    updated_lines = np.sort(inverse_indices[lines], axis=1)
    unique_lines, indices = np.unique(updated_lines, axis=0, return_index=True)
    line_set.lines = o3d.utility.Vector2iVector(unique_lines)

    if sync_colors and line_set.colors:
        line_set.colors = o3d.utility.Vector3dVector(np.asarray(line_set.colors)[indices])

    return (line_set, indices) if return_indices else (line_set, None)


def remove_duplicate_vertices_and_lines_of_lineset(line_set, return_indices=False):
    return _dedupe_o3d_lineset(line_set, sync_colors=False, return_indices=return_indices)


def remove_duplicate_vertices_and_lines_2d(line_set, return_indices=False):
    return _dedupe_o3d_lineset(line_set, sync_colors=True, return_indices=return_indices)


def remove_duplicate_vertices(lineset: dict):
    """Dedupe vertices by tolerance-rounded position and remap `lines`/`faces`."""
    keys = lineset.keys()
    key_1 = 'vertices'
    key_2 = 'lines' if 'lines' in keys else 'faces'

    vertices = lineset[key_1]
    lines = lineset[key_2]

    tolerance = 0.01
    adjusted_points = np.round(vertices / tolerance) * tolerance

    _, index, inverse_indices = np.unique(
        adjusted_points, axis=0, return_inverse=True, return_index=True
    )
    unique_points = vertices[index]
    updated_lines = inverse_indices[lines]

    output = {key_1: unique_points, key_2: updated_lines}
    if 'colors' in keys:
        output['colors'] = np.asarray(lineset['colors'])[index]
    return output


def remove_unconnected_vertices_and_lines(line_set: dict):
    vertices = line_set['vertices']
    lines = line_set['lines']

    used_points, _, inverse = np.unique(vertices, axis=0, return_index=True, return_inverse=True)

    updated_lines = np.sort(inverse[lines], axis=1)
    unique_lines, _ = np.unique(updated_lines, axis=0, return_index=True)

    line_set['vertices'] = used_points
    line_set['lines'] = unique_lines
    return line_set


def merge_line_sets(line_set1, line_set2, line_set3, dedupe=False):
    """Merge three o3d LineSets, re-indexing line endpoints."""
    points1 = np.asarray(line_set1.points)
    points2 = np.asarray(line_set2.points)
    points3 = np.asarray(line_set3.points)
    all_points = np.vstack((points1, points2, points3))

    lines1 = np.asarray(line_set1.lines)
    lines2 = np.asarray(line_set2.lines) + len(points1)
    lines3 = np.asarray(line_set3.lines) + len(points1) + len(points2)
    all_lines = np.vstack((lines1, lines2, lines3))

    colors1 = np.asarray(line_set1.colors)
    colors2 = np.asarray(line_set2.colors)
    colors3 = np.asarray(line_set3.colors)
    all_colors = np.vstack((colors1, colors2, colors3))

    new_line_set = o3d.geometry.LineSet()
    new_line_set.points = o3d.utility.Vector3dVector(all_points)
    new_line_set.lines = o3d.utility.Vector2iVector(all_lines)
    new_line_set.colors = o3d.utility.Vector3dVector(all_colors)

    if dedupe:
        new_line_set, _ = remove_duplicate_vertices_and_lines_of_lineset(new_line_set)
    return new_line_set


def merge_vertices_and_lines(line_sets: list):
    """Merge a list of dict linesets into one, re-indexing `lines`."""
    point_arrays = [np.asarray(ls['vertices']) for ls in line_sets]
    offsets = np.cumsum([0] + [len(p) for p in point_arrays[:-1]])

    all_points = np.concatenate(point_arrays, axis=0)
    all_lines = np.concatenate([
        np.asarray(ls['lines'], dtype=np.int32) + off
        for ls, off in zip(line_sets, offsets)
    ], axis=0)

    lineset = {'vertices': all_points, 'lines': all_lines}
    if 'colors' in line_sets[0]:
        lineset['colors'] = np.concatenate([np.asarray(ls['colors']) for ls in line_sets], axis=0)
    return lineset


def sort_line_set_vertices_and_lines_of_lineset(line_set, return_sorted_indices=False):
    points = np.asarray(line_set.points)
    lines = np.asarray(line_set.lines)

    sorted_indices = np.lexsort((points[:, 0], points[:, 1], points[:, 2]))
    sorted_points = points[sorted_indices]

    inverse_indices = np.argsort(sorted_indices)
    line_set.points = o3d.utility.Vector3dVector(sorted_points)

    updated_lines = inverse_indices[lines]
    sorted_line_indices = np.lexsort((updated_lines[:, 1], updated_lines[:, 0]))
    updated_lines = updated_lines[sorted_line_indices]
    line_set.lines = o3d.utility.Vector2iVector(updated_lines)

    return (line_set, sorted_line_indices) if return_sorted_indices else (line_set, None)


# ---------------------------------------------------------------------------
# BFS reordering wrappers
# ---------------------------------------------------------------------------

def bfs_line_set(line_set, is_door=False, return_line_seq=False):
    lines = np.asarray(line_set.lines)

    line_edges = derive_line_edges_from_lines_np(lines)
    line_seq = bfs_(line_edges)
    lines = lines[line_seq]

    if is_door:
        door_lines = lines.reshape(-1, 12, 2)
        points = np.asarray(line_set.points)
        for i in range(door_lines.shape[0]):
            vertex_idx = np.unique(door_lines[i])
            output_points, _ = expand_rectangle_np(points[vertex_idx])
            points[vertex_idx] = output_points
        line_set.points = o3d.utility.Vector3dVector(points)

    line_set.lines = o3d.utility.Vector2iVector(lines)
    return (line_set, line_seq) if return_line_seq else (line_set, None)


def bfs_lines(lines, return_bfs_list=False, return_line_seq=False):
    line_edges = derive_line_edges_from_lines_np(lines)

    if return_bfs_list:
        line_seq_list = bfs_(line_edges, return_list=True)
        line_seq = list(itertools.chain.from_iterable(line_seq_list))
        line_seq_lens = [len(sublist) for sublist in line_seq_list]
        return lines[line_seq], line_seq, line_seq_lens

    line_seq = bfs_(line_edges)
    lines = lines[line_seq]
    return (lines, line_seq) if return_line_seq else (lines, None)


def bfs_vertices(lineset, return_bfs_list=False, return_vertice_seq=False):
    vertices = np.asarray(lineset['vertices'])
    edges = np.asarray(lineset['lines'])

    if return_bfs_list:
        vertices_seq_list = bfs_(edges, return_list=True)
        vertices_seq = list(itertools.chain.from_iterable(vertices_seq_list))
        vertices_seq_lens = [len(sublist) for sublist in vertices_seq_list]
        return vertices[vertices_seq], vertices_seq, vertices_seq_lens

    vertices_seq = bfs_(edges)
    vertices = vertices[vertices_seq]
    return (vertices, vertices_seq) if return_vertice_seq else (vertices, None)


# ---------------------------------------------------------------------------
# geometric helpers
# ---------------------------------------------------------------------------

def expand_rectangle_np(vertices):
    x_coords = vertices[:, 0]
    y_coords = vertices[:, 1]

    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    x_range = x_max - x_min
    y_range = y_max - y_min

    if x_range > y_range:
        vertices[is_close(y_coords, y_min, 0.1), 1] -= 1
        vertices[is_close(y_coords, y_max, 0.1), 1] += 1
        direction = "x 轴"
    elif y_range > x_range:
        vertices[is_close(x_coords, x_min, 0.1), 0] -= 1
        vertices[is_close(x_coords, x_max, 0.1), 0] += 1
        direction = "y 轴"
    else:
        direction = "无需扩展，因为这是个正方形"

    return vertices, direction


def _rotations_for(points):
    if points.shape[-1] == 3:
        return get_rotaion_matrix_3d()
    if points.shape[-1] == 2:
        return get_rotation_matrix_2d()
    raise NotImplementedError("Only 2D and 3D points are supported.")


def rotate_and_flip(points):
    return rotate_and_flip_any_axis(points, axis=2)


def rotate_and_flip_any_axis(points, axis=2):
    rotation_all = _rotations_for(points)

    new_points = []
    flip_axis = (axis + 1) % 3

    for rotation in rotation_all:
        rotated_vertices = np.dot(points, rotation.T)
        new_points.append(rotated_vertices)

        flipped_vertices = rotated_vertices.copy()
        flipped_vertices[..., flip_axis] = -flipped_vertices[..., flip_axis]
        new_points.append(flipped_vertices)

    return new_points


def apply_transformation_to_point_cloud(point_cloud, transformation_matrix):
    return np.dot(point_cloud, transformation_matrix.T)


def get_transformed_vertex_list(vertices, transformation_matrix):
    return [apply_transformation_to_point_cloud(vertices, transformation)
            for transformation in transformation_matrix]


def slicing_lines_by_color(colors):
    """Run-length encode the quantized first color channel."""
    values = (colors[:, 0] * 10).astype(np.int32)
    change_idx = np.flatnonzero(np.diff(values)) + 1
    starts = np.concatenate(([0], change_idx))
    ends = np.concatenate((change_idx, [len(values)]))
    segments = np.stack([values[starts], ends - starts], axis=1)
    return segments


def find_closest_row(array, point):
    """Index of the row in `array` closest (Euclidean) to `point`."""
    # argmin is monotonic; skip the sqrt.
    return int(np.argmin(np.sum((array - np.asarray(point)) ** 2, axis=1)))


def _get_horizontal_plane(room_lineset, side: str, z_threshold_margin: float = 0.1):
    """Extract the horizontal plane at the top or bottom of a room lineset."""
    vertices = room_lineset['points']
    lines = np.asarray(room_lineset['lines'])
    z = vertices[:, 2]

    if side == 'bottom':
        threshold = np.min(z) + z_threshold_margin
        mask = (z[lines[:, 0]] < threshold) & (z[lines[:, 1]] < threshold)
    elif side == 'top':
        threshold = np.max(z) - z_threshold_margin
        mask = (z[lines[:, 0]] > threshold) & (z[lines[:, 1]] > threshold)
    else:
        raise ValueError(f"side must be 'bottom' or 'top', got {side!r}")

    selected = lines[mask]
    if len(selected) < 3:
        return {}

    plane_coords = vertices[selected].reshape(-1, 3)
    # remove_duplicate_vertices_and_lines uses the 'vertices' key, not 'points'.
    plane_lineset = {
        'vertices': plane_coords,
        'lines': np.arange(len(plane_coords)).reshape(-1, 2),
    }
    plane_lineset = remove_duplicate_vertices_and_lines(plane_lineset)

    loop = find_closed_loop(plane_lineset['lines'])
    if len(loop) < 3:
        return {}

    return {'points': plane_lineset['vertices'], 'faces': [loop]}


def get_bottom_plane(room_lineset):
    return _get_horizontal_plane(room_lineset, side='bottom')


def get_top_plane(room_lineset):
    return _get_horizontal_plane(room_lineset, side='top')


def shadow_follower_sort(A: list[int], B: list):
    """Sort A and reorder B to match. Returns two lists."""
    if len(A) != len(B):
        raise ValueError("A and B must have the same length.")
    sorted_indices = np.argsort(A)
    return np.array(A)[sorted_indices].tolist(), np.array(B)[sorted_indices].tolist()


def count_vertex_connections(vertices, edges):
    """Count how many edges reference each vertex."""
    return np.bincount(np.asarray(edges).ravel(), minlength=len(vertices))


# ---------------------------------------------------------------------------
# 10-feature line encoding / decoding
# ---------------------------------------------------------------------------

def line_10f_to_line_segments(sample):
    """Decode a (..., num_lines, 10) encoding into (..., num_lines, 6) segment endpoints."""
    p0 = sample[..., :3].copy()
    delta = sample[..., 3:6]
    mid_p = p0 + delta / 2
    idx = np.argmax(sample[..., 6:], axis=-1)

    p1 = np.zeros_like(p0)
    mask = idx == 0
    p1[mask] = p0[mask] + delta[mask]

    non_zero_mask = ~mask
    non_zero_delta = delta[non_zero_mask]
    non_zero_idx = idx[non_zero_mask] - 1
    non_zero_delta[np.arange(len(non_zero_delta)), non_zero_idx] *= -1

    non_zero_mid_p = mid_p[non_zero_mask]
    p0[non_zero_mask] = non_zero_mid_p - non_zero_delta / 2
    p1[non_zero_mask] = non_zero_mid_p + non_zero_delta / 2

    return np.concatenate([p0, p1], axis=-1)


def line_coords_to_line_10f(line_coords):
    """Encode (..., num_lines, 2, 3) line endpoints as a 10-feature descriptor."""
    min_p = np.min(line_coords, axis=-2)

    gt_line = line_coords[..., 1, :] - line_coords[..., 0, :]
    delta = np.abs(gt_line)

    # 4 candidate line directions: original + 3 single-axis sign flips.
    sign_mask = np.concatenate(
        [np.ones((1, 3)), 1 - 2 * np.eye(3)], axis=0  # shape (4, 3)
    )
    line_all = delta[..., None, :] * sign_mask       # (..., 4, 3)
    gt_line4 = gt_line[..., None, :]                 # (..., 1, 3)

    lines1_norm = np.linalg.norm(line_all, axis=-1)
    lines2_norm = np.linalg.norm(gt_line, axis=-1)[..., None]

    dot_product = np.sum(line_all * gt_line4, axis=-1)
    cosine_values = np.abs(dot_product / (lines1_norm * lines2_norm + 1e-8))

    return np.concatenate([min_p, delta, cosine_values], axis=-1)


# ---------------------------------------------------------------------------
# wireframe sampling
# ---------------------------------------------------------------------------

def sample_point_from_wf(wireframe, sample_distance: float = 0.003):
    """Sample points along every line segment of a wireframe dict. Skips zero-length
    and non-finite segments. Returns original vertices concatenated with samples."""
    if 'vertices' not in wireframe or 'lines' not in wireframe:
        raise ValueError("wireframe must contain 'vertices' and 'lines' keys")

    vertices = np.asarray(wireframe['vertices'])
    lines = np.asarray(wireframe['lines'])

    p0 = vertices[lines[:, 0]]
    p1 = vertices[lines[:, 1]]
    deltas = p1 - p0
    lengths = np.linalg.norm(deltas, axis=1)

    valid = np.isfinite(lengths) & (lengths > sample_distance + 0.001)
    if not np.any(valid):
        return vertices.copy()

    valid_p0 = p0[valid]
    valid_delta = deltas[valid]
    valid_lengths = lengths[valid]
    num_samples = np.maximum((valid_lengths / sample_distance).astype(int), 1)

    # np.linspace(0, 1, n, endpoint=False)[1:] drops the 0 sample, leaving n-1 points.
    sample_chunks = [
        valid_p0[i] + np.linspace(0, 1, n, endpoint=False)[1:, None] * valid_delta[i]
        for i, n in enumerate(num_samples)
    ]
    if not sample_chunks:
        return vertices.copy()

    sampled_points = np.concatenate(sample_chunks, axis=0)
    return np.vstack((vertices, sampled_points))


# ---------------------------------------------------------------------------
# lineset sorting pipelines
# ---------------------------------------------------------------------------

def sort_vertices_and_lines(lineset: dict, return_indices=False, multi_graph=False, tolerance=0.0001):
    lineset, indices, indices_2 = remove_duplicate_vertices_and_lines(lineset, return_indices=True)
    lineset, sorted_indices = sort_line_set_vertices_and_lines(
        lineset, return_sorted_indices=multi_graph, tolerance=tolerance,
    )

    if multi_graph:
        lineset['vertices'], vertice_seq, vertices_seq_lens = bfs_vertices(
            lineset, return_bfs_list=True, return_vertice_seq=True,
        )
    else:
        lineset['vertices'], vertice_seq = bfs_vertices(lineset, return_vertice_seq=True)

    lines = lineset['lines']
    inverse_indices = np.argsort(vertice_seq)

    updated_lines = np.sort(inverse_indices[lines], axis=1)
    sorted_line_indices = np.lexsort((updated_lines[:, 1], updated_lines[:, 0]))
    lineset['lines'] = updated_lines[sorted_line_indices]

    if multi_graph:
        indices = indices[indices_2][sorted_indices][sorted_line_indices]
        return lineset, indices, vertices_seq_lens

    if return_indices:
        indices = indices[indices_2][sorted_indices][vertice_seq]
        return lineset, indices

    return lineset


def sort_lineset(lineset: dict, return_indices=False, multi_graph=False):
    lineset, indices, indices_2 = remove_duplicate_vertices_and_lines(lineset, return_indices=True)
    lineset, sorted_indices = sort_line_set_vertices_and_lines(lineset, return_sorted_indices=multi_graph)

    if multi_graph:
        lineset['lines'], line_seq, line_seq_lens = bfs_lines(
            lines=lineset['lines'], return_bfs_list=True, return_line_seq=True,
        )
        indices = indices[indices_2][sorted_indices][line_seq]
        return lineset, indices, line_seq_lens

    lineset['lines'], line_seq = bfs_lines(lines=lineset['lines'], return_line_seq=True)

    if return_indices:
        indices = indices[indices_2][sorted_indices][line_seq]
        return lineset, indices

    return lineset


def subdivide_longest_high(lineset, max_length=256):
    segments = lineset['vertices'][lineset['lines']]
    segments = subdivide_longest(segments, max_length=max_length)

    vertices = rearrange(segments, 'nl nv nc -> (nl nv) nc')
    lines = np.arange(len(vertices)).reshape(-1, 2)

    lineset = {'vertices': vertices, 'lines': lines}
    lineset = remove_duplicate_vertices_and_lines(lineset)
    lineset, _ = sort_line_set_vertices_and_lines(lineset)

    lines, _ = bfs_lines(lines=lineset['lines'])
    lineset['lines'] = lines
    return lineset


def sort_vertices_and_curves(
    lineset: dict,
    edge_points: np.ndarray,
    multi_graph=False,
    tolerance=0.0001,
    return_rows_changed=False,
):
    lineset, indices_1, indices_2, rows_changed_1 = remove_duplicate_vertices_and_lines(
        lineset, return_indices=True, return_rows_changed=True,
    )

    lineset = remove_unused_vertices(lineset)
    if has_unreferenced_vertices(lineset):
        return None, None

    edge_points[rows_changed_1] = edge_points[rows_changed_1, ::-1]
    indices = indices_1[indices_2]
    edge_points = edge_points[indices]

    lineset, sorted_indices, rows_changed_2 = sort_line_set_vertices_and_lines(
        lineset,
        return_sorted_indices=multi_graph,
        tolerance=tolerance,
        return_rows_changed=True,
    )

    edge_points[rows_changed_2] = edge_points[rows_changed_2, ::-1]
    edge_points = edge_points[sorted_indices]

    if multi_graph:
        lineset['vertices'], vertice_seq, _ = bfs_vertices(
            lineset, return_bfs_list=True, return_vertice_seq=True,
        )
    else:
        lineset['vertices'], vertice_seq = bfs_vertices(lineset, return_vertice_seq=True)

    lines = lineset['lines']
    inverse_indices = np.argsort(vertice_seq)
    updated_lines = inverse_indices[lines]

    # Track which endpoint pairs get swapped by the inner sort.
    rows_changed_3 = updated_lines[:, 0] > updated_lines[:, 1]
    updated_lines = np.sort(updated_lines, axis=1)

    sorted_line_indices = np.lexsort((updated_lines[:, 1], updated_lines[:, 0]))
    lineset['lines'] = updated_lines[sorted_line_indices]

    edge_points[rows_changed_3] = edge_points[rows_changed_3, ::-1]
    edge_points = edge_points[sorted_line_indices]

    return lineset, edge_points


def remove_dup_v_l_ep(curveset, norm_edge_points=None):
    lineset = {'vertices': curveset['vertices'], 'lines': curveset['lines']}
    lineset, indices, indices_2, changed_rows = remove_duplicate_vertices_and_lines(
        lineset, return_indices=True, return_rows_changed=True,
    )

    edge_points = curveset['edge_points']
    edge_points[changed_rows] = edge_points[changed_rows, ::-1, :]

    indices = indices[indices_2]
    edge_points = edge_points[indices]

    if norm_edge_points is not None:
        norm_edge_points = norm_edge_points[indices]

    return {
        'vertices': lineset['vertices'],
        'lines': lineset['lines'],
        'edge_points': edge_points,
    }, norm_edge_points
