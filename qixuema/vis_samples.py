import argparse
import logging
import os
import pickle
import random
import sys

import numpy as np
import open3d as o3d
from einops import rearrange, repeat

from qixuema.geo_utils import (
    check_order_single_point,
    edge_points_to_lineset,
    inverse_transform_polyline,
)
from qixuema.helpers import get_file_list, get_filename_wo_ext
from qixuema.io_utils import read_obj_file
from qixuema.np_utils import rotation_matrix_x, rotation_matrix_y, rotation_matrix_z

logger = logging.getLogger(__name__)

# Canonical start/end pair used when unnormalizing curves.
_NORM_START_END = np.array([
    [0.0, 0.0, 0.0],
    [0.54020254, -0.77711392, 0.32291667],
])

MIN_LINES_FILTER = 48
MAX_LINES_FILTER = 128
SAMPLE_POINTS_PER_EDGE = 64
POINTS_PER_EDGE = 256


def rotate_points(points, theta, axis=2):
    if axis == 0:
        R = rotation_matrix_x(theta)
    elif axis == 1:
        R = rotation_matrix_y(theta)
    elif axis == 2:
        R = rotation_matrix_z(theta)
    else:
        return points
    return points @ R.T


def _build_lineset_and_pointcloud(vertices, lines):
    """Convert numpy vertices/lines into an o3d LineSet plus a deduplicated PointCloud."""
    vertices = vertices.astype(np.float64) if vertices.dtype != np.float64 else vertices
    lines = lines.astype(np.int32) if lines.dtype != np.int32 else lines

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.unique(vertices, axis=0))

    return line_set, pcd


def vis_curveset(
    lineset,
    norm_edge_points=None,
    edge_points=None,
    use_edge_points=False,
    is_check_order=False,
    vis=False,
):
    segments = lineset['vertices'][lineset['lines']]

    original_edge_points = []
    for i, start_and_end in enumerate(segments):
        if use_edge_points and edge_points is not None:
            start_and_end = np.stack((edge_points[i][0], edge_points[i][-1]), axis=0)

        if is_check_order and not check_order_single_point(start_and_end[0], start_and_end[1]):
            start_and_end = start_and_end[::-1]

        if norm_edge_points is not None:
            out = inverse_transform_polyline(norm_edge_points[i], start_and_end=_NORM_START_END)
            out = inverse_transform_polyline(out, start_and_end=start_and_end)
        else:
            out = edge_points[i]

        original_edge_points.append(out)

    edge_points = np.array(original_edge_points)
    new_vertices, new_lines = edge_points_to_lineset(edge_points)

    if vis:
        line_set, pcd = _build_lineset_and_pointcloud(new_vertices, new_lines)
        o3d.visualization.draw_geometries(
            [line_set, pcd], window_name='lineset', width=1600, height=1200,
        )
        return None

    return new_vertices, new_lines, edge_points


def vis_data(sample_iterator):

    def update_mesh(vis):
        logger.info("Updating visualization with next sample.")
        vis.clear_geometries()

        sample = next(sample_iterator)

        if isinstance(sample, dict):
            segments = sample['segments']
            while len(segments) < MAX_LINES_FILTER:
                sample = next(sample_iterator)
                segments = sample['segments']

            vertices = segments.reshape(-1, 3)
            lines = np.arange(0, len(vertices)).reshape(-1, 2)

            logger.info("Loading %s", sample.get('uid', '<unknown>'))
        elif isinstance(sample, np.ndarray):
            assert sample.ndim == 2
            vertices = sample
            lines = np.array([[i, i + 1] for i in range(len(vertices) - 1)])

        if isinstance(sample, dict) and 'norm_edge_points' in sample:
            vertices, lines = vis_curveset(
                sample,
                norm_edge_points=sample['norm_edge_points'],
                use_edge_points=False,
            )

        line_set, pcd = _build_lineset_and_pointcloud(vertices, lines)
        vis.add_geometry(line_set)
        vis.add_geometry(pcd)

        vis.poll_events()
        vis.update_renderer()
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=800, height=600)
    vis.register_key_callback(ord(' '), update_mesh)

    print("Press Space to generate and display a new mesh.")
    print("Press Q to quit.")

    vis.run()
    vis.destroy_window()


def _lines_for_batched_points(sampled_points: np.ndarray):
    """Turn (B, N, 3) points into flat (vertices, lines) with batch-offset line indices."""
    bs, num_points, _ = sampled_points.shape
    points_index = np.arange(num_points)
    lines = np.column_stack((points_index, np.roll(points_index, -1)))[:-1]
    lines = repeat(lines, 'nl c -> b nl c', b=bs)

    offsets = rearrange(np.arange(bs) * num_points, 'b -> b 1 1')
    lines = (lines + offsets).reshape(-1, 2)
    vertices = sampled_points.reshape(-1, 3)
    return vertices, lines


def _load_npz_sample(file_path, iterator):
    sample = np.load(file_path, allow_pickle=True)
    while len(sample['lines']) < MIN_LINES_FILTER or len(sample['lines']) > MAX_LINES_FILTER:
        file_path = next(iterator)
        sample = np.load(file_path, allow_pickle=True)

    vertices, lines, _ = vis_curveset(
        sample,
        norm_edge_points=sample['norm_edge_points'],
        use_edge_points=True,
    )
    return vertices, lines, file_path


def _load_npy_sample(file_path):
    sample = np.load(file_path)
    indices = np.linspace(0, POINTS_PER_EDGE - 1, SAMPLE_POINTS_PER_EDGE, dtype=int)
    return _lines_for_batched_points(sample[:, indices, :])


def _load_obj_sample(file_path):
    lineset = read_obj_file(file_path)
    return lineset['vertices'], lineset['lines']


def _load_pkl_sample(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    edge_pnts = data['edge_pnts']

    points_index = np.arange(POINTS_PER_EDGE)
    lines = np.column_stack((points_index, np.roll(points_index, -1)))
    lines = np.repeat(lines, len(edge_pnts), axis=0)

    offsets = rearrange(np.arange(len(edge_pnts)) * POINTS_PER_EDGE, 'b -> b 1 1')
    lines += offsets

    return edge_pnts.reshape(-1, 3), lines


def main(dir_path):
    file_path_list = get_file_list(dir_path)

    random.seed(4231)
    random.shuffle(file_path_list)

    file_path_list_iterator = iter(file_path_list)
    current = {'file_path': None}

    def update_mesh(vis):
        logger.info("Updating visualization with next sample.")
        vis.clear_geometries()

        file_path = next(file_path_list_iterator)

        if file_path.endswith('npz'):
            vertices, lines, file_path = _load_npz_sample(file_path, file_path_list_iterator)
        elif file_path.endswith('npy'):
            vertices, lines = _load_npy_sample(file_path)
        elif file_path.endswith('obj'):
            vertices, lines = _load_obj_sample(file_path)
        elif file_path.endswith('pkl'):
            vertices, lines = _load_pkl_sample(file_path)
        else:
            logger.warning("Unsupported file type: %s", file_path)
            return False

        current['file_path'] = file_path
        logger.info("Loaded %s", file_path)

        line_set, pcd = _build_lineset_and_pointcloud(vertices, lines)
        vis.add_geometry(line_set)
        vis.add_geometry(pcd)

        vis.poll_events()
        vis.update_renderer()
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=1200, height=900)
    vis.register_key_callback(ord(' '), update_mesh)

    print("Press Space to generate and display a new mesh.")
    print("Press Q to quit.")

    vis.run()
    vis.destroy_window()


def vis_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    random.shuffle(data)
    vis_data(iter(data))


def vis_npy(npy_path):
    sample = np.load(npy_path)
    vis_data(iter(sample))


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize curve/line set samples for debugging.")
    parser.add_argument("--path", required=True, type=str,
                        help="Directory containing samples to visualize.", metavar="DIR")
    parser.add_argument("--path_2", required=False, type=str,
                        help="Optional second directory.", metavar="DIR")
    parser.add_argument("--num_processes", required=False, type=int,
                        help="Number of processes.", metavar="INT", default=2)

    args, unknown = parser.parse_known_args()

    if unknown:
        print(f"Ignoring unknown arguments: {unknown}")

    return args


if __name__ == '__main__':
    args = parse_args()
    main(dir_path=args.path)
