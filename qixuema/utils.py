"""
Adapted from https://github.com/thusiyuan/cooperative_scene_parsing/blob/master/utils/sunrgbd_utils.py
"""
import numpy as np
import open3d as o3d
from einops import rearrange
# from utils.helpful_fn import normalize, get_rotaion_matrix_3d, get_rotation_matrix_2d, is_close
from qixuema.helpers import normalize, get_rotaion_matrix_3d, get_rotation_matrix_2d, is_close
import networkx as nx
import itertools

from qixuema.geo_utils import (
    find_closed_loop, subdivide_longest, remove_duplicate_vertices_and_lines, derive_line_edges_from_lines_np, remove_unused_vertices
)
from qixuema.line_utils import sort_line_set_vertices_and_lines, has_unreferenced_vertices

def parse_camera_info(camera_info, height, width):
    """ extract intrinsic and extrinsic matrix
    """
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

def get_corners_of_bb3d(basis, coeffs, centroid):
    corners = np.zeros((8, 3))
    # order the basis
    index = np.argsort(np.abs(basis[:, 0]))[::-1]
    # the case that two same value appear the same time
    if index[2] != 2:
        index[1:] = index[1:][::-1]
    basis = basis[index, :]
    coeffs = coeffs[index]
    # Now, we know the basis vectors are orders X, Y, Z. Next, flip the basis vectors towards the viewer
    basis = flip_towards_viewer(basis, centroid)
    coeffs = np.abs(coeffs)
    corners[0, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[1, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[2, :] = basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[3, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]

    corners[4, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[5, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[6, :] = basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[7, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners = corners + np.tile(centroid, (8, 1))
    return corners

def get_corners_of_bb3d_no_index(basis, coeffs, centroid):
    corners = np.zeros((8, 3))
    coeffs = np.abs(coeffs)
    corners[0, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[1, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[2, :] = basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[3, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]

    corners[4, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[5, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[6, :] = basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[7, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]

    corners = corners + np.tile(centroid, (8, 1))
    return corners

def project_3d_points_to_2d(points3d, R_ex, K):
    """
    Project 3d points from camera-centered coordinate to 2D image plane
    Parameters
    ----------
    points3d: numpy array
        3d location of point
    R_ex: numpy array
        extrinsic camera parameter
    K: numpy array
        intrinsic camera parameter
    Returns
    -------
    points2d: numpy array
        2d location of the point
    """
    points3d = R_ex.dot(points3d.T).T
    x3 = points3d[:, 0]
    y3 = -points3d[:, 1]
    z3 = np.abs(points3d[:, 2])
    xx = x3 * K[0, 0] / z3 + K[0, 2]
    yy = y3 * K[1, 1] / z3 + K[1, 2]
    points2d = np.vstack((xx, yy))
    return points2d

def project_struct_bdb_to_2d(basis, coeffs, center, R_ex, K):
    """
    Project 3d bounding box to 2d bounding box
    Parameters
    ----------
    basis, coeffs, center, R_ex, K
        : K is the intrinsic camera parameter matrix
        : Rtilt is the extrinsic camera parameter matrix in right hand coordinates
    Returns
    -------
    bdb2d: dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    """
    corners3d = get_corners_of_bb3d(basis, coeffs, center)
    corners = project_3d_points_to_2d(corners3d, R_ex, K)
    bdb2d = dict()
    bdb2d['x1'] = int(max(np.min(corners[0, :]), 1))  # x1
    bdb2d['y1'] = int(max(np.min(corners[1, :]), 1))  # y1
    bdb2d['x2'] = int(min(np.max(corners[0, :]), 2*K[0, 2]))  # x2
    bdb2d['y2'] = int(min(np.max(corners[1, :]), 2*K[1, 2]))  # y2
    # if not check_bdb(bdb2d, 2*K[0, 2], 2*K[1, 2]):
    #     bdb2d = None
    return bdb2d

# def derive_line_edges_from_lines_np(
#     lines,
# ):
#     # 将“line segment”作为 graph 的 node，vertex 作为 graph 的 edge，得到用线表示的 graph 中的 edges。
#     max_num_lines = lines.shape[0]
#     line_edges_vertices_threshold = 1

#     all_edges = np.stack(np.meshgrid(
#         np.arange(max_num_lines),
#         np.arange(max_num_lines),
#         indexing='ij'), axis=-1)

#     shared_vertices = rearrange(lines, 'i c -> i 1 c 1') == rearrange(lines, 'j c -> 1 j 1 c')
#     num_shared_vertices = shared_vertices.any(axis = -1).sum(axis = -1)
#     is_neighbor_line = num_shared_vertices == line_edges_vertices_threshold

#     line_edge = all_edges[is_neighbor_line]
#     line_edge = np.sort(line_edge, axis=1)
    
#     # Use unique to find the unique rows in the sorted tensor
#     # Since the pairs are sorted, [1, 0] and [0, 1] will both appear as [0, 1] and be considered the same
#     line_edge, _ = np.unique(line_edge, return_inverse=True, axis=0)

#     # 最终返回 line_edges
#     return line_edge # [line_1_idx, line_2_idx]

def bfs_on_each_component(graph, return_list=False):
    # 这个函数用来遍历图中的每个连通分量

    # 获取图中所有连通分量的列表
    components = list(nx.connected_components(graph))
    
    components = [sorted(s) for s in components]
    
    components = sorted(components, key=lambda s: s[0])

    # 遍历每个连通分量
    node_seq = []
    if return_list:
        node_seq_list = []
    for component in components:
        # 为每个连通分量创建一个子图
        subgraph = graph.subgraph(component)

        # 选择每个连通分量的一个起始节点
        start_node = next(iter(component))

        # 对每个连通分量进行广度优先遍历
        bfs_tree = nx.bfs_tree(subgraph, start_node)

        # 打印广度优先遍历的结果
        # print(f"BFS of component starting at {start_node}: {list(bfs_tree.nodes())}")
        if return_list:
            node_seq_list.append(list(bfs_tree.nodes()))
        node_seq = node_seq + list(bfs_tree.nodes())
    
    if return_list:
        return node_seq_list
    
    return node_seq

def bfs_(edges, return_list=False):
    # max_vertex = np.amax(line_edges)
    
    G = nx.Graph()
    G.add_edges_from(edges)
    # G.add_nodes_from(np.arange(max_vertex + 1))    
    # 广度优先遍历
    
    node_seq = bfs_on_each_component(G, return_list=return_list)
    
    return node_seq

def remove_duplicate_vertices_and_lines_of_lineset(line_set, return_indices=False):
    points = np.asarray(line_set.points)
    lines = np.asarray(line_set.lines)

    # 移除重复的点并创建索引映射    
    unique_points, inverse_indices = np.unique(points, axis=0, return_inverse=True)
    line_set.points = o3d.utility.Vector3dVector(unique_points)

    updated_lines = inverse_indices[lines] # 更新线段索引，但是线段的顺序还是不变
    
    # 删除重复的线段，确保每条线段的小索引在前，大索引在后
    updated_lines = np.sort(updated_lines, axis=1)
    unique_lines, indices = np.unique(updated_lines, axis=0, return_index=True)
    line_set.lines = o3d.utility.Vector2iVector(unique_lines)
    
    if return_indices:
        return line_set, indices
    
    return line_set, None

# def remove_duplicate_vertices_and_lines(lineset:dict, return_indices=False):
#     points, lines = lineset['vertices'], lineset['lines']
    
#     # 移除重复的点并创建索引映射    
#     unique_points, inverse_indices = np.unique(points, axis=0, return_inverse=True)

#     updated_lines = inverse_indices[lines] # 更新线段索引，但是线段的顺序还是不变
    
#     # 删除重复的线段，确保每条线段的小索引在前，大索引在后
#     updated_lines = np.sort(updated_lines, axis=1)
#     unique_lines, indices = np.unique(updated_lines, axis=0, return_index=True) # 这里 unique 之后，lines 的顺序会被打乱
    
#     sorted_indices = np.argsort(indices)
#     unique_lines = unique_lines[sorted_indices] # 因此这里对 line 的顺序进行了重新排序，恢复原有的顺序，这是有必要的
    
#     lineset['vertices'] = unique_points
#     lineset['lines'] = unique_lines
    
#     if return_indices:
#         return lineset, indices, sorted_indices
#     else:
#         return lineset
    # return line_set, None, None

def remove_duplicate_vertices_and_lines_2d(line_set, return_indices=False):
    points = np.asarray(line_set.points)
    lines = np.asarray(line_set.lines)

    # 移除重复的点并创建索引映射    
    unique_points, inverse_indices = np.unique(points, axis=0, return_inverse=True)
    line_set.points = o3d.utility.Vector3dVector(unique_points)

    updated_lines = inverse_indices[lines] # 更新线段索引，但是线段的顺序还是不变
    
    # 删除重复的线段，确保每条线段的小索引在前，大索引在后
    updated_lines = np.sort(updated_lines, axis=1)
    unique_lines, indices = np.unique(updated_lines, axis=0, return_index=True)
    line_set.lines = o3d.utility.Vector2iVector(unique_lines)
    if line_set.colors:
        line_set.colors = o3d.utility.Vector3dVector(np.asarray(line_set.colors)[indices])
    
    if return_indices:
        return line_set, indices
    
    return line_set, None

def remove_duplicate_vertices(lineset:dict):
    keys = lineset.keys()
    
    key_1 = 'vertices'
    key_2 = 'lines' if 'lines' in keys else 'faces'
    
    vertices = lineset[key_1]
    lines = lineset[key_2]

    # Example tolerance value
    tolerance = 0.01  # Adjust this as per your requirement

    # Adjust points based on tolerance
    adjusted_points = np.round(vertices / tolerance) * tolerance
    # adjusted_points = points

    # 移除重复的点并创建索引映射    
    _, index, inverse_indices = np.unique(adjusted_points, axis=0, return_inverse=True, return_index=True)
    unique_points = vertices[index]  # 使用原始点集中的点

    updated_lines = inverse_indices[lines] # 更新线段索引，但是线段的顺序还是不变
    
    output = {key_1: unique_points, key_2: updated_lines}
    if 'colors' in keys:
        output['colors'] = np.asarray(lineset['colors'])[index]
    
    return output

def remove_unconnected_vertices_and_lines(line_set:dict):
    vertices = line_set['vertices']
    lines = line_set['lines']
    
    used_points, indices, inverse = np.unique(vertices, axis=0, return_index=True, return_inverse=True)

    # 更新线段索引
    updated_lines = inverse[lines] # 更新线段索引，但是线段的顺序还是不变

    updated_lines = np.sort(updated_lines, axis=1)
    unique_lines, indices = np.unique(updated_lines, axis=0, return_index=True) # 这里线段的顺序可能发生变化
    # print(unique_lines)
    # line_set.points = o3d.utility.Vector3dVector(used_points)
    # line_set.lines = o3d.utility.Vector2iVector(unique_lines)
    line_set['vertices'] = used_points
    line_set['lines'] = unique_lines
    
    return line_set

def merge_line_sets(line_set1, line_set2, line_set3, remove_duplicate_vertices_and_lines=False):
    # 合并顶点
    points1 = np.asarray(line_set1.points)
    points2 = np.asarray(line_set2.points)
    points3 = np.asarray(line_set3.points)
    all_points = np.vstack((points1, points2, points3))

    # 更新第二、三个 LineSet 的线段索引
    lines1 = np.asarray(line_set1.lines)
    lines2 = np.asarray(line_set2.lines) + len(points1)  # 调整索引
    lines3 = np.asarray(line_set3.lines) + len(points1) + len(points2)  # 调整索引
    all_lines = np.vstack((lines1, lines2, lines3))


    colors1 = np.asarray(line_set1.colors)
    colors2 = np.asarray(line_set2.colors)
    colors3 = np.asarray(line_set3.colors)
    all_colors = np.vstack((colors1, colors2, colors3))

    # 创建新的 LineSet
    new_line_set = o3d.geometry.LineSet()
    new_line_set.points = o3d.utility.Vector3dVector(all_points)
    new_line_set.lines = o3d.utility.Vector2iVector(all_lines)
    new_line_set.colors = o3d.utility.Vector3dVector(all_colors)

    if remove_duplicate_vertices_and_lines:
        new_line_set = remove_duplicate_vertices_and_lines(new_line_set)

    return new_line_set

def merge_vertices_and_lines(line_sets: list):
    vertices_dim = line_sets[0]['vertices'].shape[-1]
    
    all_points = np.array([]).reshape(0, vertices_dim)  # 假设点是3D的
    all_lines = np.array([], dtype=np.int32).reshape(0, 2)   # 假设线段由两点组成
    all_colors = np.array([]).reshape(0, 3)  # 假设颜色是RGB

    points_length = 0  # 用于累计点的数量

    for line_set in line_sets:
        points = np.asarray(line_set['vertices'])
        lines = np.asarray(line_set['lines'], dtype=np.int32) + points_length

        all_points = np.vstack((all_points, points))
        all_lines = np.vstack((all_lines, lines))
        
        if 'colors' in line_set:
            colors = np.asarray(line_set['colors'])
            all_colors = np.vstack((all_colors, colors))

        points_length += len(points)

    lineset = {}
    lineset['vertices'] = all_points
    lineset['lines'] = all_lines
    if 'colors' in line_sets[0]:
        lineset['colors'] = all_colors

    return lineset

def sort_line_set_vertices_and_lines_of_lineset(line_set, return_sorted_indices=False):
    # 将顶点转换为 NumPy 数组
    points = np.asarray(line_set.points)
    lines = np.asarray(line_set.lines)
    
    # 根据 Z-Y-X 规则对顶点排序
    sorted_indices = np.lexsort((points[:, 0], points[:, 1], points[:, 2]))
    sorted_points = points[sorted_indices]

    # 计算 inverse indices
    inverse_indices = np.argsort(sorted_indices)

    # 更新 LineSet 的顶点
    line_set.points = o3d.utility.Vector3dVector(sorted_points)

    updated_lines = inverse_indices[lines] # 更新线段索引，但是线段的顺序还是不变

    # outer sort
    sorted_indices = np.lexsort((updated_lines[:, 1], updated_lines[:, 0]))
    updated_lines = updated_lines[sorted_indices]

    # 更新 LineSet 的线段
    line_set.lines = o3d.utility.Vector2iVector(updated_lines)

    if return_sorted_indices:
        return line_set, sorted_indices
    
    return line_set, None

def bfs_line_set(line_set, is_door = False, return_line_seq=False):
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
    
    if return_line_seq:
        return line_set, line_seq
    
    return line_set, None

def bfs_lines(lines, return_bfs_list=False, return_line_seq=False):
    
    line_edges = derive_line_edges_from_lines_np(lines)    
    if return_bfs_list:    
        line_seq_list = bfs_(line_edges, return_list=True)
        line_seq = list(itertools.chain.from_iterable(line_seq_list))
    else:
        line_seq = bfs_(line_edges)

    lines = lines[line_seq]
    
    if return_bfs_list:
        line_seq_lens = [len(sublist) for sublist in line_seq_list]

        return lines, line_seq, line_seq_lens
    
    if return_line_seq:
        return lines, line_seq
    
    return lines, None


def bfs_vertices(lineset, return_bfs_list=False, return_vertice_seq=False):
    vertices = np.asarray(lineset['vertices'])
    lines = np.asarray(lineset['lines'])
    
    edges = lines

    if return_bfs_list:    
        vertices_seq_list = bfs_(edges, return_list=True)
        vertices_seq = list(itertools.chain.from_iterable(vertices_seq_list))
    else:
        vertices_seq = bfs_(edges)

    vertices = vertices[vertices_seq]
    
    if return_bfs_list:
        vertices_seq_lens = [len(sublist) for sublist in vertices_seq_list]

        return vertices, vertices_seq, vertices_seq_lens
    
    if return_vertice_seq:
        return vertices, vertices_seq
    
    return vertices, None


def expand_rectangle_np(vertices):
    # 提取 x 和 y 坐标
    x_coords = vertices[:, 0]
    y_coords = vertices[:, 1]

    x_coords_min = np.min(x_coords)
    x_coords_max = np.max(x_coords)
    y_coords_min = np.min(y_coords)
    y_coords_max = np.max(y_coords)

    # 计算 x 轴和 y 轴上的最大距离
    x_range = x_coords_max - x_coords_min
    y_range = y_coords_max - y_coords_min

    # 根据长边的方向扩展长方形
    if x_range > y_range:
        # 扩展 y 坐标
        vertices[is_close(y_coords, y_coords_min, 0.1), 1] -= 1
        vertices[is_close(y_coords, y_coords_max, 0.1), 1] += 1
        direction = "x 轴"
    elif y_range > x_range:
        # 扩展 x 坐标
        vertices[is_close(x_coords, x_coords_min, 0.1), 0] -= 1
        vertices[is_close(x_coords, x_coords_max, 0.1), 0] += 1
        direction = "y 轴"
    else:
        direction = "无需扩展，因为这是个正方形"

    return vertices, direction

import numpy as np


# # Example usage:
# # Create a simple point cloud
# point_cloud_example = np.array([
#     [1, 0, 0],  # Point at (1,0,0)
#     [0, 1, 0],  # Point at (0,1,0)
#     [-1, 0, 0], # Point at (-1,0,0)
#     [0, -1, 0]  # Point at (0,-1,0)
# ])

# # Rotate the example point cloud by 90 degrees around the Z axis
# rotated_point_cloud_example = rotate_point_cloud_z_axis(point_cloud_example, 90)

# rotated_point_cloud_example



def rotate_and_flip(points):
    if points.shape[-1] == 3:
        rotation_all = get_rotaion_matrix_3d()
    elif points.shape[-1] == 2:
        rotation_all = get_rotation_matrix_2d()
    else:
        raise NotImplementedError("Only 2D and 3D points are supported.")

    new_points = []

    for rotation in rotation_all:
        # 旋转
        rotated_vertices  = np.dot(points, rotation.T)
        
        new_points.append(rotated_vertices)
        
        # flip
        flipped_vertices = np.copy(rotated_vertices)  # 创建副本以避免就地修改
        flipped_vertices[..., 0] = -flipped_vertices[..., 0]
        
        new_points.append(flipped_vertices)

    return new_points # 1 -> 8


def rotate_and_flip_any_axis(points, axis=2):
    if points.shape[-1] == 3:
        rotation_all = get_rotaion_matrix_3d(axis)
    elif points.shape[-1] == 2:
        rotation_all = get_rotation_matrix_2d()
    else:
        raise NotImplementedError("Only 2D and 3D points are supported.")

    new_points = []
        

    flip_axis = (axis + 1) % 3

    for rotation in rotation_all:
        # 旋转
        rotated_vertices  = np.dot(points, rotation.T)
        
        new_points.append(rotated_vertices)
        
        # flip
        flipped_vertices = np.copy(rotated_vertices)  # 创建副本以避免就地修改
        flipped_vertices[..., flip_axis] = -flipped_vertices[..., flip_axis]
        
        new_points.append(flipped_vertices)

    return new_points # 1 -> 8


# 假设点云数据是一个 N x 3 的 NumPy 数组，表示 N 个点
def apply_transformation_to_point_cloud(point_cloud, transformation_matrix):
    """将变换矩阵应用到点云的所有点上"""
    # 将点云与变换矩阵相乘
    return np.dot(point_cloud, transformation_matrix.T)

def get_transformed_vertex_list(vertices, transformation_matrix):
    transformed_vertices_list = []
    for transformation in transformation_matrix:
        vertices_tmp = vertices.copy()
        transformed_vertices = apply_transformation_to_point_cloud(vertices_tmp, transformation)
        transformed_vertices_list.append(transformed_vertices)
    
    return transformed_vertices_list

def slicing_lines_by_color(colors):
    colors_value = (colors[:, 0] * 10).astype(np.int32)

    # 初始化变量
    current_value = colors_value[0]
    current_length = 1
    segments = []

    # 遍历数组
    for item in colors_value[1:]:  # 从第二个元素开始
        if item == current_value:
            # 如果元素与当前段的元素相同，增加长度
            current_length += 1
        else:
            # 如果元素不同，记录当前段的信息，并重置变量
            segments.append((current_value, current_length))
            current_value = item
            current_length = 1

    # 添加最后一段的信息
    segments.append((current_value, current_length))

    # print(segments)    
    
    return np.asarray(segments)

def find_closest_row(array, point):
    """
    在给定的二维数组中找到最接近指定点的行的索引。

    参数:
    array -- 二维数组
    point -- 一个二维点的坐标

    返回:
    最接近指定点的行的索引。
    """
    # 计算每一行与指定点之间的欧几里得距离
    distances = np.sqrt(np.sum((array - np.array(point))**2, axis=1))

    # 找到最小距离的索引
    min_distance_index = np.argmin(distances)

    return min_distance_index

def get_bottom_plane(room_lineset):
    vertices = room_lineset['points']
    min_value = np.min(vertices[:, 2])
    lines = room_lineset['lines']

    z_threshold = min_value + 0.1

    # 筛选出满足条件的边
    # selected_edges = []
    bottom_line_coords = []
    for line in lines:
        if vertices[line[0], 2] < z_threshold and vertices[line[1], 2] < z_threshold:
            bottom_line_coords.append(vertices[line[0]])
            bottom_line_coords.append(vertices[line[1]])

    bottom_lineset = {
        'points': np.array(bottom_line_coords), 
        'lines': [[i, i + 1] for i in range(0, len(bottom_line_coords), 2)]}
    
    if len(bottom_lineset['lines']) < 3:
        return {}
    
    bottom_lineset = remove_duplicate_vertices_and_lines(bottom_lineset)
    lines = bottom_lineset['lines']
    vertices = bottom_lineset['points']
    
    loop = find_closed_loop(lines)
    if len(loop) < 3:
        return {}
    
    bottom_plane = {}
    bottom_plane['points'] = bottom_lineset['points']
    bottom_plane['faces'] = [loop]
    
    return bottom_plane

def get_top_plane(room_lineset):
    vertices = room_lineset['points']
    max_value = np.max(vertices[:, 2])
    lines = room_lineset['lines']

    z_threshold = max_value - 0.1

    # 筛选出满足条件的边
    bottom_line_coords = []
    for line in lines:
        if vertices[line[0], 2] > z_threshold and vertices[line[1], 2] > z_threshold:
            bottom_line_coords.append(vertices[line[0]])
            bottom_line_coords.append(vertices[line[1]])

    top_lineset = {
        'points': np.array(bottom_line_coords), 
        'lines': [[i, i + 1] for i in range(0, len(bottom_line_coords), 2)]}
    
    if len(top_lineset['lines']) < 3:
        return {}
    top_lineset = remove_duplicate_vertices_and_lines(top_lineset)
    lines = top_lineset['lines']
    vertices = top_lineset['points']
    
    
    loop = find_closed_loop(lines)
    if len(loop) < 3:
        return {}
    
    top_plane = {}
    top_plane['points'] = top_lineset['points']
    top_plane['faces'] = [loop]
    
    return top_plane


def shadow_follower_sort(A:list[int], B:list):
    """
    对列表A进行排序,并使列表B中的元素跟随A中相应元素的排序顺序进行调整。

    参数:
    A - 需要排序的列表。
    B - 需要跟随A排序的列表。

    返回值:
    sorted_A - 排序后的A列表。
    sorted_B - 调整顺序后的B列表,以匹配A的排序。
    """

    # 确保A和B长度相同
    if len(A) != len(B):
        raise ValueError("列表A和B的长度必须相同!")

    # 使用np.argsort获取排序后的索引
    sorted_indices = np.argsort(A)

    # 使用这些索引来排序A和B
    sorted_A = np.array(A)[sorted_indices]
    sorted_B = np.array(B)[sorted_indices]

    return sorted_A.tolist(), sorted_B.tolist()

def count_vertex_connections(vertices, edges):
    # 创建一个长度为顶点数的数组，初始计数都是0
    connection_counts = np.zeros(vertices.shape[0], dtype=int)

    # 遍历所有线段
    for edge in edges:
        # 对于线段的每个顶点，计数加1
        connection_counts[edge[0]] += 1
        connection_counts[edge[1]] += 1

    return connection_counts

def line_10f_to_line_segments(sample):
    """
    将样本转换为线段的起点和终点坐标。

    参数:
    sample (np.ndarray): 输入数组，形状为 ['bs', 'num_lines', 10]

    返回:
    np.ndarray: 输出数组，形状为 ['num_lines', 6]，包含每条线段的起点和终点坐标
    """
    # 取出起点坐标 p0
    p0 = sample[..., :3].copy()
    # 取出 delta 坐标
    delta = sample[..., 3:6]
    # 计算中点 mid_p
    mid_p = p0 + delta / 2
    # 找出最大值的索引 idx
    idx = np.argmax(sample[..., 6:], axis=-1)
    print('idx:', idx)
    # 初始化 p1 为与 p0 相同的形状
    p1 = np.zeros_like(p0)
    
    # 计算 mask，找到 idx 为 0 的位置
    mask = idx == 0
    # 对于 mask 为 True 的位置，计算 p1
    p1[mask] = p0[mask] + delta[mask]
    
    # 计算非零 mask
    non_zero_mask = ~mask
    
    # 对于非零 mask 位置，计算 non_zero_delta 和 non_zero_mid_p
    non_zero_delta = delta[non_zero_mask]
    non_zero_idx = idx[non_zero_mask] - 1
    non_zero_delta[np.arange(len(non_zero_delta)), non_zero_idx] *= -1
    
    non_zero_mid_p = mid_p[non_zero_mask]
    p0[non_zero_mask] = non_zero_mid_p - non_zero_delta / 2
    p1[non_zero_mask] = non_zero_mid_p + non_zero_delta / 2
    
    # 将 p0 和 p1 拼接起来
    line_segments = np.concatenate([p0, p1], axis=-1)
    
    return line_segments  # ['bs', 'nl', 6]

def line_coords_to_line_10f(
    line_coords, # TensorType['batch', 'nl', 2, 3, int]
):    
    bs, num_lines, _, _ = line_coords.shape

    # 计算BB的起始坐标
    min_p = np.min(line_coords, axis=-2)  # 形状: (batch, num_lines, 3)
    
    # 计算线段的 BB 的 width, height and depth, 分别对应 x,y,z 方向
    gt_line = line_coords[..., 1, :] - line_coords[..., 0, :]
    delta = np.abs(gt_line)
    
    widths_heights_depths = delta  # 形状 (batch, num_lines, 3)

    line_all = np.repeat(delta[..., np.newaxis, :], repeats=4, axis=-2)

    # 创建一个对角矩阵来反转特定位置的符号
    mask = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=bool)

    mask = np.tile(mask, (bs, num_lines, 1, 1))
    line_all[mask] *= -1
    
    gt_line = np.repeat(gt_line[..., np.newaxis, :], repeats=4, axis=-2)

    # 计算每个向量的模
    lines1_norm = np.linalg.norm(line_all, axis=-1)
    lines2_norm = np.linalg.norm(gt_line, axis=-1)

    # 计算点积
    dot_product = np.sum(line_all * gt_line, axis=-1)

    # 计算余弦值
    cosine_values = dot_product / (lines1_norm * lines2_norm + 1e-8)
    cosine_values = np.abs(cosine_values)
    
    res = np.concatenate([min_p, widths_heights_depths, cosine_values], axis=-1)

    return res

def sample_point_from_wf(wireframe):
    sample_distance=0.003
    
    # 确保wireframe包含所需的键
    if 'vertices' not in wireframe or 'lines' not in wireframe:
        raise ValueError("wireframe must contain 'vertices' and 'lines' keys")
    
    vertices = wireframe['vertices']
    lines = wireframe['lines']
    # print(len(lines))
    # sampled_points = []

    # 预估最大可能的采样点数量并预分配空间
    # max_samples = sum([max(int(np.linalg.norm(vertices[line[1]] - vertices[line[0]]) / sample_distance), 1) for line in lines])
    
    # 初始化max_samples变量来累加每条线段的样本数
    max_samples = 0

    # 遍历每条线段的索引对
    for line in lines:
        # 计算线段两个顶点的坐标差
        vertex_difference = vertices[line[1]] - vertices[line[0]]

        # 计算两个顶点之间的欧氏距离
        distance = np.linalg.norm(vertex_difference)

        if np.isinf(distance) or distance == 0:
            continue

        # 根据采样距离计算样本数量，并确保至少为1
        num_samples = int(distance / sample_distance)
        num_samples = max(num_samples, 1)

        # 将当前线段的样本数累加到总样本数中
        max_samples += num_samples
    
    sampled_points = np.empty((max_samples, vertices.shape[1]))

    count = 0
    
    for line in lines:
        start, end = vertices[line[0]], vertices[line[1]]
        # Calculate the distance between the start and end points
        line_length = np.linalg.norm(end - start)
        if line_length <= sample_distance + 0.001 or (not np.isfinite(line_length)):
            continue
        
        num_samples = max(int(line_length / sample_distance), 1)
        fractions = np.linspace(0, 1, num_samples, endpoint=False)[1:]
        line_samples = start + np.outer(fractions, end - start)

        sampled_points[count:count + len(line_samples)] = line_samples
        count += len(line_samples)
        
    
    sampled_points = sampled_points[:count]

    # Combine the original vertices with the new sampled points
    all_points = np.vstack((vertices, sampled_points))
    
    return all_points

def sort_vertices_and_lines(lineset:dict, return_indices=False, multi_graph=False, tolerance=0.0001):

    lineset, indices, indices_2 = remove_duplicate_vertices_and_lines(
        lineset, 
        return_indices=True,
    )

    lineset, sorted_indices = sort_line_set_vertices_and_lines(
        lineset, 
        return_sorted_indices=multi_graph, 
        tolerance=tolerance, 
    )
    
    # vis_lineset(lineset)
    
    if multi_graph:
        lineset['vertices'], vertice_seq, vertices_seq_lens = bfs_vertices(lineset, return_bfs_list=multi_graph, return_vertice_seq=multi_graph)
    else:
        lineset['vertices'], vertice_seq = bfs_vertices(lineset, return_vertice_seq=multi_graph)

    lines = lineset['lines']

    # 计算 inverse indices
    inverse_indices = np.argsort(vertice_seq)

    updated_lines = inverse_indices[lines] # 更新线段索引，但是线段的顺序还是不变

    # inner sort
    updated_lines = np.sort(updated_lines, axis=1)

    # outer sort
    sorted_line_indices = np.lexsort((updated_lines[:, 1], updated_lines[:, 0])) # 线段的顺序在这里发生了变化
    updated_lines = updated_lines[sorted_line_indices]        
    lineset['lines'] = updated_lines

    if multi_graph:
        indices = indices[indices_2]
        indices = indices[sorted_indices]
        indices = indices[sorted_line_indices]
    
        return lineset, indices, vertices_seq_lens
    
    # vis_lineset(line_set)

    if return_indices:
        indices = indices[indices_2]
        indices = indices[sorted_indices]
        indices = indices[vertice_seq]
        
        return lineset, indices

    return lineset

def sort_lineset(lineset:dict, return_indices=False, multi_graph=False):

    lineset, indices, indices_2 = remove_duplicate_vertices_and_lines(lineset, return_indices=True)
    lineset, sorted_indices = sort_line_set_vertices_and_lines(lineset, return_sorted_indices=multi_graph)
    
    # vis_lineset(lineset)
    
    if multi_graph:
        lineset['lines'], line_seq, line_seq_lens = bfs_lines(lines=lineset['lines'], return_bfs_list=multi_graph, return_line_seq=multi_graph)
    else:
        lineset['lines'], line_seq = bfs_lines(lines=lineset['lines'], return_line_seq=multi_graph)

    if multi_graph:
        indices = indices[indices_2]
        indices = indices[sorted_indices]
        indices = indices[line_seq]
        
        return lineset, indices, line_seq_lens
    
    # vis_lineset(line_set)

    if return_indices:
        indices = indices[indices_2]
        indices = indices[sorted_indices]
        indices = indices[line_seq]
        
        return lineset, indices

    return lineset

def subdivide_longest_high(lineset, max_length=256):
    segments = lineset['vertices'][lineset['lines']]
    
    segments = subdivide_longest(segments, max_length=max_length)
    
    # 从 segment 转换为 lineset，方便后续 remove_duplicate_vertices_and_lines 操作和排序操作
    vertices = rearrange(segments, 'nl nv nc -> (nl nv) nc')
    lines = np.arange(len(vertices)).reshape(-1, 2)

    lineset = {'vertices': vertices, 'lines': lines}    
    
    lineset = remove_duplicate_vertices_and_lines(lineset)
    
    # 对“线段”进行排序
    lineset, sorted_indices = sort_line_set_vertices_and_lines(lineset)
    
    lines, line_seq = bfs_lines(lines=lineset['lines'])
    lineset['lines'] = lines
    
    return lineset


def sort_vertices_and_curves(
    lineset:dict, 
    edge_points:np.ndarray,
    multi_graph=False, 
    tolerance=0.0001, 
    return_rows_changed=False
):
    lineset, indices_1, indices_2, rows_changed_1 = remove_duplicate_vertices_and_lines(
        lineset, 
        return_indices=True,
        return_rows_changed=return_rows_changed
    )
    
    lineset = remove_unused_vertices(lineset)

    if has_unreferenced_vertices(lineset):
        return

    edge_points[rows_changed_1] = edge_points[rows_changed_1, ::-1]
    indices = indices_1[indices_2]
    edge_points = edge_points[indices]


    lineset, sorted_indices, rows_changed_2 = sort_line_set_vertices_and_lines(
        lineset, 
        return_sorted_indices=multi_graph, 
        tolerance=tolerance, 
        return_rows_changed=return_rows_changed)
    
    edge_points[rows_changed_2] = edge_points[rows_changed_2, ::-1]
    edge_points = edge_points[sorted_indices]


    # vis_lineset(lineset)
    
    if multi_graph:
        lineset['vertices'], vertice_seq, vertices_seq_lens = bfs_vertices(lineset, return_bfs_list=multi_graph, return_vertice_seq=multi_graph)
    else:
        lineset['vertices'], vertice_seq = bfs_vertices(lineset, return_vertice_seq=multi_graph)

    lines = lineset['lines']

    # 计算 inverse indices
    inverse_indices = np.argsort(vertice_seq)

    updated_lines = inverse_indices[lines] # 更新线段索引，但是线段的顺序还是不变

    # 保存排序前的数组以便后续比较
    updated_lines_before = np.copy(updated_lines)

    # inner sort
    updated_lines = np.sort(updated_lines, axis=1)
    
    # 比较排序前后的行是否发生了变化，生成一个布尔数组，表示每一行是否改变
    rows_changed_3 = np.any(updated_lines_before != updated_lines, axis=1)  

    # outer sort
    sorted_line_indices = np.lexsort((updated_lines[:, 1], updated_lines[:, 0])) # 线段的顺序在这里发生了变化
    updated_lines = updated_lines[sorted_line_indices]        
    lineset['lines'] = updated_lines

    edge_points[rows_changed_3] = edge_points[rows_changed_3, ::-1]
    edge_points = edge_points[sorted_line_indices]


    return lineset, edge_points



def remove_dup_v_l_ep(curveset, norm_edge_points=None):
    lineset = {
        'vertices': curveset['vertices'],
        'lines': curveset['lines']
    }

    lineset, indices, indices_2, changed_rows = remove_duplicate_vertices_and_lines(lineset, return_indices=True, return_rows_changed=True)
    # vis_lineset(lineset)

    # 要注意的是，我们要同步更新 edge_points 和 norm_edge_points

    edge_points = curveset['edge_points']
    
    edge_points[changed_rows] = edge_points[changed_rows, ::-1, :]

    indices = indices[indices_2]    

    edge_points = edge_points[indices]
    
    if norm_edge_points is not None:
        norm_edge_points = norm_edge_points[indices]

    return {
        'vertices': lineset['vertices'],
        'lines': lineset['lines'],
        'edge_points': edge_points }, norm_edge_points