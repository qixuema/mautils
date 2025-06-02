import numpy as np
from einops import rearrange, repeat
from typing import Optional
import heapq
from deprecated import deprecated
from qixuema.helpers import check_nan_inf
from qixuema.o3d_utils import get_vertices_obb
import open3d as o3d
import logging
from scipy.interpolate import splprep, splev, interp1d
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)

START_END = np.array(
    [[0.0, 0.0, 0.0], 
    [0.54020254, -0.77711392, 0.32291667]]
)

START_END_R = np.array([
    [ 0.54020282, -0.77711348,  0.32291649],
    [ 0.77711348,  0.60790503,  0.1629285 ],
    [-0.32291649,  0.1629285 ,  0.93229779]
])


def safe_norm(v: np.ndarray, eps: float) -> np.ndarray:
    """
    Compute the L2 norm of `v` along the last axis and clamp it to be at least `eps`.
    """
    return np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)

def discretize(t, *, continuous_range=(-1,1), num_discrete=128):
    lo, hi = continuous_range
    assert hi > lo, "Upper bound must be greater than lower bound."

    t = (t - lo) / (hi - lo)
    t *= num_discrete
    t -= 0.5

    # Convert to discrete values, enforce data type, and clamp the values
    t = np.round(t).astype(int)
    t = np.clip(t, 0, num_discrete - 1)
    
    return t

def undiscretize(t, continuous_range=(-1,1), num_discrete=128):
    lo, hi = continuous_range
    assert hi > lo

    t = t.astype(np.float32)  # 确保t为浮点数类型

    t += 0.5
    t /= num_discrete
    return t * (hi - lo) + lo

def calculate_cosine_with_z_axis(vertex1, vertex2, vertex3):
    """
    计算由三个顶点定义的平面的法线向量与Z轴的重合度（即夹角的余弦值）。

    参数:
    vertex1, vertex2, vertex3 -- 分别代表三维空间中的三个顶点，每个顶点是一个形如 (x, y, z) 的元组或列表。

    返回值:
    cos_angle -- 法线向量与Z轴夹角的余弦值。
    """
    # 将顶点转换为NumPy数组
    A = np.array(vertex1)
    B = np.array(vertex2)
    C = np.array(vertex3)

    # 计算向量 AB 和 AC
    AB = B - A
    AC = C - A

    # 计算法线向量（AB 和 AC 的叉积）
    normal = np.cross(AB, AC)

    # 归一化法线向量
    normal_normalized = normal / (np.linalg.norm(normal) + 1e-5)

    # 定义Z轴向量
    Z = np.array([0, 0, 1])

    # 计算与Z轴的点乘
    cos_angle = np.dot(normal_normalized, Z)

    return cos_angle

def is_counter_clockwise(points):
    # 计算所有相邻顶点对的叉积
    dx = np.diff(points[:, 0])  # x坐标的差
    dy = np.diff(points[:, 1])  # y坐标的差
    sum = np.sum(dx[:-1] * dy[1:] - dx[1:] * dy[:-1])

    # 判断顶点顺序
    return sum > 0

def normalize_point_cloud(points, scale=2.0):
    # cube

    # Step 1: Find the min and max
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)

    # Step 2: Calculate the range and mid points
    ranges = max_vals - min_vals
    range_max = np.max(ranges)
    mid_points = (max_vals + min_vals) / 2

    # Step 3: Normalize each dimension to [-1, 1]
    normalized_points = (points - mid_points) * scale / range_max

    return normalized_points


def rotate_point_cloud_z_axis(point_cloud, angle_degrees):
    """
    Rotates a point cloud around the Z axis by a given angle.

    Parameters:
    - point_cloud: A numpy array of shape (N, 3) where N is the number of points.
    - angle_degrees: The rotation angle in degrees.

    Returns:
    - A new numpy array with the rotated point cloud.
    """
    
    # Convert the angle from degrees to radians
    angle_radians = np.deg2rad(angle_degrees)
    
    # Create the rotation matrix for the Z axis
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians), 0],
        [np.sin(angle_radians),  np.cos(angle_radians), 0],
        [0,                     0,                      1]
    ])
    
    # Apply the rotation to each point
    rotated_point_cloud = np.dot(point_cloud, rotation_matrix)
    
    return rotated_point_cloud


def remove_self_loops(edges):
    # 将数组转换为NumPy数组（如果它还不是）
    edges = np.array(edges)
    # 找出所有不是自环的边（即两个顶点不相同的边）
    valid_edges = edges[edges[:, 0] != edges[:, 1]]
    return valid_edges

def find_closed_loop(edges):
    edges = remove_self_loops(edges)
    # 检查每个顶点的出现次数，以确认循环的有效性
    if not all(np.sum(edges == vertex) == 2 for vertex in np.unique(edges)):
        return []

    # 选择一个起点
    start = edges[0, 0]
    loop = [start]
    current_vertex = edges[0, 1]

    # 构建循环
    while True:
        loop.append(current_vertex)
        
        # 如果回到起点，循环完成
        if current_vertex == start and len(loop) > 2:
            break

        # 寻找下一步
        for edge in edges:
            if current_vertex in edge:
                next_vertex = edge[0] if edge[1] == current_vertex else edge[1]
                if next_vertex != loop[-2]:  # 避免回到上一顶点
                    current_vertex = next_vertex
                    break

    return loop[:-1]


def move_vertices_to_origin(vertices):
    """
    移动所有顶点，使得它们的几何中心移动到原点位置。

    参数:
    vertices (np.ndarray): 顶点坐标数组，形状为(n, 3)，其中n是顶点数。

    返回:
    np.ndarray: 调整后的顶点坐标数组。
    """
    if not isinstance(vertices, np.ndarray) or vertices.shape[1] != 3:
        raise ValueError("顶点数组必须是形状为(n, 3)的NumPy数组")

    # 计算所有顶点的几何中心
    # center = np.mean(vertices, axis=0)
    # 每个维度的最小值
    min_values = np.min(vertices, axis=0)
    # 每个维度的最大值
    max_values = np.max(vertices, axis=0)
    center = (min_values + max_values) / 2

    # 移动顶点，使得中心位于原点
    adjusted_vertices = vertices - center

    return adjusted_vertices


def move_vertices_to_coord(vertices, target_coord=0.01):
    """
    将顶点的最低坐标移动到z=target_z平面上。

    参数:
    vertices (np.ndarray): 顶点坐标数组，形状为(n, 3)，其中n是顶点数。
    target_z (float): 目标z坐标。

    返回:
    np.ndarray: 调整后的顶点坐标数组。
    """
    if not isinstance(vertices, np.ndarray) or vertices.shape[1] != 3:
        raise ValueError("顶点数组必须是形状为(n, 3)的NumPy数组")

    # 找到最低的z坐标
    move_axis = 2
    # min_coord = np.min(vertices[:, move_axis])
    min_coord = np.min(vertices[:, move_axis])
    

    # 计算需要上移的距离
    delta_coord = target_coord - min_coord

    # 更新所有顶点的z坐标
    vertices[:, move_axis] += delta_coord

    return vertices


def scale_vertices_in_sphere(vertices, target_radius=1.0, center=None, return_scale_factor_and_center=False):
    # 不改变原始数据的中心点位置，只是缩放到球体内

    # 计算几何中心
    if center == None:
        # center = np.mean(vertices, axis=0)
        
        # 每个维度的最小值
        min_values = np.min(vertices, axis=0)
        # 每个维度的最大值
        max_values = np.max(vertices, axis=0)
        
        center = (min_values + max_values) / 2  

    # 计算每个顶点到中心的距离，并找到最大距离
    distances = np.linalg.norm(vertices - center, axis=1)
    max_distance = np.max(distances)

    # 计算缩放因子
    scale_factor = target_radius / max_distance

    # 应用缩放
    scaled_vertices = center + (vertices - center) * scale_factor

    if return_scale_factor_and_center:
        return scaled_vertices, (scale_factor, center)

    return scaled_vertices

def normalize_vertices_in_sphere(vertices, target_radius=1.0, center=None):

    # 计算几何中心
    if center == None:
        # center = np.mean(vertices, axis=0)
        
        # 每个维度的最小值
        min_values = np.min(vertices, axis=0)
        # 每个维度的最大值
        max_values = np.max(vertices, axis=0)
        
        center = (min_values + max_values) / 2  

    # 计算每个顶点到中心的距离，并找到最大距离
    distances = np.linalg.norm(vertices - center, axis=1)
    max_distance = np.max(distances)

    # 计算缩放因子
    scale_factor = target_radius / max_distance

    # 应用缩放
    scaled_vertices = (vertices - center) * scale_factor

    return scaled_vertices

def transform_wf_to_ground(vertices, center=None):
    vertices = move_vertices_to_origin(vertices)
    
    vertices = move_vertices_to_coord(vertices, 0.01)

    vertices = scale_vertices_in_sphere(vertices, 1.0, center)
    
    return vertices

def fit_vertices_to_unit_sphere(vertices):
    """
    Translates a group of vertices so their center moves to the origin and scales them to fit inside a unit sphere.

    :param vertices: np.array, a 2D array of vertices where each row represents a vertex.
    :return: np.array, the transformed vertices.
    """
    # Step 1: Calculate the centroid
    centroid = np.mean(vertices, axis=0)

    # Step 2: Translate vertices so centroid is at the origin
    translated_vertices = vertices - centroid

    # Step 3: Scale vertices to fit inside a unit sphere
    max_distance = np.max(np.linalg.norm(translated_vertices, axis=1))
    scaled_vertices = translated_vertices / max_distance

    return scaled_vertices

def translate_to_origin(points):
    """
    将点集的 bounding box 的中心移动到原点。
    
    参数:
    points (np.ndarray): 一个形状为 (n, 3) 的 NumPy 数组，表示 n 个三维点。
    
    返回:
    np.ndarray: 平移后的点集，形状为 (n, 3)。
    """
    # 计算 bounding box 的最小值和最大值
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)

    # 计算 bounding box 的中心点
    center = (min_vals + max_vals) / 2

    # 平移所有点
    translated_points = points - center

    return translated_points


def rotate_around_z(vertices, theta):
    """
    将一组顶点绕 Z 轴旋转 theta 度。

    参数:
    - vertices: 一个形状为 (N, 3) 的 numpy 数组，每行表示一个顶点的 (x, y, z) 坐标。
    - theta: 旋转角度（以弧度为单位）。

    返回:
    - rotated_vertices: 一个形状为 (N, 3) 的 numpy 数组，表示旋转后的顶点坐标。
    """
    # 构建绕 Z 轴旋转的旋转矩阵
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    
    # 对所有顶点应用旋转矩阵
    rotated_vertices = vertices @ rotation_matrix.T
    
    return rotated_vertices


def judge_flattened_objects(points, aspect_ratio_threshold=5.0, return_indices=False):

    # 计算每个维度的最小值和最大值
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)

    # 计算边界框的尺寸
    dimensions = max_vals - min_vals

    # 找到最长边和最短边
    longest_edge = np.max(dimensions)
    shortest_edge = np.min(dimensions)

    longest_edge_axis = np.argmax(dimensions)
    shortest_edge_axis = np.argmin(dimensions)

    # 计算最大和最小维度的比例
    aspect_ratio = longest_edge / shortest_edge

    if return_indices:

        return aspect_ratio, longest_edge_axis, shortest_edge_axis

    # 根据宽高比来决定是否保留此物体
    if aspect_ratio < aspect_ratio_threshold:
        return True    
    else:
        return False
    
# 2024-09-10
def remove_duplicate_vertices(lineset:dict, return_indices=False, tolerance=0.0001):
    vertices, lines = lineset['vertices'], lineset['lines']

    # Example tolerance value
    # Adjust this as per your requirement

    # Adjust points based on tolerance
    adjusted_points = np.round(vertices / tolerance) * tolerance
    
    # 移除重复的点并创建索引映射    
    unique_points, inverse_indices = np.unique(adjusted_points, axis=0, return_inverse=True)

    updated_lines = inverse_indices[lines] # 更新线段索引，但是线段的顺序还是不变
    
    # 确保每条线段的小索引在前，大索引在后
    updated_lines = np.sort(updated_lines, axis=1)

    # updated_lines = np.sort(updated_lines, axis=0)

    # sorted_indices = np.lexsort((updated_lines[:, 1], updated_lines[:, 0]))
    
    # updated_lines = updated_lines[sorted_indices]

    # unique_lines, indices = np.unique(updated_lines, axis=0, return_index=True) # 这里 unique 之后，lines 的顺序会被打乱
    
    # sorted_indices = np.argsort(indices)
    # unique_lines = unique_lines[sorted_indices] # 因此这里对 line 的顺序进行了重新排序，恢复原有的顺序，这是有必要的
    
    lineset['vertices'] = unique_points
    lineset['lines'] = updated_lines
    
    # if return_indices:
    #     return lineset, indices, sorted_indices
    # else:
    return lineset


def remove_duplicate_vertices_and_lines(lineset:dict, return_indices=False, tolerance=0.0001, return_rows_changed=False):
    # 注意，在这部分的代码中，我们并没有对顶点的顺序进行排序，我们只是剔除了重复（三维空间接近）的顶点,
    vertices, lines = lineset['vertices'], lineset['lines']

    # Example tolerance value
    # Adjust this as per your requirement

    # Adjust points based on tolerance
    adjusted_points = np.round(vertices / tolerance) * tolerance
    
    # 移除重复的点并创建索引映射    
    unique_points, inverse_indices = np.unique(adjusted_points, axis=0, return_inverse=True)

    updated_lines = inverse_indices[lines] # 更新线段索引，但是线段的顺序还是不变
    
    # 删除重复的线段，确保每条线段的小索引在前，大索引在后
    # 保存排序前的数组以便后续比较
    updated_lines_before = np.copy(updated_lines)

    updated_lines = np.sort(updated_lines, axis=1)
    # 比较排序前后的行是否发生了变化，生成一个布尔数组，表示每一行是否改变
    changed_rows = np.any(updated_lines_before != updated_lines, axis=1)
    
    unique_lines, indices = np.unique(updated_lines, axis=0, return_index=True) # 这里 unique 之后，lines 的顺序会被打乱
    
    sorted_indices = np.argsort(indices)
    unique_lines = unique_lines[sorted_indices] # 因此这里对 line 的顺序进行了重新排序，恢复原有的顺序，这是有必要的

    lineset['vertices'] = unique_points
    lineset['lines'] = unique_lines
    
    if return_rows_changed:
        return lineset, indices, sorted_indices, changed_rows
    elif return_indices:
        return lineset, indices, sorted_indices
    else:
        return lineset
    
def create_cube_lineset():

    # Step 1: Define the vertices of the cube
    vertices = np.array([
        [0, 0, 0],  # Vertex 0
        [1, 0, 0],  # Vertex 1
        [1, 1, 0],  # Vertex 2
        [0, 1, 0],  # Vertex 3
        [0, 0, 1],  # Vertex 4
        [1, 0, 1],  # Vertex 5
        [1, 1, 1],  # Vertex 6
        [0, 1, 1],  # Vertex 7
    ], dtype=np.float32)

    # Step 2: Define the lines (edges) of the cube
    lines = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom square
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top square
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical lines
    ])

    lineset = {'vertices': vertices, 'lines': lines}

    return lineset

def calculate_length(segment):
    """
    计算线段的长度
    输入的线段是两个点, segment 是一个 2x3 的数组，表示两个点
    """
    return np.linalg.norm(segment[0] - segment[1])

def subdivide_segment(segment):
    """
    将线段细分为两个等长的线段
    输入的线段是两个点, segment 是一个 2x3 的数组，表示两个点
    """
    midpoint = (segment[0] + segment[1]) / 2
    return np.array([segment[0], midpoint]), np.array([midpoint, segment[1]])

def subdivide_longest_low(segments, max_length=256):
    """
    找到并细分最长的线段,直到线段总数达到256
    输入是线段的集合,segments 是一个 nx2x3 的数组，表示 n 个线段，每个线段由两个点组成
    """
    while len(segments) < max_length:
        lengths = np.array([calculate_length(seg) for seg in segments])
        longest_idx = np.argmax(lengths)
        longest_segment = segments[longest_idx]
        new_segments = subdivide_segment(longest_segment)
        segments = np.delete(segments, longest_idx, axis=0)
        segments = np.append(segments, new_segments, axis=0)
    
    # 四舍五入并转换为整数
    # segments = np.round(segments).astype(np.int32)
    
    return segments


def subdivide_longest(segments, max_length=256):
    """使用优先队列细分最长的线段，并移除原始线段"""
    # 使用线段的长度和索引创建优先队列
    pq = [(-calculate_length(seg), i) for i, seg in enumerate(segments)]
    heapq.heapify(pq)
    
    active_segments = list(segments)  # 转换为列表以便可以删除元素
    active_mask = [True] * len(segments)  # 活动线段掩码
    
    while sum(active_mask) < max_length:        
        # 取出最长线段的索引
        _, longest_idx = heapq.heappop(pq)
        
        active_mask[longest_idx] = False
        
        # 细分线段
        new_segments = subdivide_segment(active_segments[longest_idx])

        # 将新线段加入数组和优先队列
        for seg in new_segments:
            idx = len(active_segments)
            heapq.heappush(pq, (-calculate_length(seg), idx))
            active_segments.append(seg)
            active_mask.append(True)

    final_segments = [seg for seg, is_active in zip(active_segments, active_mask) if is_active]

    segments = np.stack(final_segments, axis=0)
    
    # 四舍五入并转换为整数
    # segments = np.round(segments).astype(np.int32)

    return segments

def transform_polyline(points, handleCollinear=True):
    # Step 1: 平移使第一个点到原点
    translated_points = points - points[0]

    # Step 2: 旋转使最后一个点与向量 (1, 0, 0) 同方向
    pn = translated_points[-1]
    target_vector = np.array([1, 0, 0])

    # 检查 pn 是否为零向量
    if np.linalg.norm(pn) == 0:
        raise ValueError("多段线的起点和终点重合，无法确定方向。")

    # 归一化方向向量
    pn_norm = pn / np.linalg.norm(pn)
    target_norm = target_vector / np.linalg.norm(target_vector)

    # 计算点积和角度
    dot_product = np.dot(pn_norm, target_norm)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))  # 防止超出 arccos 定义域

    # 判断是否需要特殊处理
    epsilon = 1e-6
    if np.abs(dot_product + 1) < epsilon:
        if not handleCollinear:
            return None
        
        # 向量相反，选择一个与 pn_norm 正交的任意向量作为旋转轴
        arbitrary_vector = np.array([1, 0, 0])
        if np.allclose(pn_norm, arbitrary_vector) or np.allclose(pn_norm, -arbitrary_vector):
            arbitrary_vector = np.array([0, 1, 0])
        axis = np.cross(pn_norm, arbitrary_vector)
        axis = axis / np.linalg.norm(axis)
        angle = np.pi
    elif np.abs(dot_product - 1) < epsilon:
        if not handleCollinear:
            return None
        
        # 向量同向，不需要旋转
        axis = np.array([0, 0, 1])  # 轴任意，因为角度为 0
        angle = 0.0
    else:
        # 正常计算旋转轴
        axis = np.cross(pn_norm, target_norm)
        axis = axis / np.linalg.norm(axis)

    # Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    # 应用旋转矩阵
    rotated_points = np.dot(translated_points, R.T)

    # Step 3: 缩放使最后一个点到 (1, 0, 0)
    pn_prime = rotated_points[-1]

    # 检查 pn_prime[0] 是否为零，避免除以零
    if pn_prime[0] == 0:
        raise ValueError("无法通过缩放使最后一个点的 x 坐标为 1。")

    scale_factor = 2 / pn_prime[0]
    scaled_points = rotated_points * scale_factor

    # 消除数值误差，使最后一个点的 y 和 z 坐标为零
    scaled_points[:, 1] -= scaled_points[-1, 1]
    scaled_points[:, 2] -= scaled_points[-1, 2]
    scaled_points -= np.array([1, 0, 0])

    # 使得第一个点的坐标为 (0, 0, 0)
    scaled_points[0] = np.array([-1, 0, 0])
    scaled_points[-1] = np.array([1, 0, 0])

    return scaled_points

def transform_polyline_to_start_and_end(points, start_end, handleCollinear=True, epsilon=1e-6):
    start_point = start_end[0]
    end_point = start_end[1]
    
    # Step 1: 平移使第一个点到原点
    translated_points = points - points[0]
    
    # Step 2: 旋转使最后一个点与目标向量同方向
    original_vector = translated_points[-1]
    target_vector = end_point - start_point
    
    # 检查 original_vector 是否为零向量
    if np.linalg.norm(original_vector) == 0:
        raise ValueError("多段线的起点和终点重合，无法确定方向。")
    
    # 归一化方向向量
    original_norm = original_vector / (np.linalg.norm(original_vector) + epsilon)
    target_norm = target_vector / (np.linalg.norm(target_vector) + epsilon)
    
    # 计算旋转轴和角度
    dot_product = np.dot(original_norm, target_norm)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    # 判断是否需要特殊处理
    if np.abs(dot_product + 1) < epsilon:
        if not handleCollinear:
            return None
        
        # 向量相反，选择一个与 original_norm 正交的任意向量作为旋转轴
        arbitrary_vector = np.array([1, 0, 0])
        if np.allclose(original_norm, arbitrary_vector) or np.allclose(original_norm, -arbitrary_vector):
            arbitrary_vector = np.array([0, 1, 0])
        axis = np.cross(original_norm, arbitrary_vector)
        axis = axis / (np.linalg.norm(axis) + epsilon)
        angle = np.pi
    elif np.abs(dot_product - 1) < epsilon:
        if not handleCollinear:
            return None
                
        # 向量同向，不需要旋转
        axis = np.array([0, 0, 1])  # 轴任意，因为角度为 0
        angle = 0.0
    else:
        # 正常计算旋转轴
        axis = np.cross(original_norm, target_norm)
        axis = axis / (np.linalg.norm(axis) + epsilon)
    
    # Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    
    # 应用旋转矩阵
    rotated_points = np.dot(translated_points, R.T)
    
    # Step 3: 缩放使多段线长度匹配目标长度
    original_length = np.linalg.norm(rotated_points[-1])
    target_length = np.linalg.norm(target_vector)
    
    if original_length == 0:
        raise ValueError("多段线长度为零，无法缩放。")
    
    scale_factor = target_length / (original_length + epsilon)
    scaled_points = rotated_points * scale_factor
    
    # Step 4: 平移到指定的起始点位置
    final_points = scaled_points + start_point
    
    return final_points


@deprecated
def transform_polyline_old(points, is_check_order=False):
    if is_check_order:
        if not check_order_single_point(points[0], points[-1]):
            points = np.flip(points, axis=0)
    
    # Step 1: 平移使第一个点到原点
    translated_points = points - points[0]
    
    # Step 2: 旋转使最后一个点与向量 (1, 0, 0) 同方向
    pn = translated_points[-1]
    target_vector = np.array([1, 0, 0])
    
    # 检查 pn 是否为零向量
    if np.linalg.norm(pn) == 0:
        raise ValueError("多段线的起点和终点重合，无法确定方向。")

    # 归一化方向向量
    pn_norm = pn / np.linalg.norm(pn)
    target_norm = target_vector / np.linalg.norm(target_vector)

    # 找到旋转轴和角度
    axis = np.cross(pn_norm, target_norm)
    axis_length = np.linalg.norm(axis)
    axis = axis / axis_length if axis_length != 0 else axis
    angle = np.arccos(np.dot(pn_norm, target_norm))
    
    # Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    
    # 应用旋转矩阵
    rotated_points = np.dot(translated_points, R.T)
    
    # Step 3: 缩放使最后一个点到 (1, 0, 0)
    pn_prime = rotated_points[-1]
    scale_factor = 2 / np.linalg.norm(pn_prime)
    scaled_points = rotated_points * scale_factor - np.array([1, 0, 0])
    
    return scaled_points


def points_within_bounding_box(points, bbox):
    """
    判断一组三维点是否在指定的包围盒内。

    参数：
    points (numpy.ndarray): 形状为 (N, 3) 的数组，其中 N 是点的数量，每个点有 (x, y, z) 坐标。
    min_point (numpy.ndarray): 包围盒的最小点 (x, y, z) 坐标。
    max_point (numpy.ndarray): 包围盒的最大点 (x, y, z) 坐标。

    返回：
    numpy.ndarray: 布尔数组，表示每个点是否在包围盒内。
    """
    min_point, max_point = bbox

    # 检查所有点是否在 min_point 和 max_point 的范围内
    within_min_bounds = np.all(points >= min_point, axis=1)
    within_max_bounds = np.all(points <= max_point, axis=1)

    # 只有在两个范围内的点才算在包围盒内
    within_bounding_box = within_min_bounds & within_max_bounds

    return within_bounding_box

def get_bbox(points):
    """
    Get the tighest fitting 3D bounding box giving a set of points (axis-aligned)
    """

    # 找到每个维度上的最小值和最大值
    min_point = np.min(points, axis=0)
    max_point = np.max(points, axis=0)

    return min_point, max_point


def check_order_single_point(point1, point2):
    """
    判断两个点的顺序是否正确，按照 (z, y, x) 顺序进行比较。

    参数：
    - point1 (numpy.ndarray): 第一个点，包含 (x, y, z) 坐标。
    - point2 (numpy.ndarray): 第二个点，包含 (x, y, z) 坐标。

    返回：
    - bool: 如果 point1 在 point2 之前或者相等，返回 True；否则返回 False。
    """
    # Step 1: 比较 z 坐标
    if point1[2] < point2[2]:
        return True
    elif point1[2] > point2[2]:
        return False

    # Step 2: 如果 z 坐标相同，比较 y 坐标
    if point1[1] < point2[1]:
        return True
    elif point1[1] > point2[1]:
        return False

    # Step 3: 如果 z 和 y 坐标都相同，比较 x 坐标
    if point1[0] <= point2[0]:
        return True
    else:
        return False

def check_order(points1, points2, tolerance=1e-3):

    # points1 = np.where(np.abs(points1) < tolerance, 0, points1)

    points1 = np.round(points1 / tolerance) * tolerance
    points2 = np.round(points2 / tolerance) * tolerance


    # 按照 (z, y, x) 的顺序比较两组点
    # Step 1: 比较 z 坐标
    z_correct = points1[:, 2] < points2[:, 2]

    # Step 2: 对于 z 相同的情况，比较 y 坐标
    y_correct = (points1[:, 2] == points2[:, 2]) & (points1[:, 1] < points2[:, 1])

    # Step 3: 对于 z 和 y 都相同的情况，比较 x 坐标
    x_correct = (points1[:, 2] == points2[:, 2]) & (points1[:, 1] == points2[:, 1]) & (points1[:, 0] <= points2[:, 0])

    # 合并所有条件，得到顺序正确的布尔数组
    correct_order = z_correct | y_correct | x_correct

    return correct_order

# def check_order(points1, points2, tolerance=1e-5):
#     # 使用 np.isclose 计算 z, y, x 是否在容差范围内相等
#     z_equal = np.isclose(points1[:, 2], points2[:, 2], atol=tolerance)
#     y_equal = np.isclose(points1[:, 1], points2[:, 1], atol=tolerance)

#     # 逐步比较 z, y, x，并返回合并后的布尔数组
#     correct_order = (points1[:, 2] < points2[:, 2]) | \
#                     (z_equal & (points1[:, 1] < points2[:, 1])) | \
#                     (z_equal & y_equal & (points1[:, 0] <= points2[:, 0]))

#     return correct_order


def inverse_transform_multi_polyline(transformed_polylines, start_ends, handleCollinear=True):

    edge_points = []
    for i, start_ends_i in enumerate(start_ends):
        if np.linalg.norm(start_ends_i[0] - start_ends_i[1]) == 0:
            continue  # 跳过长度为零的线段
        
        edge_points_i = inverse_transform_polyline(transformed_polylines[i], start_ends_i, handleCollinear)
        if edge_points_i is None:
            return None
        edge_points.append(edge_points_i)
        
    return np.array(edge_points)

def inverse_transform_polyline(transformed_points, start_and_end, handleCollinear=True, epsilon=1e-6):
    tgt_start, tgt_end = start_and_end
    offset = - transformed_points[0]

    lengths = np.linalg.norm(transformed_points[-1] - transformed_points[0])

    # Step 1: inverse the translation
    transformed_points = transformed_points + offset

    # Step 2: calculate the scale factor
    tgt_direction = tgt_end - tgt_start
    scale_factor = np.linalg.norm(tgt_direction)

    # check if the scale factor is zero, avoid division by zero
    if scale_factor == 0:
        raise ValueError("The start and end points are the same, so the scale factor cannot be determined.")

    # Step 3: inverse the scaling
    scaled_back_points = transformed_points * scale_factor / (lengths + epsilon)

    # Step 4: calculate the inverse of the rotation matrix
    target_vector = tgt_direction

    # check if the pn_prime is a zero vector
    pn_prime = scaled_back_points[-1]
    if np.linalg.norm(pn_prime) == 0:
        raise ValueError("The transformed polyline's end point is at the origin, so the direction cannot be determined.")

    # normalize the vector
    pn_prime_norm = pn_prime / (np.linalg.norm(pn_prime) + epsilon)
    target_norm = target_vector / (np.linalg.norm(target_vector) + epsilon)

    # calculate the dot product and angle
    dot_product = np.dot(pn_prime_norm, target_norm)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

    # check if the dot product is close to -1
    if np.abs(dot_product + 1) < epsilon:
        if not handleCollinear:
            return None
        
        # the vectors are opposite, choose an arbitrary vector orthogonal to pn_prime_norm as the rotation axis
        arbitrary_vector = np.array([1, 0, 0])
        if np.allclose(pn_prime_norm, arbitrary_vector) or np.allclose(pn_prime_norm, -arbitrary_vector):
            arbitrary_vector = np.array([0, 1, 0])
        axis = np.cross(pn_prime_norm, arbitrary_vector)
        axis = axis / (np.linalg.norm(axis) + epsilon)
        angle = np.pi
    
    elif np.abs(dot_product - 1) < epsilon:
        if not handleCollinear:
            return None
        
        # the vectors are the same, no rotation is needed
        axis = np.array([0, 0, 1])  # the axis is arbitrary, because the angle is 0
        angle = 0.0

    else:
        # calculate the rotation axis
        axis = np.cross(pn_prime_norm, target_norm)
        axis = axis / (np.linalg.norm(axis) + epsilon)

    # Rodrigues' rotation formula for inverse rotation
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K) 

    # Step 5: inverse the rotation
    rotated_back_points = np.dot(scaled_back_points, R.T)

    # Step 6: inverse the translation
    restored_points = rotated_back_points + tgt_start

    return restored_points

def denorm_curves(
    norm_curves: np.ndarray, 
    corners: np.ndarray
) -> Optional[np.ndarray]:
    """
    use the given corners to denormalize the curves
    """

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
    else:
        return None


@deprecated
def inverse_transform_polyline_old(transformed_points, start_and_end):
    original_start, original_end = start_and_end

    transformed_points = transformed_points + np.array([1, 0, 0])

    # Step 1: 计算原始的缩放因子 s
    original_direction = original_end - original_start
    scale_factor = np.linalg.norm(original_direction)

    # 反向缩放
    scaled_back_points = transformed_points * scale_factor / 2

    # Step 2: 计算旋转矩阵的逆 (转置)
    target_vector = original_direction
    pn_prime = scaled_back_points[-1]
    
    # 归一化向量
    pn_prime_norm = pn_prime / np.linalg.norm(pn_prime)
    target_norm = target_vector / np.linalg.norm(target_vector)
    
    # 旋转轴和角度
    axis = np.cross(pn_prime_norm, target_norm)
    axis_length = np.linalg.norm(axis)
    axis = axis / axis_length if axis_length != 0 else axis
    angle = np.arccos(np.dot(pn_prime_norm, target_norm))
    
    angle = np.arccos(np.clip(np.dot(pn_prime_norm, target_norm), -1.0, 1.0))  # 限制范围以避免数值误差


    # Rodrigues' rotation formula for inverse rotation
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    # R_inv = R.T  # 逆旋转矩阵是原旋转矩阵的转置
    
    # 反向旋转
    rotated_back_points = np.dot(scaled_back_points, R.T)

    # Step 3: 反向平移
    translation_vector = original_start  # 恢复到原始的起始点
    restored_points = rotated_back_points + translation_vector

    return restored_points


def derive_line_edges_from_lines_np(
    lines,
):
    # 将“line segment”作为 graph 的 node，vertex 作为 graph 的 edge，得到用线表示的 graph 中的 edges。
    max_num_lines = lines.shape[0]
    line_edges_vertices_threshold = 1

    all_edges = np.stack(np.meshgrid(
        np.arange(max_num_lines),
        np.arange(max_num_lines),
        indexing='ij'), axis=-1)

    shared_vertices = rearrange(lines, 'i c -> i 1 c 1') == rearrange(lines, 'j c -> 1 j 1 c')
    num_shared_vertices = shared_vertices.any(axis = -1).sum(axis = -1)
    is_neighbor_line = num_shared_vertices == line_edges_vertices_threshold

    line_edge = all_edges[is_neighbor_line]
    line_edge = np.sort(line_edge, axis=1)
    
    # Use unique to find the unique rows in the sorted tensor
    # Since the pairs are sorted, [1, 0] and [0, 1] will both appear as [0, 1] and be considered the same
    line_edge, _ = np.unique(line_edge, return_inverse=True, axis=0)

    # 最终返回 line_edges
    return line_edge # [line_1_idx, line_2_idx]


def edge_points_to_lineset(edge_points):
    sampled_points = edge_points

    # sampled_points = vertices[:, ::4, :]
    num_points = sampled_points.shape[1]
    points_index = np.arange(num_points)
    lines = np.column_stack((points_index, np.roll(points_index, -1)))[:-1]
    bs = sampled_points.shape[0]
    lines = repeat(lines, 'nl c -> b nl c', b=bs)
    # lines = np.repeat(lines, len(sample), axis=0)
    # 再添加一些 offset，因为第二个 batch 的点云是从 256 开始的
    
    # 创建 len(edge_pnts) 个 offset, 每个 offset 都是 256 的倍数
    offset = np.arange(bs) * num_points
    offsets = rearrange(offset, 'b -> b 1 1')
    lines += offsets
    lines = lines.reshape(-1, 2)
    vertices = sampled_points.reshape(-1, 3)
    
    return vertices, lines


def calculate_polyline_lengths(points: np.ndarray, batch_size: int = 15_000_000) -> np.ndarray:
    # 确保数据是 float32 类型，减少内存开销
    points = points.astype(np.float32)

    total_batches = points.shape[0] // batch_size + int(points.shape[0] % batch_size > 0)
    results = []

    for i in range(total_batches):
        batch_points = points[i * batch_size : (i + 1) * batch_size]
        results.append(calculate_polyline_lengths_single_batch(batch_points))

    return np.concatenate(results)



def calculate_polyline_lengths_single_batch(points: np.ndarray) -> np.ndarray:
    """
    计算批量折线(polylines)的长度。
    
    参数:
    points (np.ndarray): 形状为 (batch_size, num_points, 3) 的数组，
                         其中 batch_size 是批量大小，
                         num_points 是每条折线的点数，
                         3 是每个点的三维坐标。

    返回:
    np.ndarray: 形状为 (batch_size,) 的数组，表示每条折线的总长度。
    """
    # 检查输入的数组维度
    if points.ndim != 3 or points.shape[2] != 3:
        # raise ValueError("输入数组必须是形状为 (batch_size, num_points, 3) 的三维数组")
        return None

    # 计算相邻点之间的差
    diffs = points[:, 1:, :] - points[:, :-1, :]

    # 计算每个差向量的欧几里得距离
    distances = np.linalg.norm(diffs, axis=2)

    # 计算每条折线的总长度
    polyline_lengths = distances.sum(axis=1)

    return polyline_lengths


def normalize_edges_points(edge_points, handleCollinear=True, check_bbox=False):
    if check_nan_inf(edge_points):
        return None

    new_edge_points = np.zeros_like(edge_points)

    for i, edge_points_i in enumerate(edge_points):
        norm_edge_points_i = transform_polyline(edge_points_i, handleCollinear=handleCollinear)
        if check_nan_inf(norm_edge_points_i):
            return None
        
        if check_bbox:
            center, extent, R = get_vertices_obb(norm_edge_points_i)
            min_extent = np.min(extent)

            if min_extent > 0.7:
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
        start_end_i = start_end[i]
        new_edge_points_i = transform_polyline_to_start_and_end(edge_points_i, start_end_i, handleCollinear)
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

    norm_edge_points = normalize_edges_points(edge_points_middle_status, handleCollinear=False, check_bbox=check_bbox)

    if norm_edge_points is None:
        return None
    
    return norm_edge_points


def normalize_curves_to_start_end(edge_points, start_end, eps=1e-6):
    # Step 1: Translate to origin
    translated_points = edge_points - edge_points[:, :1, :]

    # Original and target vectors
    original_vec = translated_points[:, -1, :]
    target_vec = start_end[:, 1, :] - start_end[:, 0, :]

    original_length = safe_norm(original_vec, eps)
    original_norm = original_vec / original_length
    target_length = safe_norm(target_vec, eps)
    target_norm = target_vec / target_length

    dot_product = np.einsum('ij,ij->i', original_norm, target_norm).clip(-1, 1)
    angle = np.arccos(dot_product)

    axis = np.cross(original_norm, target_norm)
    axis_norm = safe_norm(axis, eps)
    axis /= axis_norm

    # Handle special cases
    mask_reverse = np.abs(dot_product + 1) < eps
    mask_same = np.abs(dot_product - 1) < eps

    axis[mask_same] = np.array([0, 0, 1])
    axis[mask_reverse] = np.cross(original_norm[mask_reverse], np.array([1, 0, 0]))
    axis[mask_reverse] /= safe_norm(axis[mask_reverse], eps)
    angle[mask_reverse] = np.pi
    angle[mask_same] = 0

    # Rodrigues' formula
    K = np.zeros((len(edge_points), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -axis[:, 2], axis[:, 1]
    K[:, 1, 0], K[:, 1, 2] = axis[:, 2], -axis[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -axis[:, 1], axis[:, 0]

    R = np.eye(3) + np.sin(angle)[:, None, None] * K + (1 - np.cos(angle))[:, None, None] * np.matmul(K, K)
    rotated_points = np.einsum('bij,bkj->bki', R, translated_points)

    scale = target_length / original_length
    scaled_points = rotated_points * scale[:, :, None]

    final_points = scaled_points + start_end[:, :1, :]

    return final_points


def _normalize_curves_step_2(edge_points):
    """
    assume the edge_points is already normalized to the START_END,
    then transform the edge_points from START_END to [[-1, 0, 0],[1, 0, 0]]
    """
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
    norm_edge_points = _normalize_curves_step_2(edge_points_middle_status)
    return norm_edge_points

def remove_unused_vertices(lineset):
    """
    移除未被引用的顶点并更新边的索引。

    参数：
    lineset (dict):

    返回：
    tuple: 更新后的顶点数组和边数组。
    """
    vertices = lineset['vertices']
    edges = lineset['lines']
    # 找到所有被使用的顶点索引
    used_vertex_indices = np.unique(edges.flatten())

    # 创建旧索引到新索引的映射
    index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertex_indices)}

    # 更新边的索引
    updated_edges = np.array([[index_mapping[idx] for idx in edge] for edge in edges])

    # 提取被使用的顶点
    updated_vertices = vertices[used_vertex_indices]

    return {
        'vertices': updated_vertices,
        'lines': updated_edges
    }

def cumulative_polyline_length(polyline):
    """
    计算非闭合 polyline 的累计长度。

    参数:
        polyline (np.ndarray): 形状为 (256, 3) 的 polyline 坐标数组。

    返回:
        np.ndarray: 累计长度数组，形状为 (257,)。
    """
    # 计算每两个相邻点之间的差值
    deltas = np.diff(polyline, axis=0)
    # 计算每段的长度
    seg_lengths = np.linalg.norm(deltas, axis=1)
    # 累计长度，从 0 开始
    cum_length = np.concatenate(([0], np.cumsum(seg_lengths)))
    return cum_length

def sample_polyline(polyline, step=0.02):
    """
    沿着 polyline 进行采样，最小距离为 step。

    参数:
        polyline (np.ndarray): 形状为 (256, 3) 的 polyline 坐标数组。
        step (float): 采样的最小距离。

    返回:
        np.ndarray: 采样后的点云，形状为 (m, 3)，其中 m 取决于 polyline 的总长度。
    """
    cum_length = cumulative_polyline_length(polyline)
    total_length = cum_length[-1]
    
    # 生成采样点的距离位置
    num_samples = int(np.floor(total_length / step))
    sample_distances = np.linspace(0, num_samples * step, num_samples + 1)
    
    # 创建插值函数
    interp_func_x = interp1d(cum_length, polyline[:, 0], kind='linear')
    interp_func_y = interp1d(cum_length, polyline[:, 1], kind='linear')
    interp_func_z = interp1d(cum_length, polyline[:, 2], kind='linear')
    
    # 计算采样点的坐标
    sampled_x = interp_func_x(sample_distances)
    sampled_y = interp_func_y(sample_distances)
    sampled_z = interp_func_z(sample_distances)
    
    sampled_points = np.stack((sampled_x, sampled_y, sampled_z), axis=1)
    return sampled_points


def sample_polylines(polylines, step=0.02, max_points=2048):
    """
    对一组 polylines 进行采样。

    参数:
        polylines (np.ndarray): 形状为 (n, 256, 3) 的 polyline 数据。
        step (float): 采样的最小距离。

    返回:
        np.ndarray: 采样后的点云，形状为 (?, 3)。
    """
    sampled_points_list = []
    n = polylines.shape[0]
    for i in range(n):
        polyline = polylines[i]
        sampled_points = sample_polyline(polyline, step)
        sampled_points_list.append(sampled_points)
    # 将所有采样点合并成一个点云
    all_sampled_points = np.vstack(sampled_points_list)
    
    # 判断哪些行包含 NaN
    valid_rows = ~np.isnan(all_sampled_points).any(axis=1)

    # 通过布尔索引剔除包含 NaN 的行
    all_sampled_points = all_sampled_points[valid_rows]
    
    if all_sampled_points.shape[0] > max_points:
        random_idx = np.random.choice(all_sampled_points.shape[0], max_points, replace=False)
        all_sampled_points = all_sampled_points[random_idx]
    else:
        random_idx = np.random.choice(all_sampled_points.shape[0], max_points, replace=True)
        all_sampled_points = all_sampled_points[random_idx]
    
    
    return all_sampled_points


def sample_points_from_mesh(mesh, num_samples=10000):
    
    # 采样点云及其法向量
    points, face_indices = mesh.sample(num_samples, return_index=True)
    normals = mesh.face_normals[face_indices]

    return {'points': points, 'normals': normals}


def fit_bspline(points, n_samples=64, degree=3, smoothing=0.0, arc_length=False, fine_factor=10):
    """
    use scipy.interpolate.splprep to fit bspline and sample points

    Args:
    - points: (N,3) ndarray, input points
    - n_samples: int, number of output samples (including start and end)
    - degree: int, degree of the bspline, default 3
    - smoothing: float, smoothing factor s, 0 means interpolation
    - arc_length: bool, whether to sample points by arc length
    - fine_factor: int, when arc_length=True, the subdivision factor (N*fine_factor)

    Returns:
    - samples: (n_samples,3) ndarray, sampled points
    """
    # 1) fit bspline
    #   directly pass points.T to splprep, so we don't need to split x,y,z
    tck, _ = splprep(points.T, s=smoothing, k=degree)

    # 2) generate parameter u
    if arc_length:
        # pre-sample on denser u to estimate cumulative arc length
        N = len(points)
        u_fine = np.linspace(0, 1, N * fine_factor)
        coords = np.vstack(splev(u_fine, tck)).T
        seg = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        cumlen = np.concatenate(([0], np.cumsum(seg)))
        # back-interpolate to u_new by equal arc length
        u_new = np.interp(np.linspace(0, cumlen[-1], n_samples), cumlen, u_fine)
    else:
        # parameter domain equidistant
        u_new = np.linspace(0, 1, n_samples)

    # 3) generate final sampled points
    samples = np.vstack(splev(u_new, tck)).T
    return samples

def gaussian_smooth_curve(points, sigma=1.0):
    points = np.array(points)
    points_smoothed = gaussian_filter1d(points.astype(np.float32), sigma=sigma, axis=-2)
    
    # set the first and last points to be the same as the original points
    points_smoothed[...,0,:] = points[...,0,:]
    points_smoothed[..., -1,:] = points[..., -1,:]
    return points_smoothed