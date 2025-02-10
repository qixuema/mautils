import numpy as np
# from misc.vis_samples import vis_curveset

# =============================================================================


def transform_polyline(points):
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
        # 向量相反，选择一个与 pn_norm 正交的任意向量作为旋转轴
        arbitrary_vector = np.array([1, 0, 0])
        if np.allclose(pn_norm, arbitrary_vector) or np.allclose(pn_norm, -arbitrary_vector):
            arbitrary_vector = np.array([0, 1, 0])
        axis = np.cross(pn_norm, arbitrary_vector)
        axis = axis / np.linalg.norm(axis)
        angle = np.pi
    elif np.abs(dot_product - 1) < epsilon:
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

def normalize_edges_points(edge_points, uid=None, is_check_order=False):


    norm_edge_points = np.zeros_like(edge_points)

    for i, edge_points_i in enumerate(edge_points):
        norm_edge_points_i = transform_polyline(edge_points_i)        

        norm_edge_points[i] = norm_edge_points_i

    return norm_edge_points



# =============================================================================

def sort_line_set_vertices_and_lines(line_set:dict, return_sorted_indices=False, tolerance=0.000_1, return_rows_changed=False, edge_points=None):
    points, lines = line_set['vertices'], line_set['lines']
    
    # Example tolerance value
    # tolerance = 0.0005  # Adjust this as per your requirement

    # Adjust points based on tolerance
    adjusted_points = np.round(points / tolerance) * tolerance
    # adjusted_points = points

    if points.shape[-1] == 3:
        # 根据 Z-Y-X 规则对顶点排序
        sorted_indices = np.lexsort((adjusted_points[:, 0], adjusted_points[:, 1], adjusted_points[:, 2]))
    elif points.shape[-1] == 2:
        # 根据 Y-X 规则对顶点排序
        sorted_indices = np.lexsort((adjusted_points[:, 0], adjusted_points[:, 1]))
    else:
        raise NotImplementedError
    
    sorted_points = points[sorted_indices]

    # 计算 inverse indices
    inverse_indices = np.argsort(sorted_indices)

    updated_lines = inverse_indices[lines] # 更新线段索引，但是线段的顺序还是不变


    # lineset = {
    #     'vertices': sorted_points,
    #     'lines': updated_lines
    # }
    # norm_edge_points = normalize_edges_points(edge_points)
    # vis_curveset(lineset, norm_edge_points, vis=True)


    # 保存排序前的数组以便后续比较
    updated_lines_before = np.copy(updated_lines)

    # inner sort
    updated_lines = np.sort(updated_lines, axis=1)
    
    # 比较排序前后的行是否发生了变化，生成一个布尔数组，表示每一行是否改变
    changed_rows = np.any(updated_lines_before != updated_lines, axis=1)    



    # lineset = {
    #     'vertices': sorted_points,
    #     'lines': updated_lines
    # }
    # edge_points[changed_rows] = edge_points[changed_rows, ::-1, :]

    # norm_edge_points = normalize_edges_points(edge_points)
    # vis_curveset(lineset, norm_edge_points, edge_points, vis=True)


    # outer sort
    sorted_indices = np.lexsort((updated_lines[:, 1], updated_lines[:, 0])) # 线段的顺序在这里发生了变化
    updated_lines = updated_lines[sorted_indices]

    # 更新 LineSet 的线段
    line_set['vertices'] = sorted_points
    line_set['lines'] = updated_lines

    if return_rows_changed:
        return line_set, sorted_indices, changed_rows

    if return_sorted_indices:
        return line_set, sorted_indices
    
    return line_set, None


def compute_diffs(lines):
    col_diff = np.diff(lines[:,0], prepend=lines[0,0])
    row_diff = np.clip(lines[:, 1] - lines[:, 0] - 1, 0, None)
    return np.stack([col_diff, row_diff], axis=1)

def reconstruct_lines(diffs):
    # Recover col_start using cumulative sum
    col_start = np.cumsum(diffs[:, 0])
    
    # Recover col_end by adding row_diff + 1 to col_start
    col_end = col_start + diffs[:, 1] + 1
    
    # Stack them to form the lines array
    lines = np.stack([col_start, col_end], axis=1)
    return lines


def has_unreferenced_vertices(curveset):
    """
    检查顶点数组中是否有未被线段数组引用的顶点。

    :param vertices: 顶点数组，形状为 (n, 3)
    :param lines: 线段数组，形状为 (m, 2)，每行包含两个顶点索引
    :return: 如果存在未被引用的顶点，返回 True，否则返回 False
    """
    vertices = curveset['vertices']
    lines = curveset['lines']
    
    # 获取所有线段中用到的顶点索引（去重）
    referenced_vertices = set(lines.flatten())
    
    # 所有顶点的索引集合
    all_vertices = set(range(len(vertices)))
    
    # 找出未被引用的顶点
    unreferenced_vertices = all_vertices - referenced_vertices
    
    # 如果有未引用的顶点，则返回 True
    return len(unreferenced_vertices) > 0



def find_non_isolated_lines_indices(lines):
    # 将所有端点展平为一维数组
    flat_points = lines.flatten()

    # 统计每个顶点的出现次数
    point_counts = np.bincount(flat_points)  # bincount 是高效统计正整数出现次数的函数
    
    # 获取每条线段的两个端点的出现次数
    p1_counts = point_counts[lines[:, 0]]  # lines[:, 0] 是所有线段的第一个顶点
    p2_counts = point_counts[lines[:, 1]]  # lines[:, 1] 是所有线段的第二个顶点

    # 查找非孤立线段：端点至少有一个出现次数大于 1
    non_isolated_mask = (p1_counts > 1) & (p2_counts > 1)

    # 返回非孤立线段的索引
    non_isolated_indices = np.where(non_isolated_mask)[0]
    
    return non_isolated_indices
