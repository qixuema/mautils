import numpy as np


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
