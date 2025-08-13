import numpy as np
import trimesh
from trimesh.creation import cylinder

def _rgba255(color):
    """
    把 [0,1] 或 [0,255] 的 3/4 通道颜色转成 uint8 RGBA(4通道)
    """
    c = np.asarray(color, dtype=np.float64).flatten()
    if c.max() <= 1.0:
        c = (c * 255.0).round()
    if c.size == 3:
        c = np.concatenate([c, [255.0]])
    return np.clip(c, 0, 255).astype(np.uint8)

def create_colored_prism_segment(s_p, e_p, radius=0.1, color=(1,0,0), n_sides=5):
    """
    用 trimesh.creation.cylinder 创建从 s_p 到 e_p 的 n 边柱（默认五棱柱）。
    """
    s_p = np.asarray(s_p, dtype=np.float64)
    e_p = np.asarray(e_p, dtype=np.float64)
    if np.linalg.norm(e_p - s_p) <= 1e-3:
        return None

    # cylinder 会自动生成沿 segment 的网格
    mesh = cylinder(
        radius=radius,
        sections=n_sides,       # 五棱柱
        segment=(s_p, e_p)      # 端点
    )

    # 上色
    mesh.visual.vertex_colors = np.tile(_rgba255(color), (mesh.vertices.shape[0], 1))
    return mesh

def segments_to_prisms(segments, base_mesh=None, radius=0.01, n_sides=5,
                       random_colors=True, default_color=(1,0,0)):
    """
    将 segments 转换为棱柱 mesh，并可选与 base_mesh 合并。

    Args:
        segments: (N,2,3) 线段端点数组
        base_mesh: trimesh.Trimesh 或 None（可选）
        radius: 棱柱半径
        n_sides: 棱柱边数（5=五棱柱）
        random_colors: 是否随机颜色
        default_color: 非随机颜色时使用的 RGB (0~1)

    Returns:
        trimesh.Trimesh 合并后的网格
    """
    meshes = []

    # 如果有 base_mesh，先放进去
    if base_mesh is not None:
        meshes.append(base_mesh)

    # 每个 segment 生成一个棱柱
    for seg in segments:
        s_p, e_p = seg
        color = np.random.rand(3) if random_colors else default_color
        prism = create_colored_prism_segment(s_p, e_p, radius=radius, color=color, n_sides=n_sides)
        if prism is not None:
            meshes.append(prism)

    if not meshes:
        return None  # 没有任何 mesh

    return trimesh.util.concatenate(meshes)