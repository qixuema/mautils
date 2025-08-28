import numpy as np
import trimesh
from trimesh.creation import cylinder
from shapely.geometry import Polygon

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
                       random_colors=False, color=(1,0,0)):
    """
    将 segments 转换为棱柱 mesh, 并可选与 base_mesh 合并.

    Args:
        segments: (N,2,3) 线段端点数组
        base_mesh: trimesh.Trimesh 或 None(可选)
        radius: 棱柱半径
        n_sides: 棱柱边数 (5=五棱柱)
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
        chain_color = np.random.rand(3) if random_colors else color
        prism = create_colored_prism_segment(s_p, e_p, radius=radius, color=chain_color, n_sides=n_sides)
        if prism is not None:
            meshes.append(prism)

    if not meshes:
        return None  # 没有任何 mesh

    return trimesh.util.concatenate(meshes)

def _regular_ngon_2d(n_sides: int, radius: float, start_angle: float = 0.0) -> np.ndarray:
    """在 XY 平面生成正 N 边形顶点 (N,2)。"""
    angles = np.linspace(0.0 + start_angle, 2*np.pi + start_angle, n_sides, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.column_stack([x, y])  # (N,2)

def polyline_to_prism(
    polyline: np.ndarray,
    *,
    n_sides: int = 5,
    radius: float = 0.01,
    color: tuple[float, float, float] = (1, 0, 0),
    end_caps: bool = True,
) -> trimesh.Trimesh:
    """
    沿 3D polyline 扫掠一个正 N 边形截面，生成连续棱柱（默认五棱柱）。

    Args:
        polyline: (M,3) 采样点 (至少2个)。
        n_sides:  截面边数（=5 即五棱柱）。
        radius:   截面外接圆半径。
        color:    顶点颜色 RGB [0,1]。
        end_caps: 是否封住首尾 (默认 False).
    """
    polyline = np.asarray(polyline, dtype=float)
    if polyline.ndim != 2 or polyline.shape[1] != 3 or len(polyline) < 2:
        raise ValueError("polyline 需为 (M,3) 且 M>=2")

    # 生成 2D 正 N 边形截面（XY 平面），交由 sweep_polygon 扫掠
    polygon_2d = _regular_ngon_2d(n_sides=n_sides, radius=radius)
    
    polygon_2d = Polygon(polygon_2d)  # Convert numpy array to shapely Polygon

    mesh = trimesh.creation.sweep_polygon(polygon_2d, path=polyline, cap=end_caps)

    # 上色
    rgba = (np.clip(np.array(color + (1.0,)), 0, 1) * 255).astype(np.uint8)
    mesh.visual.vertex_colors = np.tile(rgba, (len(mesh.vertices), 1))

    return mesh

def polylines_to_mesh(polylines, radius=0.1, n_sides=5):
    """
    Args:
        polylines: List[Array(M,3)] 采样点。
        radius:   截面外接圆半径。
    """
    meshes = []
    
    for polyline in polylines:
        color = np.random.rand(3)
        
        prism =  polyline_to_prism(polyline, color=color, radius=radius, n_sides=n_sides)
        
        meshes.append(prism)
        
    mesh = trimesh.util.concatenate(meshes)
    
    return mesh
