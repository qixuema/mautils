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
    color: np.ndarray = np.array([1, 0, 0]),
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
    rgba = (np.clip(np.concatenate([color, [1.0]]), 0, 1) * 255).astype(np.uint8)
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

def get_vertices_obb(
    vertices: np.ndarray,
    *,
    jitter: float = 1e-6,
    rng: np.random.Generator | int | None = None,
    robust_fallback: bool = True,
):
    """
    计算点集的有向包围盒 (OBB).

    Args:
        vertices: (N, 3) 点集。
        jitter:   相对抖动幅度（相对于点云包围盒对角线），
                  None 表示不抖动；仅在失败时（或退化）才使用回退。
        rng:      随机源 (np.random.Generator 或 seed)，确保可复现。
        robust_fallback:
                  当 `oriented_bounds` 失败时，使用 PCA 回退。

    Returns:
        centroid: (3,) OBB 中心
        extents:  (3,) OBB 尺寸 (沿盒子局部 x/y/z)
        T:        (4,4) 齐次变换矩阵 (从 OBB 局部到世界)
    """
    V = np.asarray(vertices, dtype=np.float64)
    if V.ndim != 2 or V.shape[1] != 3 or len(V) < 4:
        raise ValueError("`vertices` 必须是形状 (N, 3), 且 N >= 4")

    if np.allclose(V, V[0], atol=1e-8):
        raise ValueError("所有点都相同")

    # 清除异常值 (NaN/Inf)
    if not np.isfinite(V).all():
        V = V[np.all(np.isfinite(V), axis=1)]
        if len(V) == 0:
            raise ValueError("所有点都包含 NaN/Inf")

    def obb_by_trimesh(points: np.ndarray):
        T, extents = trimesh.bounds.oriented_bounds(points, ordered=False)  # 轻量且快
        centroid = T[:3, 3]
        return centroid, extents, T

    # 首次尝试：不加噪声，直接计算
    try:
        return obb_by_trimesh(V)
    except Exception:
        pass  # 可能是退化/奇异，下面尝试回退

    # 可选：相对尺度的轻微抖动（可复现）
    if jitter is not None and jitter > 0:
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)  # rng 可以是 seed 或 None
        scale = np.linalg.norm(V.ptp(axis=0))  # 点云总体尺度
        eps = (jitter * scale) if scale > 0 else jitter
        V_jit = V + rng.uniform(-eps, eps, size=V.shape)
        try:
            return obb_by_trimesh(V_jit)
        except Exception:
            pass

    # 最后回退：PCA OBB（非最小体积，但稳定、可复现）
    if robust_fallback:
        c = V.mean(axis=0)
        X = V - c
        # SVD 比协方差特征分解更稳
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        R = Vt  # 3x3，行向量为主轴
        if np.linalg.det(R) < 0:  # 保证右手系
            R[2] *= -1
        proj = X @ R.T
        pmin = proj.min(axis=0)
        pmax = proj.max(axis=0)
        extents = pmax - pmin
        center_local = 0.5 * (pmin + pmax)
        centroid = c + R.T @ center_local
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = centroid
        return centroid, extents, T

    # 如果不启用回退，就把第一次异常抛出去
    # （这里人为抛一个通用错误）
    raise RuntimeError("Failed to compute OBB from given vertices.")


def points_to_spheres(
    points: np.ndarray,
    *,
    radius: float = 0.005,
    subdivisions: int = 1,
) -> list[trimesh.Trimesh]:
    """
    Args:
        points: (N, 3)
        radius: float
        subdivisions: int

    Returns:
        list[trimesh.Trimesh]
    """

    # 存放所有球体mesh的列表
    spheres = []

    for point in points:
        # 创建单位球
        sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
        
        # 移动球体到目标点
        sphere.apply_translation(point)

        # 加入列表
        spheres.append(sphere)
        
    return spheres

def to_mesh(vertices, faces, post_process=False):
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    if post_process:
        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())
        mesh.fix_normals()
    
    return mesh

def uniform_pc_sampling(mesh, pc_num_total=20480):
    points, face_idx = mesh.sample(pc_num_total, return_index=True)

    normals = mesh.face_normals[face_idx]

    pc_normal = np.concatenate([points, normals], axis=-1, dtype=np.float16)

    return pc_normal