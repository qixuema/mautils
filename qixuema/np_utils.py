import logging

import numpy as np
from typing import Iterable, Optional, Tuple, Union

logger = logging.getLogger(__name__)

def check_nan_inf(data) -> bool:
    return bool(np.any(np.isnan(data)) or np.any(np.isinf(data)))

def is_close(a, b, atol=1e-5):
    return np.isclose(a, b, atol=atol)

def safe_norm(v: np.ndarray, eps: float) -> np.ndarray:
    """
    Compute the L2 norm of `v` along the last axis and clamp it to be at least `eps`.
    """
    return np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)

def normalize(vector, eps=1e-6):
    return vector / safe_norm(vector, eps)


def check_npy_file(file_path):
    try:
        np.load(file_path)
    except Exception:
        return False
    return True

def interpolate_1d(
    t,
    data,
):
    """
    对给定的数据进行一维线性插值。
    参数:
    t (Tensor): 插值坐标，范围在 [0, 1] 之间，形状为 (n,)。
    data (Tensor): 原始数据，形状为 (num_points, n_channels)。
    返回:
    Tensor: 插值后的数据，形状为 (n, n_channels)。
    """
    assert len(t.shape) == 1, "t 应该是一个一维张量，形状为 (n,)"
    assert len(data.shape) == 2, "data 应该是一个二维张量，形状为 (num_points, channels)"

    num_reso = data.shape[0]
    t = t * (num_reso - 1)

    left = np.floor(t).astype(np.int32)
    right = np.ceil(t).astype(np.int32)
    alpha = t - left

    left = np.clip(left, a_min=0, a_max=num_reso - 1)
    right = np.clip(right, a_min=0, a_max=num_reso - 1)

    left_values = data[left, :]
    right_values = data[right, :]

    alpha = alpha[:, None]
    return (1 - alpha) * left_values + alpha * right_values

def interpolate_1d_batch(t, data):
    """
    对一批数据进行一维线性插值。
    参数:
    t: 插值坐标，范围在 [0, 1] 之间，形状为 (B, n)。
    data: 原始数据，形状为 (B, num_points, n_channels)。
    返回:
    形状为 (B, n, n_channels) 的插值结果。
    """
    assert t.ndim == 2
    assert data.ndim == 3
    B, n = t.shape
    _, num_points, _ = data.shape

    t_scaled = t * (num_points - 1)

    left = np.clip(np.floor(t_scaled).astype(np.int32), 0, num_points - 1)
    right = np.clip(np.ceil(t_scaled).astype(np.int32), 0, num_points - 1)
    alpha = (t_scaled - left)[..., None]

    batch_indices = np.arange(B)[:, None]
    left_values = data[batch_indices, left]
    right_values = data[batch_indices, right]

    return (1 - alpha) * left_values + alpha * right_values

def normalize_vertices(vertices, scale=1.0):
    bbmin, bbmax = vertices.min(0), vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    denom = (bbmax - bbmin).max()

    if denom <= 1e-6:
        logger.warning("denom is too small: %s", denom)
        return None, None, None

    scale = 2.0 * scale / denom
    vertices = (vertices - center) * scale
    return vertices, center, scale


def deduplicate_lines(lines):
    """
    deduplicate lines, return unique lines without change line direction
    lines: (N,2) int
    """
    
    sorted_lines = np.sort(lines, axis=1) # inner sort
    _, indices = np.unique(sorted_lines, axis=0, return_index=True)
    unique_lines = lines[np.sort(indices)]
    
    return unique_lines

def deduplicate_faces(faces, faces_uv=None):
    """
    deduplicate faces by xyz, return unique faces without change face normal
    
    Args:
        faces: (N,3) int array of face indices
        faces_uv: optional (N,3) int array of UV face indices, synchronized with faces
    
    Returns:
        faces_unique: (M,3) int array of unique faces
        faces_uv_unique: (M,3) int array of unique UV faces, only returned if faces_uv is provided
    """
    sorted_faces = np.sort(faces, axis=1)
    _, indices = np.unique(sorted_faces, axis=0, return_index=True)
    keep_idx = np.sort(indices)
    faces_unique = faces[keep_idx]

    if faces_uv is not None:
        faces_uv_unique = faces_uv[keep_idx]
        return faces_unique, faces_uv_unique
    
    return faces_unique

def clean_invalid_faces(faces, faces_uv=None):
    """
    remove invalid faces, return valid faces (and UV faces if provided)
    invalid faces are those with duplicate vertex indices (e.g., [0,1,0] or [1,1,2])
    
    Args:
        faces: (N,3) int array of face indices
        faces_uv: optional (N,3) int array of UV face indices, synchronized with faces
    
    Returns:
        faces_valid: (M,3) int array of valid faces
        faces_uv_valid: (M,3) int array of valid UV faces, only returned if faces_uv is provided
    """
    f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]
    invalid = (f0 == f1) | (f0 == f2) | (f1 == f2)
    valid_mask = ~invalid
    
    faces_valid = faces[valid_mask]
    
    if faces_uv is not None:
        faces_uv_valid = faces_uv[valid_mask]
        return faces_valid, faces_uv_valid
    
    return faces_valid

def clean_invalid_lines(lines):
    """
    remove invalid lines, return valid lines indices
    lines: (N,2) int
    """
    return lines[lines[:, 0] != lines[:, 1]]


def discretize(
    t,
    n_bits = 8,
    continuous_range = (-1, 1),
):
    lo, hi = continuous_range
    assert hi > lo
    
    num_discrete = 2 ** n_bits
    
    t = (t - lo) / (hi - lo)
    t *= num_discrete
    t -= 0.5

    return t.round().astype(np.int32).clip(min = 0, max = num_discrete - 1)


def undiscretize(
    t,
    *,
    n_bits = 8,
    continuous_range = (-1, 1),
) :
    lo, hi = continuous_range
    assert hi > lo

    num_discrete = 2 ** n_bits

    t = t.astype(np.float32)

    t += 0.5
    t /= num_discrete
    return t * (hi - lo) + lo


def rotation_matrix_x(theta):
    theta_rad = np.radians(theta)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    return np.array([
        [1, 0, 0],
        [0, cos_t, -sin_t],
        [0, sin_t, cos_t]
    ])

def rotation_matrix_y(theta):
    theta_rad = np.radians(theta)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    return np.array([
        [cos_t, 0, sin_t],
        [0, 1, 0],
        [-sin_t, 0, cos_t]
    ])

def rotation_matrix_z(theta):
    theta_rad = np.radians(theta)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    return np.array([
        [cos_t, -sin_t, 0],
        [sin_t, cos_t, 0],
        [0, 0, 1]
    ])

def pad_sequence_np(
    sequences: Iterable[np.ndarray],
    maxlen: Optional[int] = None,
    dtype: Optional[np.dtype] = None,
    pad_value: Union[int, float] = 0,
    padding: str = "post",
    truncating: str = "post",
    return_lengths: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    将不等长序列 padding 成 (N, M, *feat_shape) 的规则数组。

    Args:
        sequences: 可迭代的数组，每个元素形状为 (Ti, *feat_shape)；标量会视为 Ti=1。
        maxlen: 目标长度 M；若为 None，则取所有 Ti 的最大值。
        dtype: 结果 dtype；默认根据 pad_value 与各序列 dtype 推断。
        pad_value: 补齐用的值。
        padding: "post" 或 "pre" (补在尾部/头部)。
        truncating: 当 Ti > M 时从 "post" (尾部截掉) 或 "pre" (头部截掉)。
        return_lengths: 是否返回每条序列真实放入的长度 (min(Ti, M))。

    Returns:
        padded: 形状 (N, M, *feat_shape) 的数组。
        lengths (可选): 形状 (N,) 的实际长度。
    """
    seqs = [np.asarray(s) for s in sequences]
    N = len(seqs)
    if N == 0:
        out = np.empty((0, 0), dtype=(dtype or np.asarray(pad_value).dtype))
        return (out, np.empty((0,), dtype=int)) if return_lengths else out

    def seq_len_and_tail(s: np.ndarray):
        if s.ndim == 0:
            return 1, ()
        return s.shape[0], s.shape[1:]

    tail_shape = None
    lengths_raw = []
    for s in seqs:
        L, tail = seq_len_and_tail(s)
        lengths_raw.append(L)
        if tail_shape is None:
            tail_shape = tail
        elif tail_shape != tail:
            raise ValueError(f"All sequences must share the same feature shape after axis 0. "
                             f"Got {tail_shape} vs {tail}.")

    lengths_raw = np.asarray(lengths_raw, dtype=int)
    if maxlen is None:
        maxlen = int(lengths_raw.max())

    if dtype is None:
        dtype = np.result_type(pad_value, *[s.dtype for s in seqs])

    out = np.full((N, maxlen, *tail_shape), pad_value, dtype=dtype)
    used_lengths = np.minimum(lengths_raw, maxlen)

    for i, s in enumerate(seqs):
        if s.ndim == 0:
            s = s.reshape(1, *tail_shape)

        L = used_lengths[i]
        if L == 0:
            continue

        if truncating == "post":
            s_trunc = s[:L]
        elif truncating == "pre":
            s_trunc = s[-L:]
        else:
            raise ValueError("truncating must be 'post' or 'pre'")

        if padding == "post":
            out[i, :L] = s_trunc
        elif padding == "pre":
            out[i, -L:] = s_trunc
        else:
            raise ValueError("padding must be 'post' or 'pre'")

    return (out, used_lengths) if return_lengths else out

# 也许可以写一个关于 polyline 的 class
def polyline_length(polyline: np.ndarray) -> float:
    diffs = np.diff(polyline, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))



def boundary_edges(faces: np.ndarray) -> np.ndarray:
    """Return boundary edges (those appearing exactly once in the face list)."""
    if faces.size == 0:
        return np.array([], dtype=np.int64)

    edges = faces_to_edges(faces)
    edges = np.sort(edges, axis=1)

    uniq_edges, counts = np.unique(edges, axis=0, return_counts=True)
    return uniq_edges[counts == 1]

def boundary_vertex_indices(faces: np.ndarray) -> np.ndarray:
    """Return unique boundary vertex indices (sorted)."""
    return np.unique(boundary_edges(faces).ravel())

def check_finite(data: list[np.ndarray]) -> bool:
    return all(np.isfinite(arr).all() for arr in data)


def safe_gather(
    source: np.ndarray,
    index_array: np.ndarray,
    pad_value: int | list[int],
    fill_value: float = 0.
):
    """Index `source` by `index_array`, replacing `pad_value` positions with `fill_value`."""
    if not np.issubdtype(index_array.dtype, np.integer):
        raise ValueError("index_array must be integer dtype")

    D = source.shape[1]

    if isinstance(pad_value, int):
        pad_value = [pad_value]
    elif isinstance(pad_value, list):
        pad_value = np.array(pad_value)

    mask = np.isin(index_array, pad_value)

    safe_indices = index_array.copy()
    safe_indices[mask] = 0

    result = source[safe_indices.flatten()].reshape(index_array.shape + (D,))

    if np.isscalar(fill_value):
        fill_value = np.full((D,), fill_value, dtype=source.dtype)
    else:
        fill_value = np.array(fill_value, dtype=source.dtype)
        if fill_value.shape != (D,):
            raise ValueError(f"fill_value shape should be ({D},), got {fill_value.shape}")

    result[mask] = fill_value
    return result



def cut_before_value(arr: np.ndarray, value, axis: int = 0):
    """Find the first slice along `axis` equal to `value` and return everything before it."""
    arr = np.asarray(arr)
    value = np.asarray(value)

    if arr.ndim == 0:
        raise ValueError("arr must be at least 1-D")

    if axis < 0:
        axis += arr.ndim
    if not (0 <= axis < arr.ndim):
        raise ValueError(f"axis out of range: {axis} for arr.ndim={arr.ndim}")

    a = np.moveaxis(arr, axis, 0)
    eq = (a == value)

    if a.ndim == 1:
        matches = eq
    else:
        matches = np.all(eq, axis=tuple(range(1, a.ndim)))

    if np.any(matches):
        first_idx = int(np.argmax(matches))
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(0, first_idx)
        return arr[tuple(sl)], first_idx
    return arr.copy(), None


def tolerant_lexsort(vertices, eps=1e-3, tie_break=False):
    """
    抗噪字典序排序 (主序:z→y→x).
    - eps: 标量容差，三个轴共用.
    - tie_break: 是否用原始值(z→y→x)作为平局破坏，保证同格内顺序稳定.
    返回: 排序后的索引.
    """
    v = np.asarray(vertices, dtype=np.float64)

    # 一次性量化到整数格；乘以倒数避免三次除法
    q = np.rint(v * (1.0 / eps)).astype(np.int64, copy=False)

    if tie_break:
        # 主键: 量化 z→y→x；次键: 原始 z→y→x
        return np.lexsort((v[:, 0], v[:, 1], v[:, 2],
                           q[:, 0], q[:, 1], q[:, 2]))
    else:
        # 仅用量化键 (更快)
        return np.lexsort((q[:, 0], q[:, 1], q[:, 2]))

def dedup_with_mean(
    vertices, 
    tol=1e-6, 
    return_index: bool = False,
    dtype=np.float64  
):
    """
    按 xyz/tolerance 的 round 分组；单点组保持原值，多点组取原始点的均值。
    
    Parameters
    ----------
    xyz : (N, D) array_like
    tolerance : float 或 (D,) array_like
    
    Returns
    -------
    xyz_unique : (K, D) ndarray
        每个分组的代表点（均值）
    inverse : (N,) ndarray of int
        每个原始点映射到 xyz_unique 的索引
    """
    vertices = np.asarray(vertices, dtype=dtype)
    tol = np.asarray(tol, dtype=dtype)
    if tol.ndim == 0:
        tol = np.full(vertices.shape[1], tol)
    if np.any(tol <= 0):
        raise ValueError("tolerance must be > 0")
    
    keys = np.round(vertices / tol).astype(np.int64)  # 分组键（格子坐标）
    _, idx, inv, counts = np.unique(keys, axis=0, return_index=True, return_inverse=True, return_counts=True)

    sums = np.zeros((counts.size, vertices.shape[1]), dtype=dtype)
    np.add.at(sums, inv, vertices)                    # 按组累加
    xyz_unique = sums / counts[:, None]          # 组均值（单元素组即原值）
    if return_index:
        return xyz_unique, idx, inv
    return xyz_unique, inv


def remap_with_pad(
    mapping: np.ndarray,
    indices: np.ndarray,
    pad_value: int = -1,
) -> np.ndarray:
    """
    对 indices 中的索引做映射: indices==-1(或 pad_value) 保持为 pad_value,否则替换为 mapping[indices]。
    - indices: 任意形状，整数 dtype
    - mapping: 一维映射表，长度为 N
    - pad_value: padding 的标记值 (默认 -1)

    返回: 与 indices 同形状的整数数组
    """
    indices = np.asarray(indices)
    mapping = np.asarray(mapping)

    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("source 必须是整数类型")
    if mapping.ndim != 1:
        raise ValueError("mapping 必须是一维数组")

    # 输出 dtype 兼容 pad_value
    out_dtype = np.result_type(indices.dtype, mapping.dtype, type(pad_value))
    out = np.empty(indices.shape, dtype=out_dtype)

    mask = (indices == pad_value)
    out[mask] = pad_value

    if (~mask).any():
        idx = indices[~mask]
        # 边界检查（可选，去掉也行）
        if idx.min() < 0 or idx.max() >= mapping.shape[0]:
            raise IndexError("索引越界：存在 <0 或 >= len(mapping) 的值")
        out[~mask] = mapping[idx]

    return out


def faces_to_edges(faces, return_index=False):
    """
    Given a list of faces (n,3), return a list of edges (n*3,2)

    Parameters
    -----------
    faces : (n, 3) int
      Vertex indices representing faces

    Returns
    -----------
    edges : (n*3, 2) int
      Vertex indices representing edges
    """
    faces = np.asanyarray(faces, np.int64)

    # each face has three edges
    edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2))

    if return_index:
        # edges are in order of faces due to reshape
        face_index = np.tile(np.arange(len(faces)), (3, 1)).T.reshape(-1)
        return edges, face_index
    return edges

def triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    vertices: (V,3)
    faces:    (F,3)  dtype=int, 每行是三角面顶点索引
    return:   (F,)   每个三角形的面积
    """
    v = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int32)

    if v.shape[1] == 2:
        v = np.pad(v, ((0, 0), (0, 1)))  # z=0

    a = v[f[:, 0]]
    b = v[f[:, 1]]
    c = v[f[:, 2]]

    ab = b - a
    ac = c - a
    cross = np.cross(ab, ac)            # (F,3)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    return areas

def order_and_sort_edges(seam_edges: np.ndarray) -> np.ndarray:
    """Sort edges: inner sort (min, max) per row, then lexsort by (col0, col1)."""
    a = np.asarray(seam_edges, dtype=np.int64)
    lo = np.minimum(a[:, 0], a[:, 1])
    hi = np.maximum(a[:, 0], a[:, 1])
    a = np.column_stack((lo, hi))
    idx = np.lexsort((a[:, 1], a[:, 0]))
    return a[idx]


def np_pad_to(arr, shape, fill=0):
    arr = np.asarray(arr)
    out = np.full(shape, fill, dtype=arr.dtype)
    if arr.shape[0] > shape[0]:
        arr = arr[:shape[0]]
    
    slices = tuple(slice(0, min(s, arr.shape[i])) for i, s in enumerate(shape))
    out[slices] = arr[slices]
    return out

def pad_to_canvas(
    img: np.ndarray,
    out_h: int,
    out_w: int | None = None,
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    将图像居中放置到指定画布大小 (out_h, out_w)，四周用 pad_color 填充。
    如果 out_w 为 None, 则默认做成正方形画布 out_h x out_h。
    """
    if out_w is None:
        out_w = out_h

    h, w = img.shape[:2]
    if h > out_h or w > out_w:
        raise ValueError(f"target size ({out_h}, {out_w}) smaller than image ({h}, {w})")

    # 需要填充的总量
    pad_h = out_h - h
    pad_w = out_w - w

    # 上下 / 左右分配（尽量居中）
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    if img.ndim == 3:
        pad_width = ((top, bottom), (left, right), (0, 0))
    else:
        pad_width = ((top, bottom), (left, right))

    return np.pad(
        img,
        pad_width=pad_width,
        mode="constant",
        constant_values=pad_value
    )
    
def resize_numpy_nn(
    img: np.ndarray,
    scale: float | None = None,
    size: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    纯 numpy 实现的最近邻缩放：
    - img: (H, W, C) 或 (H, W)
    - scale: 按比例缩放
    - size: (out_h, out_w)，指定输出大小
    二选一: scale 或 size 必须提供一个。
    """
    if (scale is None) == (size is None):
        raise ValueError("必须二选一：要么指定 scale，要么指定 size")

    in_h, in_w = img.shape[:2]

    if size is not None:
        out_h, out_w = size
    else:
        out_h = max(1, int(in_h * scale))
        out_w = max(1, int(in_w * scale))

    # 计算输出网格对应到输入的坐标（最近邻）
    # 这里使用 np.linspace 覆盖 [0, in_h) / [0, in_w) 区间
    row_coords = (np.linspace(0, in_h - 1, out_h)).astype(np.int64)
    col_coords = (np.linspace(0, in_w - 1, out_w)).astype(np.int64)

    # 利用广播构造索引网格
    out = img[row_coords[:, None], col_coords[None, ...]]

    return out