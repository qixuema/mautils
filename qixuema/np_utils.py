import numpy as np
from typing import Iterable, Tuple, Optional, Union, List

def check_nan_inf(data):
    contains_nan_inf = np.any(np.isnan(data)) or np.any(np.isinf(data))
    if contains_nan_inf:
        return True
    return False

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
        # try to load the file
        data = np.load(file_path)
    except Exception as e:
        # if the file is not a valid .npy file, return False
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
    # 检查输入张量的形状是否符合预期
    assert len(t.shape) == 1, "t 应该是一个一维张量，形状为 (n,)"
    assert len(data.shape) == 2, "data 应该是一个二维张量，形状为 (num_points, channels)"

    # coords range [0, 1]
    num_reso = data.shape[0]
    # 将插值坐标 t 映射到 [0, num_reso - 1] 范围内
    t = t * (num_reso - 1)

    # Perform linear interpolation
    left = np.floor(t).astype(np.int32)
    right = np.ceil(t).astype(np.int32)
    alpha = t - left

    # 防止超出索引范围

    left = np.clip(left, a_min=0, a_max=num_reso - 1)
    right = np.clip(right, a_min=0, a_max=num_reso - 1)


    # 使用 NumPy 索引获取左侧和右侧的值
    # 数据形状为 (batch_size, num_points, n_channels)，需要用高级索引获取值
    left_values = data[left, :]
    right_values = data[right, :]

    # 扩展 alpha 维度，以匹配 left_values 和 right_values 的形状
    alpha = alpha[:, None]

    interpolated = (1 - alpha) * left_values + alpha * right_values
    
    return interpolated

def interpolate_1d_batch(
    t,
    data,
):
    """
    对一批数据进行一维线性插值。

    参数:
    t (Tensor): 插值坐标，范围在 [0, 1] 之间，形状为 (B, n)。
    data (Tensor): 原始数据，形状为 (B, num_points, n_channels)。

    返回:
    Tensor: 插值后的数据，形状为 (B, n, n_channels)。
    """
    assert t.ndim == 2, "t 应该是一个二维张量，形状为 (B, n)"
    assert data.ndim == 3, "data 应该是一个三维张量，形状为 (B, num_points, channels)"
    B, n = t.shape
    _, num_points, n_channels = data.shape

    # 将 t 映射到 [0, num_points - 1]
    t_scaled = t * (num_points - 1)

    left = np.floor(t_scaled).astype(np.int32)
    right = np.ceil(t_scaled).astype(np.int32)
    alpha = t_scaled - left

    # clip 防止越界
    left = np.clip(left, 0, num_points - 1)
    right = np.clip(right, 0, num_points - 1)

    # 用高级索引取数据：每个 batch 要取自己的点
    batch_indices = np.arange(B)[:, None]  # shape: (B, 1)

    left_values = data[batch_indices, left]     # shape: (B, n, n_channels)
    right_values = data[batch_indices, right]   # shape: (B, n, n_channels)

    # alpha shape: (B, n) => (B, n, 1) for broadcasting
    alpha = alpha[..., None]

    interpolated = (1 - alpha) * left_values + alpha * right_values

    return interpolated

def normalize_vertices(vertices, scale=1.0):
    bbmin, bbmax = vertices.min(0), vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    denom = (bbmax - bbmin).max()

    if denom <= 1e-6:
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
    """
    sorted_faces = np.sort(faces, axis=1)
    _, indices = np.unique(sorted_faces, axis=0, return_index=True)
    keep_idx = np.sort(indices)
    faces_unique = faces[keep_idx]

    if faces_uv is not None:
        faces_uv_unique = faces_uv[keep_idx]
        return faces_unique, faces_uv_unique
    
    return faces_unique

def clean_invalid_faces(faces):
    """
    remove invalid faces, return valid faces indices
    faces: (N,3) int
    """
    f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]
    invalid = (f0 == f1) | (f0 == f2) | (f1 == f2)
    return faces[~invalid]

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
    padding: str = "post",       # "post" 左对齐 (在尾部补) ，"pre" 右对齐 (在头部补) 
    truncating: str = "post",    # 超过 maxlen 时从哪边截断："post" 前段保留，"pre" 后段保留
    return_lengths: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    将不等长序列 padding 成 (N, M, *feat_shape) 的规则数组。

    Args:
        sequences: 可迭代的数组，每个元素形状为 (Ti, *feat_shape)；标量会视为 Ti=1。
        maxlen: 目标长度 M；若为 None，则取所有 Ti 的最大值。
        dtype: 结果 dtype；默认根据 pad_value 与各序列 dtype 推断。
        pad_value: 补齐用的值。
        padding: "post" 或 "pre" (补在尾部/头部) 。
        truncating: 当 Ti > M 时从 "post" (尾部截掉) 或 "pre" (头部截掉) 。
        return_lengths: 是否返回每条序列真实放入的长度 (min(Ti, M)) 。

    Returns:
        padded: 形状 (N, M, *feat_shape) 的数组。
        lengths (可选): 形状 (N,) 的实际长度。
    """
    seqs = [np.asarray(s) for s in sequences]
    N = len(seqs)
    if N == 0:
        out = np.empty((0, 0), dtype=(dtype or np.asarray(pad_value).dtype))
        return (out, np.empty((0,), dtype=int)) if return_lengths else out

    # 统一特征维 (length 维是轴 0) ，标量当作长度为 1 的序列
    def seq_len_and_tail(s: np.ndarray):
        if s.ndim == 0:
            return 1, ()
        return s.shape[0], s.shape[1:]

    # 决定 feat_shape，并校验一致性
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

    # 推断 dtype
    if dtype is None:
        dtype = np.result_type(pad_value, *[s.dtype for s in seqs])

    # 预分配输出
    out = np.full((N, maxlen, *tail_shape), pad_value, dtype=dtype)
    used_lengths = np.minimum(lengths_raw, maxlen)

    # 填充
    for i, s in enumerate(seqs):
        # 标量视作 (1, *tail_shape)
        if s.ndim == 0:
            s = s.reshape(1, *tail_shape)

        L = used_lengths[i]
        if L == 0:
            continue

        # 截断
        if truncating == "post":
            s_trunc = s[:L]
        elif truncating == "pre":
            s_trunc = s[-L:]
        else:
            raise ValueError("truncating must be 'post' or 'pre'")

        # 放置
        if padding == "post":
            out[i, :L] = s_trunc
        elif padding == "pre":
            out[i, -L:] = s_trunc
        else:
            raise ValueError("padding must be 'post' or 'pre'")

    return (out, used_lengths) if return_lengths else out

# 也许可以写一个关于 polyline 的 class
def polyline_length(polyline: np.ndarray) -> float:
    diffs = np.diff(polyline, axis=0)        # shape = (N-1, D)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    return np.sum(seg_lengths)


def boundary_vertex_indices(faces_idx: np.ndarray) -> np.ndarray:
    """
    返回边界顶点的索引 (升序、唯一) 。
    faces_idx: (M,3) 三角形面片的顶点索引 (int), 要求是 unique 的
    """
    if faces_idx.size == 0:
        return np.array([], dtype=np.int64)

    # 1) 生成所有无向边 (每个三角形3条) 
    e01 = faces_idx[:, [0, 1]]
    e12 = faces_idx[:, [1, 2]]
    e20 = faces_idx[:, [2, 0]]
    edges = np.vstack((e01, e12, e20))  # (3M, 2)

    # 2) 无向化：每条边小索引在前
    edges = np.sort(edges, axis=1)

    # 3) 统计每条边出现次数
    # 使用 np.unique(axis=0) 是最快/最简洁的纯 NumPy 方法
    uniq_edges, counts = np.unique(edges, axis=0, return_counts=True)

    # 4) 边界边 (只出现 1 次) 
    boundary_edges = uniq_edges[counts == 1]

    # 5) 边界顶点 = 边界边的端点集合
    boundary_vertices = np.unique(boundary_edges.ravel())

    return boundary_vertices

def check_finite(data: List[np.ndarray]):
    for value in data:
        if not np.isfinite(value).all():
            return False
    return True


def safe_gather(
    source: np.ndarray, 
    index_array: np.ndarray, 
    pad_value: int | List[int], 
    fill_value: float = 0.
):
    """
    参数：
        source: shape=(N, D) 的顶点数组
        index_array: 任意形状的索引数组，其中 padding_value 表示 padding
        fill_value: 用于替换 padding_value 位置的值，可以是标量或 shape=(D,) 的数组

    返回：
        result: shape = index_array.shape + (D,) 的安全索引结果数组
    """
    if not np.issubdtype(index_array.dtype, np.integer):
        raise ValueError("index_array 必须是整数类型")

    # 顶点维度 D
    D = source.shape[1]

    # mask where index == padding_value
    if isinstance(pad_value, int):
        pad_value = [pad_value]
    elif isinstance(pad_value, list): 
        pad_value = np.array(pad_value)
    
    mask = np.isin(index_array, pad_value)

    # 复制一份索引，并把 padding_value 替换成合法索引 (比如 0) 
    safe_indices = index_array.copy()
    safe_indices[mask] = 0

    # 使用高级索引：flatten 再 reshape 成原索引形状 + D
    flat_indices = safe_indices.flatten()
    selected = source[flat_indices]  # shape = (prod(shape), D)
    result = selected.reshape(index_array.shape + (D,))  # 还原原始结构

    # 处理 fill_value
    if np.isscalar(fill_value):
        fill_value = np.full((D,), fill_value, dtype=source.dtype)
    else:
        fill_value = np.array(fill_value, dtype=source.dtype)
        if fill_value.shape != (D,):
            raise ValueError(f"fill_value 的 shape 应该是 ({D},)，但得到 {fill_value.shape}")

    # 替换掉原来是 -1 的位置
    result[mask] = fill_value

    return result



def cut_before_value(arr: np.ndarray, value, axis: int = 0):
    """
    在指定 axis 上，找到第一个切片等于 value 的位置，并返回 axis 之前的部分。

    参数：
        arr   : 任意维 ndarray
        value : 标量或可广播到 arr.shape[:axis] + arr.shape[axis+1:] 的数组
        axis  : 沿哪个维度切割 (默认 0) 

    返回：
        arr_cut  : 在 axis 上切到 first_idx 之前的数组 (不含 first_idx) 
        first_idx: 第一次匹配的下标；若未找到则为 None
    """
    arr = np.asarray(arr)
    value = np.asarray(value)

    if arr.ndim == 0:
        raise ValueError("arr 至少应为 1 维")

    # 规范化 axis
    if axis < 0:
        axis += arr.ndim
    if not (0 <= axis < arr.ndim):
        raise ValueError(f"axis 超出范围：{axis} 对于 arr.ndim={arr.ndim}")

    # 把目标轴移到最前，方便逐切片比较
    a = np.moveaxis(arr, axis, 0)  # shape: (N, ...rest)
    # 与 value 比较；允许广播
    eq = (a == value)

    # 计算每个切片是否“完全等于”
    if a.ndim == 1:
        # 1D: 每个切片是标量，eq 形状就是 (N,)
        matches = eq
    else:
        # 多维: 每个切片是 (...rest)，需在其余轴上 all
        matches = np.all(eq, axis=tuple(range(1, a.ndim)))  # -> (N,)

    if np.any(matches):
        first_idx = int(np.argmax(matches))
        # 构造切片：在 axis 维取 :first_idx
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(0, first_idx)
        return arr[tuple(sl)], first_idx
    else:
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
