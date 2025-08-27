import numpy as np
from typing import Iterable, Tuple, Optional, Union

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
        raise ValueError(f"Degenerate bounding box: bbmin={bbmin}, bbmax={bbmax}, denom={denom}")
    
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
    padding: str = "post",       # "post" 左对齐（在尾部补），"pre" 右对齐（在头部补）
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
        padding: "post" 或 "pre"（补在尾部/头部）。
        truncating: 当 Ti > M 时从 "post"（尾部截掉）或 "pre"（头部截掉）。
        return_lengths: 是否返回每条序列真实放入的长度（min(Ti, M)）。

    Returns:
        padded: 形状 (N, M, *feat_shape) 的数组。
        lengths (可选): 形状 (N,) 的实际长度。
    """
    seqs = [np.asarray(s) for s in sequences]
    N = len(seqs)
    if N == 0:
        out = np.empty((0, 0), dtype=(dtype or np.asarray(pad_value).dtype))
        return (out, np.empty((0,), dtype=int)) if return_lengths else out

    # 统一特征维（length 维是轴 0），标量当作长度为 1 的序列
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
