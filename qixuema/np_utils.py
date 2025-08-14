import numpy as np


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
    scale = 2.0 * scale / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale
    return vertices, center, scale


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

