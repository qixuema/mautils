import torch
from einops import repeat, rearrange
from torchtyping import TensorType

def safe_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return norm.clamp(min=eps)


def interpolate_1d(
    t: TensorType["bs", "n"],
    data: TensorType["bs", "c", "n"],
):
    """
    对给定的数据进行一维线性插值。

    参数:
    t (Tensor): 插值坐标，范围在 [0, 1] 之间，形状为 (batch_size, n)。
    data (Tensor): 原始数据，形状为 (batch_size, channels, num_points)。

    返回:
    Tensor: 插值后的数据，形状为 (batch_size, channels, n)。
    """
    # 检查输入张量的形状是否符合预期
    assert t.dim() == 2, "t 应该是一个二维张量，形状为 (batch_size, n)"
    assert data.dim() == 3, "data 应该是一个三维张量，形状为 (batch_size, num_points, channels)"

    # coords range [0, 1]
    num_reso = data.shape[-1]
    # 将插值坐标 t 映射到 [0, num_reso - 1] 范围内
    t = t * (num_reso - 1)

    # Perform linear interpolation
    left = torch.floor(t).long()
    right = torch.ceil(t).long()
    alpha = t - left

    # 防止超出索引范围
    left = torch.clamp(left, max=num_reso - 1)
    right = torch.clamp(right, max=num_reso - 1)

    c = data.shape[-2]

    left = repeat(left, 'bs n -> bs c n', c=c)
    left_values = torch.gather(data, -1, left)

    right = repeat(right, 'bs n -> bs c n', c=c)
    right_values = torch.gather(data, -1, right)

    alpha = repeat(alpha, 'bs n -> bs c n', c=c)

    interpolated = (1 - alpha) * left_values + alpha * right_values
    
    return interpolated




def calculate_polyline_lengths(points: TensorType['b', 'n', 3, float]) -> TensorType['b', float]:
    """
    计算批量折线(polylines)的长度。

    参数:
    points (torch.Tensor): 形状为 (batch_size, num_points, 3) 的张量，
                           其中 batch_size 是批量大小，
                           num_points 是每条折线的点数，
                           3 是每个点的三维坐标。

    返回:
    torch.Tensor: 形状为 (batch_size,) 的张量，表示每条折线的总长度。
    """
    points = rearrange(points, 'b d n -> b n d')

    # 检查输入的张量维度
    if points.dim() != 3 or points.size(2) != 3:
        raise ValueError("输入张量必须是形状为 (batch_size, num_points, 3) 的三维张量")

    # 计算相邻点之间的差
    diffs = points[:, 1:, :] - points[:, :-1, :]

    # 计算每个差向量的欧几里得距离
    distances = torch.norm(diffs, dim=2)

    # 计算每条折线的总长度
    polyline_lengths = distances.sum(dim=1)

    return polyline_lengths

def down_sample_edge_points(batch_edge_points, num_points=32):
    # edge_points: (bs, 256, 3)
    batch_edge_points = rearrange(batch_edge_points, 'b n c -> b c n')
    
    t = torch.linspace(0, 1, num_points).to(batch_edge_points.device)
    bs = batch_edge_points.shape[0]
    t = repeat(t, 'n -> b n', b=bs)
    
    batch_edge_points = interpolate_1d(t, batch_edge_points)
    
    batch_edge_points = rearrange(batch_edge_points, 'b c n -> b n c')
    
    return batch_edge_points