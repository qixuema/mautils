import torch
from torch import Tensor
from einops import repeat
from torchtyping import TensorType


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
