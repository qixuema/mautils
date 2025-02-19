import os
import numpy as np
import sys
import re
from pathlib import Path

def check_nan_inf(data):
    contains_nan_inf = np.any(np.isnan(data)) or np.any(np.isinf(data))
    if contains_nan_inf:
        return True
    return False

def is_debug():
    return True if sys.gettrace() else False

def is_close(a, b, atol=1e-5):
    return np.isclose(a, b, atol=atol)

def get_file_list_with_extension(folder_path, ext):
    """
    在给定的文件夹及其子文件夹中查找所有指定扩展名的文件。

    参数:
    folder_path (str): 要搜索的文件夹路径。
    extensions (str): 要搜索的文件扩展名，以点开头（例如 '.ply'）。

    返回:
    list: 找到的所有指定扩展名文件的路径列表。
    """
    files_with_extension = []
    
    if isinstance(ext, str):
        ext = [ext]
    
    # 使用 Path 对象遍历文件夹
    folder_path = Path(folder_path)
    
    # 遍历文件夹中的所有文件
    for file_path in folder_path.rglob('*'):  # rglob('*') 遍历所有子文件夹
        if file_path.is_file() and any(file_path.suffix.lower() == e.lower() for e in ext):
            # 使用 .as_posix() 转换为正斜杠格式
            files_with_extension.append(file_path.as_posix())
    
    return files_with_extension

def get_parent_directory(file_path):
    # 获取当前目录
    current_directory = os.path.dirname(file_path)
    # 获取上一级目录
    parent_directory = os.path.dirname(current_directory)
    return parent_directory

def get_directory_path(file_path):
    # 返回文件所在的目录路径
    return os.path.dirname(file_path)

def get_filename_wo_ext(file_path):
    # 获取文件名和扩展名
    base_name = os.path.basename(file_path)
    # 分割文件名和扩展名，并返回不带扩展名的文件名
    return os.path.splitext(base_name)[0]

def get_file_list(dir_path, ext=None):
    file_path_list = [os.path.join(dir_path, i) for i in os.listdir(dir_path)]
    file_path_list.sort()
    return file_path_list

def normalize(vector):
    return vector / np.linalg.norm(vector)

def scaling_and_translation(points):
    # scale first
    points /= 128
    
    # translate second
    z_coord_center = np.mean(points[:,2])
    # points -= coord_center

    points[:, 2] -= z_coord_center
    points[:, :2] -= 1
    
    return points

def scaling_and_translation_z(points):
    # scale first
    points /= 128
    
    # translate second
    z_coord_center = np.mean(points[:,2])

    points[:, 2] -= z_coord_center
    
    return points

def translation_xy(points):

    points[:, :2] -= 128
    
    return points

def translation_xyz(points, z):

    points[:, :2] -= 128
    points[:, 2] -= z
    
    return points


# 定义一个函数来创建Z轴的旋转矩阵
def rotation_matrix_z(angle):
    radians = np.radians(angle)
    cos_theta, sin_theta = np.cos(radians), np.sin(radians)
    return np.array([[cos_theta, -sin_theta, 0], 
                     [sin_theta, cos_theta, 0], 
                     [0, 0, 1]])

def get_rotaion_matrix_3d():
    rot_matrix_all = np.zeros((4, 3, 3))
    
    angles = [0, 90, 180, 270]
    for i in range(4):
        # 创建旋转矩阵
        angle = angles[i]
        rot_matrix = rotation_matrix_z(angle)
        rot_matrix_all[i] = rot_matrix
    
    return rot_matrix_all

def rotation_matrix_2d(angle):
    """生成2D旋转矩阵"""
    rad = np.radians(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    return np.array([[cos_a, -sin_a], [sin_a, cos_a]])

def get_rotation_matrix_2d():
    rot_matrix_all = np.zeros((4, 2, 2))
    
    angles = [0, 90, 180, 270]
    for i, angle in enumerate(angles):
        # 创建2D旋转矩阵
        rot_matrix_all[i] = rotation_matrix_2d(angle)
    
    return rot_matrix_all


def remove_third_underscore_section(original_string):
    """
    移除字符串中第三个下划线之后的部分。

    参数:
    original_string (str): 原始字符串。

    返回:
    str: 修改后的字符串。
    """
    # 找到所有下划线的位置
    underscore_positions = [pos for pos, char in enumerate(original_string) if char == '_']

    # 确保有足够的下划线来找到第三个和第四个
    if len(underscore_positions) >= 3:
        start = underscore_positions[2] + 1  # 第三个下划线后的第一个字符
        end = underscore_positions[3] if len(underscore_positions) > 3 else None

        # 剔除指定位置的字符串
        return original_string[:start] + original_string[end:]
    else:
        return original_string  # 不足够的下划线，保留原始字符串

# # 使用示例
# original_string = "seq_obj_0_1.0_0_generate_lines_sub"
# modified_string = remove_third_underscore_section(original_string)
# print(modified_string)

def transform_string(original_string):
    """
    从特定字符串中移除第三个下划线之后的部分，直到下一个下划线。

    参数:
    original_string (str): 原始字符串。

    返回:
    str: 修改后的字符串。
    """
    # 找到所有下划线的位置
    underscore_positions = [pos for pos, char in enumerate(original_string) if char == '_']

    # 确保有足够的下划线来找到第三个和第四个
    if len(underscore_positions) >= 4:
        start = underscore_positions[0] + 1  # 第三个下划线后的第一个字符
        end = underscore_positions[1] + 1  # 第四个下划线的位置

        # 剔除指定位置的字符串
        return original_string[:start] + original_string[end:]
    else:
        return original_string  # 不足够的下划线，保留原始字符串

# # 使用示例
# original_string = "seq_obj_0_1.0_0_generate_lines_sub"
# modified_string = transform_string(original_string)
# print(modified_string)

def extract_last_number(s):
    """
    从给定字符串中提取最后一个下划线后的数字。

    参数:
        s (str): 输入的字符串，如 'room_1_points_1.obj'

    返回:
        str: 从字符串中提取的数字，如果没有找到则返回一个错误消息
    """
    # 使用正则表达式来找到下划线后的数字
    matches = re.findall(r'_(\d+)\.', s)
    
    # 检查是否找到匹配项
    if matches:
        # 返回最后一个匹配项
        return matches[-1]
    else:
        # 如果没有找到数字，返回一个错误消息
        print(f"Error: No number found in string '{s}'")
        return None
    
def check_npy_file(file_path):
    try:
        # 尝试加载.npy文件
        data = np.load(file_path)
    except Exception as e:
        # 如果打开文件时发生异常，打印文件路径
        return False
    
    return True


def get_all_directories(root_path):
    directories = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        for dirname in dirnames:
            # 获取每个目录的完整路径
            directories.append(os.path.join(dirpath, dirname))
    return directories


def filter_none_results(results):
    """过滤掉结果中的 None 值"""
    return [result for result in results if result is not None]




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