import os
import numpy as np
import sys
import re
from pathlib import Path
import datetime
import json
import time
import functools

def first(it):
    return it[0]

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def exists(x):
    return x is not None

def get_current_time():
    # 获取当前时间
    current_time = datetime.datetime.now()
    # 格式化输出时间
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    
    return formatted_time

def cycle(dl):
    while True:
        for data in dl:
            yield data

def divisible_by(num, den):
    return (num % den) == 0

def is_odd(n):
    return not divisible_by(n, 2)

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def is_debug():
    return True if sys.gettrace() else False



def get_file_list_with_extension(folder_path, ext):
    """
    Search for all files with the specified extension(s) in the given folder and its subfolders.

    Args:
    folder_path (str): Path to the folder where the search will be performed.
    ext (str or list of str): File extension(s) to search for, starting with a dot (e.g., '.ply').

    Returns:
    list: A list of file paths (in POSIX format) matching the specified extension(s).
    """
    file_path_list_with_extension = []

    # Ensure 'ext' is a list
    if isinstance(ext, str):
        ext = [ext]

    ext_set = {e.lower() for e in ext}

    folder_path = Path(folder_path)

    # Traverse all files recursively
    for file_path in folder_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in ext_set:
            file_path_list_with_extension.append(file_path.as_posix())
    
    return file_path_list_with_extension

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


def get_or_create_file_list_json(dataset_dir_path, json_path, extension='.npz'):
    """
    Get or create a JSON file containing a list of files with a given extension under a directory.
    If the JSON file exists, load the file list from it; otherwise, generate the file list and save to JSON.

    Args:
        dataset_dir_path (str): Path to the dataset directory.
        json_path (str): Path to the JSON file to save/load the file list.
        extension (str): File extension to search for (default: '.npz').

    Returns:
        list: List of file paths with the specified extension.
    """

    if not os.path.exists(json_path):
        file_list = get_file_list_with_extension(dataset_dir_path, extension)
        with open(json_path, 'w') as f:
            json.dump(file_list, f)
    else:
        with open(json_path, 'r') as f:
            file_list = json.load(f)
    
    file_list.sort()
    
    return file_list


def timeit(func):
    """装饰器：打印函数运行时间（单位秒）"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

def generate_random_string(length, batch_size=1):
    chars = string.ascii_letters + string.digits
    pool = random.choices(chars, k=length * batch_size)

    return [''.join(pool[i*length:(i+1)*length]) for i in range(batch_size)]
