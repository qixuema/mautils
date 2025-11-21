import os
import numpy as np
import sys
import re
import string
from pathlib import Path
from datetime import datetime
import json
import time
import functools
from typing import Union, Sequence, Iterable
import random
from collections import defaultdict


def safe_name(name: str) -> str:
    """Make a filesystem-friendly name."""
    if not name:
        return "unnamed"
    return re.sub(r'[^a-zA-Z0-9_.-]+', '_', name)

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



def find_files_by_ext(folder_path, exts):
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
    if isinstance(exts, str):
        exts = [exts]
    else:
        exts = list(exts)        

    norm_exts = []
    for e in exts:
        if not e:
            continue
        e = e if e.startswith('.') else f'.{e}'
        norm_exts.append(e.lower())

    folder_path = Path(folder_path)

    if not folder_path.is_dir():
        raise NotADirectoryError(f"{folder_path!r} is not a directory")

    # Traverse all files recursively
    for file_path in folder_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in norm_exts:
            file_path_list_with_extension.append(file_path.as_posix())
    
    return file_path_list_with_extension

def get_parent_directory(file_path):
    # 获取当前目录
    current_directory = os.path.dirname(file_path)
    # 获取上一级目录
    parent_directory = os.path.dirname(current_directory)
    return parent_directory

def get_dir_path(file_path):
    # 返回文件所在的目录路径
    return os.path.dirname(file_path)

def get_filename(file_path, with_ext=False):
    # 获取文件名和扩展名
    base_name = os.path.basename(file_path)
    if with_ext:
        return base_name
    else:
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
        file_list = find_files_by_ext(dataset_dir_path, extension)
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


def iter_files_with_ext(
    root: Union[str, os.PathLike],
    exts: Union[str, Sequence[str]] = ".npz",
    max_num: int = None
) -> Iterable[str]:
    """
    高效递归列出指定扩展名文件（生成器，按需产出，支持早停）
    """
    if isinstance(exts, (str, bytes)):
        exts = (exts,)
    # 统一成 .ext 小写
    exts = tuple(e if e.startswith('.') else '.'+e for e in exts)
    exts = tuple(e.lower() for e in exts)

    stack = [os.fspath(root)]
    emitted = 0

    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        # 比 Path.suffix 更省开销
                        name = entry.name
                        if name.lower().endswith(exts):
                            yield entry.path
                            emitted += 1
                            if max_num is not None and emitted >= max_num:
                                return
        except (PermissionError, FileNotFoundError):
            # 跳过无权限或瞬时消失的目录
            continue


class Timer(object):

    def __init__(self):
        self.reset()

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1

    def tictoc(self, diff):
        self.diff = diff
        self.total_time += diff
        self.calls += 1

    def total(self):
        """ return the total amount of time """
        return self.total_time

    def avg(self):
        """ return the average amount of time """
        return self.total_time / float(self.calls)

    def reset(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.

class Timers(object):

    def __init__(self):
        self.timers = defaultdict(Timer)

    def tic(self, key):
        self.timers[key].tic()

    def toc(self, key):
        self.timers[key].toc()

    def tictoc(self, key, diff):
        self.timers[key].tictoc( diff)

    def print(self, key=None):
        if key is None:
            # print all time
            for k, v in self.timers.items():
                print("{:}: \t  average {:.4f},  total {:.4f} ,\t calls {:}".format(k.ljust(30),  v.avg(), v.total_time, v.calls))
        else:
            print("Average time for {:}: {:}".format(key, self.timers[key].avg()))

    def get_print_string(self):
        strings = []
        for k, v in self.timers.items():
            strings.append("{:}: \t  average {:.4f},  total {:.4f} ,\t calls {:}".format(k.ljust(30),  v.avg(), v.total_time, v.calls))        
        string = "\n".join(strings)
        return string

    def get_avg(self, key):
        return self.timers[key].avg()
        

def make_safe_filename(
    filename: str, replace_spaces: bool = True, 
    allow_chars: tuple = ('-', '_', '.')
) -> str:
    """
    将文件名转换为安全的格式
    
    Args:
        filename: 原始文件名（可以包含路径）
        replace_spaces: 是否将空格替换为下划线（默认True）
        allow_chars: 允许保留的特殊字符元组（默认允许 - _.）
    
    Returns:
        安全的文件名
        
    Examples:
        >>> make_safe_filename("my file name.txt")
        'my_file_name.txt'
        >>> make_safe_filename("test-file 123.obj")  
        'test-file_123.obj'
        >>> make_safe_filename("file with[brackets].ply")
        'file_with_brackets_.ply'
        >>> make_safe_filename("my file.txt", replace_spaces=False)
        'myfile.txt'
    """
    # 分离文件名和扩展名
    if '.' in filename:
        name_part, ext_part = filename.rsplit('.', 1)
        has_ext = True
    else:
        name_part, ext_part = filename, ''
        has_ext = False
    
    # 处理空格替换
    if replace_spaces:
        name_part = name_part.replace(' ', '_')
    
    # 清理不安全字符
    safe_chars = []
    for char in name_part:
        if char.isalnum() or char in allow_chars:
            safe_chars.append(char)
    
    safe_name_part = ''.join(safe_chars)
    
    # 重新组合文件名和扩展名
    if has_ext:
        return f"{safe_name_part}.{ext_part}"
    else:
        return safe_name_part

def is_modified_after(file_path: str, dt: datetime) -> bool:
    """
    判断文件是否在指定时间之后被修改。
    
    参数:
        file_path (str): 文件路径
        dt (datetime): 参考时间
    
    返回:
        bool: 如果文件在指定时间之后修改，则返回 True，否则 False
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    file_time = datetime.fromtimestamp(path.stat().st_mtime, tz=dt.tzinfo)
    return file_time > dt


if __name__ == '__main__':
    
    from pathlib import Path
    import shutil
    
    input_dir = '/root/studio/datasets/texverse/skeleton_data/src/warrior_helmet/data/'
    output_dir = '/root/studio/datasets/texverse/skeleton_data/objs/warrior_helmet/data/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    obj_paths = list(Path(input_dir).glob('*.obj'))
    
    for obj_path in obj_paths:
        safe_name = make_safe_filename(obj_path.name)
        safe_path = Path(output_dir) / safe_name
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(obj_path, safe_path)
