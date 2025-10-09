import numpy as np
import yaml
from typing import List, Tuple, Optional, Dict

def read_vertices_obj(filename):
    vertices = []
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            if parts[0] == 'v':  # Vertex description
                # Convert x, y, z to float and append to the vertices list
                vertices.append([float(coord) for coord in parts[1:4]])
    return np.array(vertices)

def read_obj_file(file_path):
    vertices = []
    faces = []
    lines = []
    vertice_normals = []

    with open(file_path, 'r') as file:
        for line in file:
            components = line.strip().split()

            if not components:
                continue

            if components[0] == 'v':
                # Parse vertex coordinates
                vertex = tuple(map(float, components[1:4]))
                vertices.append(vertex)
            elif components[0] == 'f':
                # Parse face indices (assuming triangular faces)
                face = tuple(map(int, components[1:4]))
                # face = [list(map(int, p.split('/'))) for p in components[1:]]
                faces.append(face)
                
            elif components[0] == 'l':
                # Parse face indices (assuming triangular faces)
                line = tuple(map(int, components[1:3]))
                lines.append(line)
            
            elif components[0] == 'vn':
                vertice_normal = tuple(map(float, components[1:4]))
                vertice_normals.append(vertice_normal)

    return {
        'vertices': np.array(vertices), 
        'faces': np.array(faces) - 1, 
        'lines': np.array(lines) - 1,
        'v_normals': np.array(vertice_normals)}

def write_obj_file(file_path, vertices, faces=None, vtx_colors=None, is_line=False, is_point=False):
    """
    Save a simple OBJ file with vertices and faces.

    Parameters:
    file_path (str): The path to save the OBJ file.
    vertices (list of tuples): A list of vertices, each vertex is a tuple (x, y, z).
    faces (list of tuples): A list of faces, each face is a tuple of vertex indices (1-based).
    """
    
    # 先转换为numpy数组
    vertices = np.array(vertices) if isinstance(vertices, list) else vertices
    faces = np.array(faces) if isinstance(faces, list) else faces
    
    if faces is not None:
        if faces.shape[1] == 2:
            is_line = True

    with open(file_path, 'w') as file:
        # Write vertices
        if vtx_colors is None:
            for vertex in vertices:
                    file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        else:
            for vertex, vtx_color in zip(vertices, vtx_colors):
                file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]} {vtx_color[0]} {vtx_color[1]} {vtx_color[2]}\n")

        if is_point:
            return
        
        # Write faces
        if faces is None:
            return
        
        for face in faces:
            face_str = ' '.join([str(index + 1) for index in face])
            if is_line:
                file.write(f"l {face_str}\n")
            else:
                file.write(f"f {face_str}\n")

# Load configuration from YAML
def load_yaml(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
            

def _round_if_needed(arr: np.ndarray, precision: Optional[int]) -> np.ndarray:
    if precision is None:
        return arr
    m = 10 ** precision
    return np.round(np.asarray(arr) * m) / m


def _write_data(fp, xyz, uvs, faces_v, faces_vt, cur_v_off=0, cur_vt_off=0):
    # Write vertices and uvs
    for v in xyz:
        fp.write(f"v {v[0]} {v[1]} {v[2]}\n")
    for t in uvs:
        fp.write(f"vt {t[0]} {t[1]}\n")

    # Write faces with proper offsets; OBJ indices are 1-based
    # faces are assumed triangles; extend easily for ngons if needed
    for fv, fvt in zip(faces_v, faces_vt):
        fp.write(
            "f "
            + " ".join(
                f"{int(fv[j]) + 1 + cur_v_off}/{int(fvt[j]) + 1 + cur_vt_off}"
                for j in range(3)
            )
            + "\n"
        )

def save_obj_with_uv(output_path, xyz, uvs, faces_xyz, faces_uv, precision=None):
    """
    保存带 UV 坐标的 OBJ 文件，可控制浮点数精度。
    
    Args:
        output_path: 输出文件路径
        xyz: 顶点坐标数组 (N, 3)
        uvs: UV 坐标数组 (N, 2)  
        faces_xyz: 顶点面索引数组 (M, 3)
        faces_uv: UV 面索引数组 (M, 3)
        precision: 浮点数精度（小数位数），默认为 None，不进行截断
    """
    # Optional rounding for clean files and smaller diffs
    xyz = _round_if_needed(xyz, precision)
    uvs = _round_if_needed(uvs, precision)
        
    with open(output_path, 'w') as file:
        _write_data(file, xyz, uvs, faces_xyz, faces_uv)

def save_obj_parts_with_uv(
    output_path: str,
    parts: List[Dict[str, np.ndarray]],
    precision: Optional[int] = None,
):
    """
    Save multiple parts into a single OBJ with UVs.
    Each part is a dict with keys:
        - 'xyz'        : (Nv, 3) float
        - 'uvs'        : (Nvt, 2) float
        - 'faces_xyz'  : (F, 3) int   (indices into xyz, 0-based)
        - 'faces_uv'   : (F, 3) int   (indices into uvs, 0-based)
        - 'name'       : Optional[str]  object/group name for readability

    Args
    ----
    output_path : path to the OBJ file to write.
    parts       : list of part dicts (see above).
    precision   : decimals to round xyz/uvs when writing; None = no rounding.
    """
    v_offset, vt_offset = (0, 0)

    with open(output_path, w, encoding="utf-8") as fp:
        fp.write("# OBJ file generated by save_obj_parts_with_uv\n")

        cur_v_off, cur_vt_off = v_offset, vt_offset

        for idx, part in enumerate(parts):
            name = part.get("name", f"part_{idx}")

            xyz = np.asarray(part["xyz"], dtype=float)
            uvs = np.asarray(part["uvs"], dtype=float)
            faces_v = np.asarray(part["faces_xyz"], dtype=int)
            faces_vt = np.asarray(part["faces_uv"], dtype=int)

            # Optional rounding for clean files and smaller diffs
            xyz = _round_if_needed(xyz, precision)
            uvs = _round_if_needed(uvs, precision)

            # Header lines for readability
            fp.write(f"\n# ---- {name} ----\n")
            fp.write(f"o {name}\n")  # you can switch to 'g' if you prefer groups

            # Write vertices and uvs
            _write_data(fp, xyz, uvs, faces_v, faces_vt, cur_v_off, cur_vt_off)

            # Bump offsets for the next part
            cur_v_off += xyz.shape[0]
            cur_vt_off += uvs.shape[0]

if __name__=="__main__":
    # Example usage
    vertices = [(1, 2, 3), (5, 4, 6), (9, 8, 7)]
    faces = [(0, 1, 2)]
    uvs = [(0.5, 0.5), (0.75, 0.5), (0.75, 0.75)]
    save_obj_with_uv('example.obj', vertices, uvs, faces, faces_uv=faces)