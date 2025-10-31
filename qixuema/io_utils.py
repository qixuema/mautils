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

def read_obj(file_path):
    """
    Read an OBJ file and extract vertices, UVs, normals, faces, and lines.

    Returns:
        dict with:
            - xyz: (N, 3)
            - uv: (M, 2)
            - v_normals: (K, 3)
            - faces_xyz: (F, 3) vertex indices (0-based)
            - faces_uv: (F, 3) uv indices (0-based, or -1 if missing)
            - lines: (L, 2) line indices (0-based)
    """    
    xyz = []
    uv = []
    faces_xyz = []
    faces_uv = []
    face_normals = []
    lines = []
    vertice_normals = []

    with open(file_path, 'r') as file:
        for line in file:
            if not line.strip() or line.startswith("#"):
                continue
                        
            parts = line.strip().split()

            if not parts:
                continue

            if parts[0] == 'v':
                # Parse vertex coordinates
                vertex = tuple(map(float, parts[1:4]))
                xyz.append(vertex)

            elif parts[0] == "vt":
                # UV can be 2D or 3D (u,v[,w]) → usually only u,v
                uv.append(tuple(map(float, parts[1:3])))                

            elif parts[0] == "vn":
                vertice_normals.append(tuple(map(float, parts[1:4])))

            elif parts[0] == 'f':
                v_idx, vt_idx, vn_idx = [], [], []
                for v in parts[1:]:
                    tokens = v.split("/")
                    # handle v, v/vt, v//vn, v/vt/vn cases
                    vi = int(tokens[0]) - 1 if tokens[0] else -1
                    ti = int(tokens[1]) - 1 if len(tokens) > 1 and tokens[1] else -1
                    ni = int(tokens[2]) - 1 if len(tokens) > 2 and tokens[2] else -1

                    v_idx.append(vi)
                    vt_idx.append(ti)
                    vn_idx.append(ni)

                # triangulate faces with >3 vertices
                for i in range(1, len(v_idx) - 1):
                    faces_xyz.append((v_idx[0], v_idx[i], v_idx[i + 1]))
                    faces_uv.append((vt_idx[0], vt_idx[i], vt_idx[i + 1]))
                    face_normals.append((vn_idx[0], vn_idx[i], vn_idx[i + 1]))
                
            elif parts[0] == 'l':
                # Parse face indices (assuming triangular faces)
                # line = tuple(map(int, parts[1:3]))
                # lines.append(line)
                if len(parts) >= 3:
                    indices = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                    for i in range(len(indices) - 1):
                        lines.append((indices[i], indices[i + 1]))                

    return {
        "xyz": np.array(xyz, dtype=np.float32),
        "uv": np.array(uv, dtype=np.float32),
        "v_normals": np.array(vertice_normals, dtype=np.float32),
        "faces_xyz": np.array(faces_xyz, dtype=np.int32) if faces_xyz else np.empty((0, 3), dtype=np.int32),
        "faces_uv": np.array(faces_uv, dtype=np.int32) if faces_uv else np.empty((0, 3), dtype=np.int32),
        "face_normals": np.array(face_normals, dtype=np.int32) if face_normals else np.empty((0, 3), dtype=np.int32),
        "lines": np.array(lines, dtype=np.int32) if lines else np.empty((0, 2), dtype=np.int32),
    }


def write_obj(
    file_path: str,
    data: Dict[str, np.ndarray],
    float_fmt: str = "{:.6f}",
    write_header_comment: bool = True,
) -> None:
    """
    Write an OBJ file from a dict compatible with `read_obj`.

    Args:
        file_path: output .obj path
        data: dict with keys (same as read_obj output):
            - xyz: (N, 3) float32
            - uv: (M, 2) float32
            - v_normals: (K, 3) float32
            - faces_xyz: (F, 3) int32 (0-based)
            - faces_uv: (F, 3) int32 (0-based, or -1 if missing)
            - face_normals: (F, 3) int32 (0-based, or -1 if missing)
            - lines: (L, 2) int32 (0-based)
        float_fmt: formatting for floats (e.g., "{:.6f}")
        write_header_comment: write a brief comment header
    """
    xyz: np.ndarray = data.get("xyz", np.empty((0, 3), dtype=np.float32))
    uv: np.ndarray = data.get("uv", np.empty((0, 2), dtype=np.float32))
    v_normals: np.ndarray = data.get("v_normals", np.empty((0, 3), dtype=np.float32))
    faces_xyz: np.ndarray = data.get("faces_xyz", np.empty((0, 3), dtype=np.int32))
    faces_uv: np.ndarray = data.get("faces_uv", np.empty((0, 3), dtype=np.int32))
    face_normals: np.ndarray = data.get("face_normals", np.empty((0, 3), dtype=np.int32))
    lines: np.ndarray = data.get("lines", np.empty((0, 2), dtype=np.int32))

    # Basic validations / shape normalization
    if xyz.size and xyz.shape[1] != 3:
        raise ValueError("xyz must be (N, 3).")
    if uv.size and uv.shape[1] < 2:
        raise ValueError("uv must be (M, 2) or (M, >=2).")
    if v_normals.size and v_normals.shape[1] != 3:
        raise ValueError("v_normals must be (K, 3).")
    if faces_xyz.size and faces_xyz.shape[1] != 3:
        raise ValueError("faces_xyz must be (F, 3) triangles.")
    if faces_uv.size and faces_uv.shape != faces_xyz.shape:
        # allow empty faces_uv; otherwise must align with faces_xyz
        if faces_uv.size != 0:
            raise ValueError("faces_uv must have shape (F, 3) matching faces_xyz.")
    if face_normals.size and face_normals.shape != faces_xyz.shape:
        if face_normals.size != 0:
            raise ValueError("face_normals must have shape (F, 3) matching faces_xyz.")
    if lines.size and lines.shape[1] < 2:
        raise ValueError("lines must be (L, >=2).")

    has_uv_faces = faces_uv.size != 0
    has_vn_faces = face_normals.size != 0
    has_vt_pool = uv.size != 0
    has_vn_pool = v_normals.size != 0

    with open(file_path, "w", encoding="utf-8") as f:
        if write_header_comment:
            f.write("# Written by write_obj\n")
            f.write(f"# V={xyz.shape[0]} VT={uv.shape[0]} VN={v_normals.shape[0]} "
                    f"F={faces_xyz.shape[0]} L={lines.shape[0]}\n")

        # v
        for v in xyz:
            f.write("v " + " ".join(float_fmt.format(x) for x in v[:3]) + "\n")

        # vt
        if has_vt_pool:
            for t in uv:
                # OBJ vt can be 2D or 3D; we write only first two (u,v)
                f.write("vt " + " ".join(float_fmt.format(x) for x in t[:2]) + "\n")

        # vn
        if has_vn_pool:
            for n in v_normals:
                f.write("vn " + " ".join(float_fmt.format(x) for x in n[:3]) + "\n")

        # f (triangles)
        F = faces_xyz.shape[0]
        for i in range(F):
            vi = faces_xyz[i]  # (3,)
            # Defensive checks
            if np.any(vi < 0):
                raise ValueError("faces_xyz contains negative indices.")
            tokens = []
            for c in range(3):
                v_idx = int(vi[c]) + 1  # OBJ is 1-based

                # decide vt / vn per corner
                ti = None
                ni = None

                if has_uv_faces and has_vt_pool:
                    t_raw = int(faces_uv[i, c])
                    if t_raw >= 0:
                        ti = t_raw + 1

                if has_vn_faces and has_vn_pool:
                    n_raw = int(face_normals[i, c])
                    if n_raw >= 0:
                        ni = n_raw + 1

                # assemble per-corner token
                if ti is None and ni is None:
                    token = f"{v_idx}"
                elif ti is None and ni is not None:
                    token = f"{v_idx}//{ni}"
                elif ti is not None and ni is None:
                    token = f"{v_idx}/{ti}"
                else:
                    token = f"{v_idx}/{ti}/{ni}"

                tokens.append(token)

            f.write("f " + " ".join(tokens) + "\n")

        # l (polyline segments). Each row may contain 2+ indices; we support both.
        if lines.size:
            # If lines is strictly (L,2), write pairs; if (L,>2), write polyline
            if lines.ndim == 2:
                for row in lines:
                    row = np.asarray(row).ravel()
                    if row.size < 2:
                        continue
                    # OBJ 'l' expects 1-based vertex indices
                    idx_str = " ".join(str(int(x) + 1) for x in row)
                    f.write("l " + idx_str + "\n")
            else:
                # Allow (variable-length) list of index arrays
                for row in lines:
                    row = np.asarray(row).ravel()
                    if row.size < 2:
                        continue
                    idx_str = " ".join(str(int(x) + 1) for x in row)
                    f.write("l " + idx_str + "\n")


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

def save_obj_with_uv(output_path, xyz, uv, faces_xyz, faces_uv, precision=None):
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
    uv = _round_if_needed(uv, precision)
        
    with open(output_path, 'w') as file:
        _write_data(file, xyz, uv, faces_xyz, faces_uv)

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

    with open(output_path, 'w', encoding="utf-8") as fp:
        fp.write("# OBJ file generated by save_obj_parts_with_uv\n")

        cur_v_off, cur_vt_off = v_offset, vt_offset

        for idx, part in enumerate(parts):
            name = part.get("name", f"part_{idx}")

            xyz = np.asarray(part["xyz"], dtype=float)
            uv = np.asarray(part["uv"], dtype=float)
            faces_v = np.asarray(part["faces_xyz"], dtype=int)
            faces_vt = np.asarray(part["faces_uv"], dtype=int)

            # Optional rounding for clean files and smaller diffs
            xyz = _round_if_needed(xyz, precision)
            uv = _round_if_needed(uv, precision)

            # Header lines for readability
            fp.write(f"\n# ---- {name} ----\n")
            fp.write(f"o {name}\n")  # you can switch to 'g' if you prefer groups

            # Write vertices and uvs
            _write_data(fp, xyz, uv, faces_v, faces_vt, cur_v_off, cur_vt_off)

            # Bump offsets for the next part
            cur_v_off += xyz.shape[0]
            cur_vt_off += uv.shape[0]

if __name__=="__main__":
    # Example usage
    vertices = [(1, 2, 3), (5, 4, 6), (9, 8, 7)]
    faces_xyz = [(0, 1, 2)]
    uvs = [(0.5, 0.5), (0.75, 0.5), (0.75, 0.75)]
    save_obj_with_uv('example.obj', vertices, uvs, faces_xyz, faces_uv=faces_xyz)