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

def _bulk_floats(payloads, ncols, dtype):
    if not payloads:
        return np.empty((0, ncols), dtype=dtype)
    arr = np.fromstring(' '.join(payloads), sep=' ', dtype=dtype)
    if arr.size == len(payloads) * ncols:
        return arr.reshape(-1, ncols)
    return np.array([line.split()[:ncols] for line in payloads], dtype=dtype)


def _triangulate_int_lines(payloads):
    if all(line.count(' ') == 2 for line in payloads):
        arr = np.fromstring(' '.join(payloads), sep=' ', dtype=np.int32) - 1
        return arr.reshape(-1, 3)
    out = []
    for line in payloads:
        idx = [int(p) - 1 for p in line.split()]
        for i in range(1, len(idx) - 1):
            out.append((idx[0], idx[i], idx[i + 1]))
    return np.array(out, dtype=np.int32) if out else np.empty((0, 3), dtype=np.int32)


def _parse_faces(f_lines):
    empty = lambda: np.empty((0, 3), dtype=np.int32)
    if not f_lines:
        return empty(), empty(), empty()

    if not any('/' in line for line in f_lines):
        return _triangulate_int_lines(f_lines), empty(), empty()

    faces_xyz_l, faces_uv_l, face_normals_l = [], [], []
    for line in f_lines:
        v_idx, vt_idx, vn_idx = [], [], []
        for tok in line.split():
            sub = tok.split('/')
            vi = int(sub[0]) - 1 if sub[0] else -1
            ti = int(sub[1]) - 1 if len(sub) > 1 and sub[1] else -1
            ni = int(sub[2]) - 1 if len(sub) > 2 and sub[2] else -1
            v_idx.append(vi); vt_idx.append(ti); vn_idx.append(ni)
        for i in range(1, len(v_idx) - 1):
            faces_xyz_l.append((v_idx[0], v_idx[i], v_idx[i + 1]))
            faces_uv_l.append((vt_idx[0], vt_idx[i], vt_idx[i + 1]))
            face_normals_l.append((vn_idx[0], vn_idx[i], vn_idx[i + 1]))

    return (
        np.array(faces_xyz_l, dtype=np.int32) if faces_xyz_l else empty(),
        np.array(faces_uv_l, dtype=np.int32) if faces_uv_l else empty(),
        np.array(face_normals_l, dtype=np.int32) if face_normals_l else empty(),
    )


def _parse_line_segments(l_lines):
    if not l_lines:
        return np.empty((0, 2), dtype=np.int32)
    if all(line.count(' ') == 1 and '/' not in line for line in l_lines):
        arr = np.fromstring(' '.join(l_lines), sep=' ', dtype=np.int32) - 1
        return arr.reshape(-1, 2)
    out = []
    for line in l_lines:
        idx = [int(p.split('/')[0]) - 1 for p in line.split()]
        for i in range(len(idx) - 1):
            out.append((idx[i], idx[i + 1]))
    return np.array(out, dtype=np.int32) if out else np.empty((0, 2), dtype=np.int32)


def read_obj(file_path):
    """
    Read an OBJ file and extract vertices, UVs, normals, faces, and lines.

    Returns:
        dict with:
            - xyz: (N, 3) float32
            - uv: (M, 2) float32
            - v_normals: (K, 3) float32
            - faces_xyz: (F, 3) int32 vertex indices (0-based)
            - faces_uv: (F, 3) int32 uv indices (0-based, or -1 if missing)
            - face_normals: (F, 3) int32 normal indices (0-based, or -1 if missing)
            - lines: (L, 2) int32 segment indices (0-based)
    """
    with open(file_path, 'r') as fp:
        text = fp.read()

    v_lines, vt_lines, vn_lines, f_lines, l_lines = [], [], [], [], []
    for line in text.splitlines():
        if not line or line[0] == '#':
            continue
        if line.startswith('v '):
            v_lines.append(line[2:])
        elif line.startswith('vt '):
            vt_lines.append(line[3:])
        elif line.startswith('vn '):
            vn_lines.append(line[3:])
        elif line.startswith('f '):
            f_lines.append(line[2:])
        elif line.startswith('l '):
            l_lines.append(line[2:])

    faces_xyz, faces_uv, face_normals = _parse_faces(f_lines)

    return {
        "xyz": _bulk_floats(v_lines, 3, np.float32),
        "uv": _bulk_floats(vt_lines, 2, np.float32),
        "v_normals": _bulk_floats(vn_lines, 3, np.float32),
        "faces_xyz": faces_xyz,
        "faces_uv": faces_uv,
        "face_normals": face_normals,
        "lines": _parse_line_segments(l_lines),
    }


def _to_percent_fmt(brace_fmt: str) -> str:
    if brace_fmt.startswith('{:') and brace_fmt.endswith('}'):
        return '%' + brace_fmt[2:-1]
    return brace_fmt


def _format_faces(faces_xyz, faces_uv, face_normals, write_uv, write_vn) -> str:
    if write_uv and write_vn and (faces_uv >= 0).all() and (face_normals >= 0).all():
        v = (faces_xyz + 1).tolist()
        t = (faces_uv + 1).tolist()
        n = (face_normals + 1).tolist()
        return '\n'.join(
            f"f {v[i][0]}/{t[i][0]}/{n[i][0]} {v[i][1]}/{t[i][1]}/{n[i][1]} {v[i][2]}/{t[i][2]}/{n[i][2]}"
            for i in range(len(v))
        ) + '\n'
    if write_uv and not write_vn and (faces_uv >= 0).all():
        v = (faces_xyz + 1).tolist()
        t = (faces_uv + 1).tolist()
        return '\n'.join(
            f"f {v[i][0]}/{t[i][0]} {v[i][1]}/{t[i][1]} {v[i][2]}/{t[i][2]}"
            for i in range(len(v))
        ) + '\n'
    if write_vn and not write_uv and (face_normals >= 0).all():
        v = (faces_xyz + 1).tolist()
        n = (face_normals + 1).tolist()
        return '\n'.join(
            f"f {v[i][0]}//{n[i][0]} {v[i][1]}//{n[i][1]} {v[i][2]}//{n[i][2]}"
            for i in range(len(v))
        ) + '\n'
    if not write_uv and not write_vn:
        v = (faces_xyz + 1).tolist()
        return '\n'.join(f"f {row[0]} {row[1]} {row[2]}" for row in v) + '\n'
    return _format_faces_mixed(faces_xyz, faces_uv, face_normals, write_uv, write_vn)


def _format_faces_mixed(faces_xyz, faces_uv, face_normals, write_uv, write_vn) -> str:
    fx = (faces_xyz + 1).tolist()
    fu = (faces_uv + 1).tolist() if write_uv else None
    fn = (face_normals + 1).tolist() if write_vn else None
    out = []
    for i in range(len(fx)):
        toks = []
        for c in range(3):
            v_idx = fx[i][c]
            ti = fu[i][c] if write_uv and fu[i][c] >= 1 else None
            ni = fn[i][c] if write_vn and fn[i][c] >= 1 else None
            if ti is None and ni is None:
                toks.append(str(v_idx))
            elif ti is None:
                toks.append(f"{v_idx}//{ni}")
            elif ni is None:
                toks.append(f"{v_idx}/{ti}")
            else:
                toks.append(f"{v_idx}/{ti}/{ni}")
        out.append('f ' + ' '.join(toks))
    return '\n'.join(out) + '\n'


def _format_lines(lines: np.ndarray) -> str:
    if lines.ndim == 2 and lines.shape[1] == 2:
        idx = (lines + 1).tolist()
        return '\n'.join(f"l {a} {b}" for a, b in idx) + '\n'
    if lines.ndim == 2:
        idx = (lines + 1).tolist()
        return '\n'.join('l ' + ' '.join(map(str, row)) for row in idx if len(row) >= 2) + '\n'
    out = []
    for row in lines:
        row = np.asarray(row).ravel()
        if row.size < 2:
            continue
        out.append('l ' + ' '.join(str(int(x) + 1) for x in row))
    return '\n'.join(out) + '\n' if out else ''


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
            - lines: (L, >=2) int32 (0-based)
        float_fmt: brace-style format for floats (e.g., "{:.6f}").
        write_header_comment: write a brief comment header.
    """
    xyz = data.get("xyz", np.empty((0, 3), dtype=np.float32))
    uv = data.get("uv", np.empty((0, 2), dtype=np.float32))
    v_normals = data.get("v_normals", np.empty((0, 3), dtype=np.float32))
    faces_xyz = data.get("faces_xyz", np.empty((0, 3), dtype=np.int32))
    faces_uv = data.get("faces_uv", np.empty((0, 3), dtype=np.int32))
    face_normals = data.get("face_normals", np.empty((0, 3), dtype=np.int32))
    lines = data.get("lines", np.empty((0, 2), dtype=np.int32))

    if xyz.size and xyz.shape[1] != 3:
        raise ValueError("xyz must be (N, 3).")
    if uv.size and uv.shape[1] < 2:
        raise ValueError("uv must be (M, >=2).")
    if v_normals.size and v_normals.shape[1] != 3:
        raise ValueError("v_normals must be (K, 3).")
    if faces_xyz.size and faces_xyz.shape[1] != 3:
        raise ValueError("faces_xyz must be (F, 3) triangles.")
    if faces_uv.size and faces_uv.shape != faces_xyz.shape:
        raise ValueError("faces_uv must have shape (F, 3) matching faces_xyz.")
    if face_normals.size and face_normals.shape != faces_xyz.shape:
        raise ValueError("face_normals must have shape (F, 3) matching faces_xyz.")
    if lines.size and lines.ndim == 2 and lines.shape[1] < 2:
        raise ValueError("lines must be (L, >=2).")
    if faces_xyz.size and (faces_xyz < 0).any():
        raise ValueError("faces_xyz contains negative indices.")

    has_uv_faces = faces_uv.size != 0
    has_vn_faces = face_normals.size != 0
    has_vt_pool = uv.size != 0
    has_vn_pool = v_normals.size != 0

    pct = _to_percent_fmt(float_fmt)
    v_fmt = f"v {pct} {pct} {pct}"
    vt_fmt = f"vt {pct} {pct}"
    vn_fmt = f"vn {pct} {pct} {pct}"

    with open(file_path, "w", encoding="utf-8") as f:
        if write_header_comment:
            f.write("# Written by write_obj\n")
            f.write(
                f"# V={xyz.shape[0]} VT={uv.shape[0]} VN={v_normals.shape[0]} "
                f"F={faces_xyz.shape[0]} L={lines.shape[0]}\n"
            )

        if xyz.size:
            np.savetxt(f, np.ascontiguousarray(xyz[:, :3]), fmt=v_fmt)
        if has_vt_pool:
            np.savetxt(f, np.ascontiguousarray(uv[:, :2]), fmt=vt_fmt)
        if has_vn_pool:
            np.savetxt(f, np.ascontiguousarray(v_normals[:, :3]), fmt=vn_fmt)

        if faces_xyz.size:
            f.write(_format_faces(
                faces_xyz, faces_uv, face_normals,
                write_uv=has_uv_faces and has_vt_pool,
                write_vn=has_vn_faces and has_vn_pool,
            ))

        if lines.size:
            f.write(_format_lines(lines))


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