# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`qixuema` is a personal utility library for 3D geometry processing, focusing on wireframe/curve manipulation, mesh operations, and visualization. Published to PyPI as `qixuema`.

## Build & Release Commands

This project uses `uv` for dependency management and building.

**Local development install:**
```bash
uv pip install -e ".[dev]"
```

**Build & publish to PyPI:**
```bash
# Clean old builds
rm -rf build dist *.egg-info

# Build
uv build

# Publish (handles check + upload)
uv publish
```

**Install from PyPI:**
```bash
pip install --upgrade qixuema -i https://pypi.org/simple
```

## Architecture

### Module Dependency Graph

The codebase is organized as a layered utility library:

```
np_utils (foundation)
    ↓
geo_utils, helpers, line_utils
    ↓
utils (high-level operations)
    ↓
vis_samples (visualization scripts)
```

**Core modules:**

- **`np_utils.py`**: Foundation layer - numpy utilities (safe_norm, normalize, rotation matrices, interpolation, array operations). No internal dependencies.

- **`geo_utils.py`**: Geometry operations on polylines/curves (transform, normalize, inverse transform, sampling, subdivision). Depends on `np_utils` and `o3d_utils`.

- **`utils.py`**: High-level lineset operations (BFS traversal, deduplication, sorting, merging, camera projection). Depends on `geo_utils`, `helpers`, `np_utils`, `line_utils`.

- **`helpers.py`**: File I/O utilities (find_files_by_ext, get_filename_wo_ext), rotation matrix generators, timing utilities. Imports rotation matrices from `np_utils`.

- **`line_utils.py`**: Line-specific utilities (sorting, validation). Minimal dependencies.

**Specialized modules:**

- **`io_utils.py`**: OBJ/image file I/O
- **`img_utils.py`**: Image saving with auto-normalization
- **`o3d_utils.py`**: Open3D wrappers (OBB computation, visualization)
- **`trimesh_utils.py`**: Trimesh operations
- **`plot_utils.py`**: Matplotlib frequency distribution plots
- **`vis_samples.py`**: Interactive visualization script for ABC dataset samples
- **`config.py`**: YAML config loading with inheritance
- **`torch_utils.py`**, **`scipy_utils.py`**: Framework-specific utilities

### Key Design Patterns

**Coordinate normalization pipeline:**
1. `transform_polyline()` / `transform_polyline_to_start_and_end()` - normalize curves to canonical space
2. `inverse_transform_polyline()` - denormalize back to original space
3. Used throughout for curve processing (see `geo_utils.py`)

**Lineset representation:**
- Dict format: `{'vertices': np.ndarray (N,3), 'lines': np.ndarray (M,2)}`
- Open3D format: `o3d.geometry.LineSet` with `.points` and `.lines`
- Conversion: `edge_points_to_lineset()` in `geo_utils.py`

**BFS-based sorting:**
- `utils.py` uses networkx BFS to traverse line graphs
- `bfs_()` → `bfs_lines()` / `bfs_vertices()` → `sort_lineset()` / `sort_vertices_and_lines()`

## Important Conventions

**Function naming:**
- `*_polyline` - operates on single polyline (N, 3)
- `*_polylines` - batch operation (B, N, 3)
- `*_edges_points` - operates on edge point arrays
- `*_lineset` - operates on dict/Open3D lineset format

**Error handling:**
- Geometry functions return `None` on degenerate input (e.g., collinear points)
- Callers must check for `None` before using results
- Use `check_nan_inf()` from `np_utils` to validate arrays

**Rotation matrices:**
- Always import from `np_utils.py`: `rotation_matrix_x/y/z(angle_degrees)`
- `helpers.py` provides batch versions: `get_rotaion_matrix_3d()` (note: typo in name is intentional, kept for API compatibility)

**Tolerance-based operations:**
- `remove_duplicate_vertices_and_lines()` uses `tolerance=0.0001` for vertex deduplication
- `tolerant_lexsort()` in `np_utils` for noise-resistant sorting

## Common Pitfalls

1. **Mutation vs. immutability**: Most functions return new arrays, but some (like old `scaling_and_translation` in helpers) mutate inputs. Check docstrings.

2. **Return value inconsistency**: Some functions return different types based on parameters (e.g., `sort_vertices_and_lines` returns tuple or single value depending on `return_indices`). Always check return signature.

3. **Open3D type conversions**: Open3D requires `float64` for vertices and `int32` for lines. Use `_build_lineset_and_pointcloud()` pattern from `vis_samples.py`.

4. **Collinear handling**: `transform_polyline()` and related functions have `handleCollinear` parameter. When `False`, returns `None` for degenerate cases.

## Testing

No formal test suite currently exists. When adding features:
- Manually test with `vis_samples.py` for visualization
- Use small numpy arrays for unit-level verification
- Check edge cases: empty arrays, single-point polylines, collinear points
