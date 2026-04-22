# mautils

A personal utility library for 3D geometry processing — focused on wireframe/curve manipulation, mesh operations, and visualization. Published on PyPI as [`qixuema`](https://pypi.org/project/qixuema/).

## Installation

```bash
pip install qixuema -i https://pypi.org/simple
```

Requires Python ≥ 3.11. Core dependencies: `numpy`, `einops`, `einx`, `pyyaml`. Some submodules additionally require `open3d`, `trimesh`, `scipy`, `matplotlib`, or `torch` — install them as needed for the modules you use.

## Modules

| Module | Purpose |
| --- | --- |
| `np_utils` | Foundation numpy utilities: `safe_norm`, `normalize`, rotation matrices, interpolation, NaN/Inf checks |
| `geo_utils` | Polyline/curve operations: transform, normalize, inverse transform, sampling, subdivision |
| `utils` | High-level lineset operations: BFS traversal, deduplication, sorting, merging, camera projection |
| `helpers` | File I/O helpers (`find_files_by_ext`, `get_filename_wo_ext`), rotation matrix generators, timing |
| `line_utils` | Line-specific sorting and validation |
| `io_utils` | OBJ and image file I/O |
| `img_utils` | Image saving with auto-normalization |
| `o3d_utils` | Open3D wrappers (OBB computation, visualization) |
| `trimesh_utils` | Trimesh operations (e.g., segments → prism meshes) |
| `plot_utils` | Matplotlib frequency distribution plots |
| `config` | YAML config loading with inheritance |
| `torch_utils`, `scipy_utils` | Framework-specific helpers |

Import from submodules directly — nothing is re-exported at the package level:

```python
from qixuema.np_utils import normalize
from qixuema.geo_utils import sample_curve
```

## Quick Examples

**Resample a curve**

```python
import numpy as np
from qixuema.geo_utils import sample_curve

pts = np.random.rand(20, 3)
resampled = sample_curve(pts, n_samples=100, arc_length=True)
```

**Find files and read an OBJ**

```python
from qixuema.helpers import find_files_by_ext
from qixuema.io_utils import read_obj

for path in find_files_by_ext("./data", exts=[".obj"]):
    mesh = read_obj(path)  # dict with xyz, uv, faces, normals, lines
```

**Load a YAML config with inheritance**

```python
from qixuema.config import load_config

cfg = load_config("configs/experiment.yaml")
```

**Convert line segments to prism meshes**

```python
from qixuema.trimesh_utils import segments_to_prisms

mesh = segments_to_prisms(segments, radius=0.01, color_scheme="rainbow")
```

## Conventions

- `*_polyline` — operates on a single polyline `(N, 3)`
- `*_polylines` — batched `(B, N, 3)`
- `*_lineset` — operates on dict `{'vertices', 'lines'}` or `o3d.geometry.LineSet`
- Geometry functions return `None` on degenerate input (e.g., collinear points) — check before using
- Use `check_nan_inf()` from `np_utils` to validate arrays

## Development & Release

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management and building.

**Install locally for development**
```bash
uv pip install -e ".[dev]"
```

**1. Clean up old build files**
```bash
rm -rf build dist *.egg-info
```

**2. Build the package**
```bash
uv build
```

**3. Publish to PyPI**
```bash
uv publish
```

*Note: `uv publish` handles both checking and uploading. It will prompt for your PyPI token if not set via environment variable.*
