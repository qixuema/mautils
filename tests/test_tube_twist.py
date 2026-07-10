"""Twist gate for qixuema.trimesh_utils.polyline_to_prism (RMF rewrite).

Metric: for every pair of consecutive cross-section rings, project matching
vertex offsets onto the plane ⊥ the shared tangent and measure the signed
rotation angle. A twisted joint shows |angle| approaching pi; a clean sweep
stays well under half a facet (pi / n_sides).

Also checks closed-loop welding (watertight, consistent winding, volume > 0,
seam ring pair untwisted) and degenerate inputs. Compares against the old
trimesh.creation.sweep_polygon path for reference.
"""
from __future__ import annotations

import sys

import numpy as np
import trimesh

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from qixuema.trimesh_utils import polyline_to_prism  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def ring_twist_angles(verts, n_sides, n_rings, closed=False, k_offset=0):
    """Max |rotation| between corresponding vertices of consecutive rings."""
    rings = verts[: n_rings * n_sides].reshape(n_rings, n_sides, 3)
    centers = rings.mean(axis=1)
    pairs = [(i, i + 1, list(range(n_sides))) for i in range(n_rings - 1)]
    if closed:
        pairs.append((n_rings - 1, 0, [(j + k_offset) % n_sides for j in range(n_sides)]))
    worst = 0.0
    for i0, i1, jmap in pairs:
        t = centers[i1] - centers[i0]
        nt = np.linalg.norm(t)
        if nt < 1e-12:
            continue
        t /= nt
        for j in range(n_sides):
            a = rings[i0, j] - centers[i0]
            b = rings[i1, jmap[j]] - centers[i1]
            a = a - t * np.dot(a, t)
            b = b - t * np.dot(b, t)
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-12 or nb < 1e-12:
                continue
            ang = float(np.arctan2(np.dot(np.cross(a, b), t), np.dot(a, b)))
            worst = max(worst, abs(ang))
    return worst


def helix(n=80, turns=3.0, r=1.0, pitch=0.4):
    s = np.linspace(0, 2 * np.pi * turns, n)
    return np.column_stack([r * np.cos(s), r * np.sin(s), pitch * s])


def wavy_loop(n=64, r=1.0, amp=0.25, freq=5):
    s = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([r * np.cos(s), r * np.sin(s), amp * np.sin(freq * s)])
    return np.vstack([pts, pts[:1]])


def main():
    n_sides = 5
    half_facet = np.pi / n_sides

    # --- open helix: new implementation must have no twisting joints ---
    pl = helix()
    mesh = polyline_to_prism(pl, n_sides=n_sides, radius=0.05, end_caps=True)
    w = ring_twist_angles(np.asarray(mesh.vertices), n_sides, len(pl))
    check("helix: max joint twist < half facet", w < half_facet, f"max={np.degrees(w):.1f}° (limit {np.degrees(half_facet):.0f}°)")
    check("helix: watertight with caps", mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0,
          f"watertight={mesh.is_watertight} winding={mesh.is_winding_consistent} vol={mesh.volume:.2e}")

    # --- old sweep_polygon reference on the same helix (informational) ---
    try:
        from shapely.geometry import Polygon
        ang = np.arange(n_sides) * 2 * np.pi / n_sides
        poly = Polygon(np.column_stack([0.05 * np.cos(ang), 0.05 * np.sin(ang)]))
        old = trimesh.creation.sweep_polygon(poly, path=pl, cap=False)
        # sweep_polygon vertex layout differs; measure via ring grouping only if size matches
        if len(old.vertices) >= len(pl) * n_sides:
            w_old = ring_twist_angles(np.asarray(old.vertices), n_sides, len(pl))
            print(f"       (old sweep_polygon on same helix: max joint twist {np.degrees(w_old):.1f}°)")
    except Exception as e:
        print(f"       (old sweep reference skipped: {type(e).__name__})")

    # --- closed wavy loop: seamless weld ---
    pl = wavy_loop()
    mesh = polyline_to_prism(pl, n_sides=n_sides, radius=0.05)
    m = len(pl) - 1
    # recover k the same way the implementation does: brute-check smallest twist
    best = min(range(n_sides), key=lambda kk: ring_twist_angles(
        np.asarray(mesh.vertices), n_sides, m, closed=True, k_offset=kk))
    w = ring_twist_angles(np.asarray(mesh.vertices), n_sides, m, closed=True, k_offset=best)
    check("loop: max joint twist (incl. weld seam) < half facet", w < half_facet,
          f"max={np.degrees(w):.1f}°, seam offset k={best}")
    check("loop: watertight seamless weld", mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0,
          f"watertight={mesh.is_watertight} winding={mesh.is_winding_consistent} vol={mesh.volume:.2e}")

    # --- circle (planar) sanity ---
    s = np.linspace(0, 2 * np.pi, 33)
    circle = np.column_stack([np.cos(s), np.sin(s), np.zeros_like(s)])
    mesh = polyline_to_prism(circle, n_sides=n_sides, radius=0.05)
    check("circle: watertight", mesh.is_watertight and mesh.volume > 0, f"vol={mesh.volume:.2e}")

    # --- loop rings must lie in the angle-bisector plane at EVERY vertex,
    #     including the loop start (was: start ring stood ⊥ first segment) ---
    poly = np.array([[0, 0, 0], [2, 0.2, 0.1], [2.5, 1.5, -0.2], [1, 2.4, 0.3],
                     [-0.8, 1.6, 0.0], [-0.5, 0.5, -0.1], [0, 0, 0.0]])
    mesh = polyline_to_prism(poly, n_sides=n_sides, radius=0.05)
    m = len(poly) - 1
    rings = np.asarray(mesh.vertices)[: m * n_sides].reshape(m, n_sides, 3)
    worst_deg, worst_i = 0.0, -1
    for i in range(m):
        off = rings[i] - rings[i].mean(axis=0)
        plane_n = np.linalg.svd(off)[2][-1]                       # ring plane normal
        p_prev, p_cur, p_next = poly[(i - 1) % m], poly[i], poly[(i + 1) % m]
        d1 = (p_cur - p_prev) / np.linalg.norm(p_cur - p_prev)
        d2 = (p_next - p_cur) / np.linalg.norm(p_next - p_cur)
        bis = (d1 + d2) / np.linalg.norm(d1 + d2)                 # bisector tangent
        ang = np.degrees(np.arccos(np.clip(abs(np.dot(plane_n, bis)), -1, 1)))
        if ang > worst_deg:
            worst_deg, worst_i = ang, i
    check("loop: every ring in bisector plane (incl. start vertex)", worst_deg < 0.5,
          f"worst={worst_deg:.3f}° at vertex {worst_i}")

    # --- degenerates ---
    two = polyline_to_prism(np.array([[0, 0, 0], [1, 0, 0.0]]), n_sides=n_sides, radius=0.05)
    check("2-point line ok", two.is_watertight and two.volume > 0, "")
    dup = polyline_to_prism(np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 1, 0.0]]),
                            n_sides=n_sides, radius=0.05)
    check("duplicate points filtered", dup.is_watertight, "")
    sharp = polyline_to_prism(np.array([[0, 0, 0], [1, 0, 0], [0, 1e-4, 0.0]]), n_sides=n_sides, radius=0.01)
    check("180° turnback no crash", len(sharp.vertices) > 0, "")

    ok = all(RESULTS)
    print("\n" + ("ALL PASS" if ok else "FAILURES PRESENT"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
