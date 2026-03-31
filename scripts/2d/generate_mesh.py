"""
2D plate with a centered circular hole — gmsh Python API
=========================================================
Geometry
--------
  Plate : rectangle  [-Lx/2, Lx/2] × [-Ly/2, Ly/2]
  Hole  : circle of radius R centered at the origin

Physical groups (for boundary condition assignment)
---------------------------------------------------
  "left"   : left edge   (x = -Lx/2)
  "right"  : right edge  (x =  Lx/2)
  "bottom" : bottom edge (y = -Ly/2)
  "top"    : top edge    (y =  Ly/2)
  "hole"   : inner circular boundary
  "plate"  : 2-D surface (the plate minus the hole)

Output
------
  plate_with_hole.msh  (Gmsh MSH4 format)
"""

import gmsh
import meshio
import sys

# ── Parameters ────────────────────────────────────────────────────────────────
Lx = 2.0  # plate width  (x-direction)
Ly = 2.0  # plate height (y-direction)
R = 0.3  # hole radius  (must be < min(Lx, Ly) / 2)

lc_far = 0.15  # mesh size far from the hole (plate corners)
lc_hole = 0.03  # mesh size on the hole boundary (refined)

output_file = "plate_with_hole.mesh"
# ──────────────────────────────────────────────────────────────────────────────


def generate_mesh():
    gmsh.initialize(sys.argv)
    gmsh.model.add("plate_with_hole")
    gmsh.option.setNumber("General.Verbosity", 2)

    # ── 1. Points ──────────────────────────────────────────────────────────────
    hx, hy = Lx / 2, Ly / 2

    # Plate corners  (coarse mesh size)
    p1 = gmsh.model.geo.addPoint(-hx, -hy, 0, lc_far)
    p2 = gmsh.model.geo.addPoint(hx, -hy, 0, lc_far)
    p3 = gmsh.model.geo.addPoint(hx, hy, 0, lc_far)
    p4 = gmsh.model.geo.addPoint(-hx, hy, 0, lc_far)

    # Circle points: center + 4 cardinal points  (fine mesh size)
    pc = gmsh.model.geo.addPoint(0, 0, 0, lc_hole)
    pr = gmsh.model.geo.addPoint(R, 0, 0, lc_hole)
    pt = gmsh.model.geo.addPoint(0, R, 0, lc_hole)
    pl = gmsh.model.geo.addPoint(-R, 0, 0, lc_hole)
    pb = gmsh.model.geo.addPoint(0, -R, 0, lc_hole)

    # ── 2. Lines (plate boundary) ──────────────────────────────────────────────
    l_bottom = gmsh.model.geo.addLine(p1, p2)
    l_right = gmsh.model.geo.addLine(p2, p3)
    l_top = gmsh.model.geo.addLine(p3, p4)
    l_left = gmsh.model.geo.addLine(p4, p1)

    # ── 3. Arcs (hole — four quarter-circles, CCW as seen from outside → CW
    #    in the surface loop so the hole is a void) ────────────────────────────
    a1 = gmsh.model.geo.addCircleArc(pr, pc, pt)  #  0° →  90°
    a2 = gmsh.model.geo.addCircleArc(pt, pc, pl)  # 90° → 180°
    a3 = gmsh.model.geo.addCircleArc(pl, pc, pb)  # 180° → 270°
    a4 = gmsh.model.geo.addCircleArc(pb, pc, pr)  # 270° → 360°

    # ── 4. Curve loops ─────────────────────────────────────────────────────────
    outer_loop = gmsh.model.geo.addCurveLoop([l_bottom, l_right, l_top, l_left])
    # Hole loop traversed CW (negative tags) so it cuts a void in the surface
    hole_loop = gmsh.model.geo.addCurveLoop([-a1, -a4, -a3, -a2])

    # ── 5. Surface (outer loop minus hole loop) ────────────────────────────────
    surface = gmsh.model.geo.addPlaneSurface([outer_loop, hole_loop])

    gmsh.model.geo.synchronize()

    # ── 6. Physical groups ─────────────────────────────────────────────────────
    gmsh.model.addPhysicalGroup(1, [l_left], name="left")
    gmsh.model.addPhysicalGroup(1, [l_right], name="right")
    gmsh.model.addPhysicalGroup(1, [l_bottom], name="bottom")
    gmsh.model.addPhysicalGroup(1, [l_top], name="top")
    gmsh.model.addPhysicalGroup(1, [a1, a2, a3, a4], name="hole")
    gmsh.model.addPhysicalGroup(2, [surface], name="plate")

    # ── 7. Mesh refinement near hole (distance field) ─────────────────────────
    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", [a1, a2, a3, a4])
    gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 100)

    f_thresh = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", lc_hole)
    gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", lc_far)
    gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", R * 0.1)
    gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", R * 2.0)

    gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh)

    # Prevent the built-in size from overriding the background field
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    # ── 8. Generate & save ─────────────────────────────────────────────────────
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Netgen")  # improve triangle quality

    gmsh.write(output_file)
    print(f"[gmsh] Mesh written to '{output_file}'")

    # Open interactive GUI if not running in batch mode
    if "-nopopup" not in sys.argv:
        gmsh.fltk.run()

    gmsh.finalize()


if __name__ == "__main__":
    generate_mesh()
    mesh = meshio.read("plate_with_hole.msh")
    meshio.write("plate_with_hole.xdmf", mesh)
