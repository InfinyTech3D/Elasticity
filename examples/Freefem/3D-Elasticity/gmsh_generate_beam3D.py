import json
import os
import sys
import gmsh

RESULTS_DIR = "results"

def generate_beam3D(length, height, width, nx, ny, nz, filename):
    """
    Generate a 3D beam mesh (tetrahedral elements)
    """
    gmsh.initialize()
    gmsh.model.add("beam3d")

    # Corner points (bottom rectangle)
    gmsh.model.geo.addPoint(0,      0,      0, tag=1)
    gmsh.model.geo.addPoint(length, 0,      0, tag=2)
    gmsh.model.geo.addPoint(length, height, 0, tag=3)
    gmsh.model.geo.addPoint(0,      height, 0, tag=4)

    # Boundary lines
    bottom = gmsh.model.geo.addLine(1, 2, tag=2)  # label 2, y=0
    right  = gmsh.model.geo.addLine(2, 3, tag=3)  # label 3, x=length
    top    = gmsh.model.geo.addLine(3, 4, tag=4)  # label 4, y=height
    left   = gmsh.model.geo.addLine(4, 1, tag=1)  # label 1, x=0 (clamped)

    loop = gmsh.model.geo.addCurveLoop([bottom, right, top, left], tag=1)
    surf = gmsh.model.geo.addPlaneSurface([loop], tag=1)

    gmsh.model.geo.synchronize()

    # Structured transfinite layout on base surface
    gmsh.model.mesh.setTransfiniteCurve(bottom, nx)
    gmsh.model.mesh.setTransfiniteCurve(top,    nx)
    gmsh.model.mesh.setTransfiniteCurve(left,   ny)
    gmsh.model.mesh.setTransfiniteCurve(right,  ny)
    gmsh.model.mesh.setTransfiniteSurface(surf)

    # Recombine triangles → quads on the base face before extrusion
    gmsh.model.mesh.setRecombine(2, surf)

    # Extrude surface to create volume
    extruded = gmsh.model.geo.extrude(
        [(2, surf)], 0, 0, width,
        numElements=[nz],
        recombine=True,
    )

    gmsh.model.geo.synchronize()

    # The volume created by extrusion
    volume_tag = extruded[1][1]

    # Ensure the volume respects nx/ny/nz — required for the subdivision to
    # produce the correct number of tets.
    gmsh.model.mesh.setTransfiniteVolume(volume_tag)

    # Physical groups — boundary tags match .edp labels 1–4
    gmsh.model.addPhysicalGroup(1, [left],   tag=1, name="Fixed")
    gmsh.model.addPhysicalGroup(1, [bottom], tag=2, name="Bottom")
    gmsh.model.addPhysicalGroup(1, [right],  tag=3, name="Right")
    gmsh.model.addPhysicalGroup(1, [top],    tag=4, name="Top")
    gmsh.model.addPhysicalGroup(3, [volume_tag], tag=5, name="Beam")
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 2)

    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(1)

    _, node_coords, _ = gmsh.model.mesh.getNodes()
    x_positions = node_coords[0::3]
    y_positions = node_coords[1::3]
    z_positions = node_coords[2::3]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    msh_path = os.path.join(RESULTS_DIR, filename)

    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)  # gmshload requires MSH v2
    gmsh.write(msh_path)

    gmsh.finalize()

    return msh_path, x_positions, y_positions, z_positions


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "params.json"
    with open(config_file) as f:
        cfg = json.load(f)

    msh_path, x_positions, y_positions, z_positions = generate_beam3D(
        length=float(cfg["length"]),
        height=float(cfg["height"]),
        width=float(cfg["width"]),
        nx=int(cfg["nx"]),
        ny=int(cfg["ny"]),
        nz=int(cfg["nz"]),
        filename=cfg.get("meshfile"),
    )
    print("Wrote:", msh_path)
