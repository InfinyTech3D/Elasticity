import json
import os
import sys
import gmsh
 
MESH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh")
DEFAULT_MESH_FILENAME = "beam3d_circular_tet.msh"


def generate_beam3D_circular_tet(length, radius, mesh_size, filename=None):
     
    if filename is None:
        filename = DEFAULT_MESH_FILENAME

    gmsh.initialize()
    gmsh.model.add("beam3d_circular_tet")

    
    disk_fixed = gmsh.model.occ.addDisk(0, 0, 0, radius, radius,
                                         zAxis=[1, 0, 0], xAxis=[0, 1, 0])
    gmsh.model.occ.synchronize()
 
    out = gmsh.model.occ.extrude([(2, disk_fixed)], length, 0, 0)
    gmsh.model.occ.synchronize()

    vol_tag = [e[1] for e in out if e[0] == 3][0]
    surf_tags = [e[1] for e in out if e[0] == 2]
 
    disk_loaded = None
    lateral = None
    for s in surf_tags:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(2, s)
        if abs(xmax - xmin) < 1e-6:  
            disk_loaded = s
        else:
            lateral = s
    assert disk_loaded is not None and lateral is not None, \
              "Verify the faces "
    gmsh.model.addPhysicalGroup(2, [disk_fixed],  tag=1, name="Fixed")
    gmsh.model.addPhysicalGroup(2, [disk_loaded], tag=2, name="Loaded")
    gmsh.model.addPhysicalGroup(2, [lateral],     tag=3, name="Lateral")
    gmsh.model.addPhysicalGroup(3, [vol_tag],     tag=4, name="Beam")
 
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)

    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(1)  
    _, node_coords, _ = gmsh.model.mesh.getNodes()

    os.makedirs(MESH_DIR, exist_ok=True)
    msh_path = os.path.join(MESH_DIR, filename)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2) 
    gmsh.write(msh_path)
    gmsh.finalize()

    return msh_path, len(node_coords) // 3


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "params.json"
    with open(config_file) as f:
        all_cfg = json.load(f)
    cfg = all_cfg["beam3d_circle_tet"]

    msh_path, n_nodes = generate_beam3D_circular_tet(
        length=float(cfg["length"]),
        radius=float(cfg["radius"]),
        mesh_size=float(cfg["mesh_size"]),
        filename=cfg.get("meshfile", DEFAULT_MESH_FILENAME),
    )
    print("Wrote:", msh_path)
    print("Number of nodes:", n_nodes)