"""
2D Beam Simulation - Uniaxial Compression - Plane Stress - P1 Triangles
Cross-validation SOFA vs FreeFEM, with a far-field analytical check
(uniform compression, valid away from the clamped edge - see PR notes).

Physical labels in beam2d_tri.msh (Gmsh 2.2 format):
    1 = Fixed  (x=0, boundary edges) -> Dirichlet (ux=uy=0)
    2 = Bottom (y=0)
    3 = Right  (x=L)                 -> Neumann traction (-q, 0)  [compression]
    4 = Top    (y=H)
    5 = Beam   (2D domain, triangles)

Plane stress is obtained via template="Vec2d" (genuine 2D DOFs, no z
component at all - unlike the plane strain counterpart
sofa_beam2d_compression.py, which uses template="Vec3d" with a locked
z-DOF). This mirrors the convention already used for the distributed-load
pair (sofa_beam2d_distributed.py / sofa_beam2d_distributed_cp.py).
"""
import json
import os
import sys
import numpy as np
import Sofa
import Sofa.Core
import Sofa.Simulation

RESULTS_DIR = "results"


def read_gmsh_2d(path):
    """Minimal Gmsh 2.2 ASCII reader.

    Returns:
        nodes      : (N,2) array of x,y coordinates, in file node-id order
        triangles  : (M,3) int array of 0-based node indices, region 5 (Beam)
        fixed_idx  : sorted list of 0-based node indices on boundary label 1 (Fixed)
        right_edges: (K,2) int array of 0-based node indices, boundary label 3 (Right)
    """
    with open(path) as f:
        lines = f.read().splitlines()

    i = 0
    nodes = None
    triangles = []
    fixed_nodes = set()
    right_edges = []

    while i < len(lines):
        line = lines[i].strip()
        if line == "$Nodes":
            n = int(lines[i + 1])
            nodes = np.zeros((n, 2))
            for k in range(n):
                parts = lines[i + 2 + k].split()
                node_id = int(parts[0])
                nodes[node_id - 1, 0] = float(parts[1])
                nodes[node_id - 1, 1] = float(parts[2])
            i += 2 + n
        elif line == "$Elements":
            m = int(lines[i + 1])
            for k in range(m):
                parts = lines[i + 2 + k].split()
                elm_type = int(parts[1])
                n_tags = int(parts[2])
                physical = int(parts[3])
                node_ids = [int(x) - 1 for x in parts[3 + n_tags:]]
                if elm_type == 2:  # 3-node triangle
                    if physical == 5:
                        triangles.append(node_ids)
                elif elm_type == 1:  # 2-node line
                    if physical == 1:  # Fixed
                        fixed_nodes.update(node_ids)
                    elif physical == 3:  # Right
                        right_edges.append(node_ids)
            i += 2 + m
        else:
            i += 1

    triangles = np.array(triangles, dtype=int)
    right_edges = np.array(right_edges, dtype=int)
    fixed_idx = sorted(fixed_nodes)
    return nodes, triangles, fixed_idx, right_edges


def consistent_nodal_forces_traction(nodes, edges, q):
    """Consistent (1/2 per node) nodal forces for a constant traction
    (-q, 0) per unit length, integrated over each boundary edge (P1)."""
    N = len(nodes)
    F = np.zeros((N, 2))
    for e in edges:
        p0, p1 = nodes[e[0]], nodes[e[1]]
        Le = np.linalg.norm(p1 - p0)
        for nid in e:
            F[nid, 0] += -q * Le / 2.0
    return F


def _default_mesh_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "beam2d_tri.msh")


def _default_params_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "params_beam2d_compression.json")


def create_scene_args(rootNode, mesh_file, q, young_modulus, poisson_ratio):
    requiredPlugins = [
        "Elasticity",
        "Sofa.Component.Constraint.Projective",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.MechanicalLoad",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Dynamic",
        "Sofa.Component.Visual",
        "Sofa.GL.Component.Rendering3D",
    ]
    rootNode.addObject('RequiredPlugin', pluginName=requiredPlugins)
    rootNode.addObject('DefaultAnimationLoop')
    rootNode.addObject('VisualStyle', displayFlags=["showBehaviorModels", "showForceFields"])

    nodes, triangles, fixed_idx, right_edges = read_gmsh_2d(mesh_file)
    N = len(nodes)

    # Plane stress -> template Vec2d (genuine 2D elasticity, no z dof)
    template = "Vec2d"
    positions = nodes.tolist()
    tris_list = triangles.tolist()

    F_nodal = consistent_nodal_forces_traction(nodes, right_edges, q)
    forces_list = F_nodal.tolist()

    with rootNode.addChild('Beam') as Beam:
        Beam.addObject('NewtonRaphsonSolver'
                    , name="newtonSolver"
                    , printLog=True
                    , maxNbIterationsNewton=30
                    , absoluteResidualStoppingThreshold=1e-12)
        Beam.addObject('SparseLDLSolver'
                    , name="linearSolver"
                    , template="CompressedRowSparseMatrixd")
        Beam.addObject('StaticSolver'
                    , name="staticSolver"
                    , newtonSolver="@newtonSolver"
                    , linearSolver="@linearSolver")

        dofs = Beam.addObject('MechanicalObject'
                            , name="dofs"
                            , template=template
                            , position=positions
                            , showObject=True
                            , showObjectScale=0.01)

        Beam.addObject('TriangleSetTopologyContainer'
                    , name="topology"
                    , position="@dofs.position"
                    , triangles=tris_list)
        Beam.addObject('TriangleSetTopologyModifier')

        Beam.addObject('LinearSmallStrainFEMForceField'
                    , name="FEM"
                    , template=template
                    , youngModulus=young_modulus
                    , poissonRatio=poisson_ratio
                    , topology="@topology")

        Beam.addObject('FixedProjectiveConstraint'
                    , name="dirichlet"
                    , indices=fixed_idx)

        Beam.addObject('ConstantForceField'
                    , name="CompressionLoad"
                    , indices=list(range(N))
                    , forces=forces_list)

    return rootNode, dofs, np.array(positions)


def createScene(rootNode):
    with open(_default_params_path()) as f:
        cfg = json.load(f)
    create_scene_args(rootNode
                    , mesh_file=_default_mesh_path()
                    , q=float(cfg["q"])
                    , young_modulus=float(cfg["youngModulus"])
                    , poisson_ratio=float(cfg["poissonRatio"]))
    return rootNode


def sofaRun(mesh_file, q, young_modulus, poisson_ratio):
    root = Sofa.Core.Node("root")
    _, dofs, pos0 = create_scene_args(root
                                    , mesh_file=mesh_file
                                    , q=q
                                    , young_modulus=young_modulus
                                    , poisson_ratio=poisson_ratio)
    Sofa.Simulation.init(root)
    Sofa.Simulation.animate(root, root.dt.value)

    pos_final = np.array(dofs.position.toList())
    u = pos_final - pos0   # (N,2): ux, uy directly, no z dof to drop
    x0 = pos0

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "sofa_beam2d_compression_cp_results.txt")
    with open(out_path, 'w') as f:
        f.write(f"{'x0':>12}  {'y0':>12}  {'ux':>12}  {'uy':>12}\n")
        f.write("-" * 54 + "\n")
        for (xi, yi), (uxi, uyi) in zip(x0, u):
            f.write(f"{xi:12.6f}  {yi:12.6f}  {uxi:12.6f}  {uyi:12.6f}\n")

    return x0, u


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else _default_params_path()
    with open(config_file) as f:
        cfg = json.load(f)

    sofaRun(mesh_file=_default_mesh_path()
            , q=float(cfg["q"])
            , young_modulus=float(cfg["youngModulus"])
            , poisson_ratio=float(cfg["poissonRatio"]))