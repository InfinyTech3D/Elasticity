import json
import os
import sys
import numpy as np
import Sofa
import Sofa.Core
import Sofa.Simulation

RESULTS_DIR = "results"

L = 1.0
H = 0.2
TOL = 1e-3


def _default_mesh_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "beam2d_tri.msh")


def _default_params_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "params_beam3pt.json")


def create_scene_args(rootNode, mesh_file, P, young_modulus, poisson_ratio, plane_type="strain"):
    assert plane_type in ("strain", "stress"), "plane_type must be 'strain' or 'stress'"

    requiredPlugins = [
        "Elasticity",
        "Sofa.Component.Constraint.Projective",
        "Sofa.Component.Engine.Select",
        "Sofa.Component.IO.Mesh",
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

    template = "Vec3d" if plane_type == "strain" else "Vec2d"

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

        loader = Beam.addObject('MeshGmshLoader'
                    , name="loader"
                    , filename=mesh_file)

        if template == "Vec2d":
            # MeshGmshLoader always returns 3D positions (x, y, 0) - drop the
            # redundant z column for the plane stress (Vec2d) formulation.
            positions_3d = np.array(loader.position.toList())
            position_arg = positions_3d[:, :2].tolist()
        else:
            position_arg = "@loader.position"

        dofs = Beam.addObject('MechanicalObject'
                            , name="dofs"
                            , template=template
                            , position=position_arg
                            , showObject=True
                            , showObjectScale=0.01)

        topology = Beam.addObject('TriangleSetTopologyContainer'
                    , name="topology"
                    , src="@loader")
        Beam.addObject('TriangleSetTopologyModifier')

        Beam.addObject('LinearSmallStrainFEMForceField'
                    , name="FEM"
                    , template=template
                    , youngModulus=young_modulus
                    , poissonRatio=poisson_ratio
                    , topology="@topology")

        if template == "Vec3d":
            # Plane strain: lock every node's z displacement to 0.
            Beam.addObject('BoxROI'
                        , name="allNodesROI"
                        , box=[-TOL, -TOL, -TOL, L + TOL, H + TOL, TOL]
                        , drawBoxes=False)
            Beam.addObject('PartialFixedProjectiveConstraint'
                        , name="zLock"
                        , fixedDirections=[0, 0, 1]
                        , indices="@allNodesROI.indices")

        # --- Left support: pin (ux=0, uy=0) ---
        Beam.addObject('BoxROI'
                    , name="leftSupportROI"
                    , box=[-TOL, -TOL, -TOL, TOL, TOL, TOL]
                    , drawBoxes=True)
        Beam.addObject('FixedProjectiveConstraint'
                    , name="leftSupport"
                    , indices="@leftSupportROI.indices")

        # --- Right support: roller (uy=0 only, ux free) ---
        right_fixed_dirs = [0, 1, 0] if template == "Vec3d" else [0, 1]
        Beam.addObject('BoxROI'
                    , name="rightSupportROI"
                    , box=[L - TOL, -TOL, -TOL, L + TOL, TOL, TOL]
                    , drawBoxes=True)
        Beam.addObject('PartialFixedProjectiveConstraint'
                    , name="rightSupport"
                    , fixedDirections=right_fixed_dirs
                    , indices="@rightSupportROI.indices")

        # --- Load point: concentrated force (0, -P[, 0]) at midspan/top ---
        force_vec = [0.0, -P, 0.0] if template == "Vec3d" else [0.0, -P]
        Beam.addObject('BoxROI'
                    , name="loadROI"
                    , box=[L / 2 - TOL, H - TOL, -TOL, L / 2 + TOL, H + TOL, TOL]
                    , drawBoxes=True)
        Beam.addObject('ConstantForceField'
                    , name="PointLoad"
                    , indices="@loadROI.indices"
                    , forces=[force_vec])

    return rootNode, dofs, topology


def createScene(rootNode):
    with open(_default_params_path()) as f:
        cfg = json.load(f)
    create_scene_args(rootNode
                    , mesh_file=_default_mesh_path()
                    , P=float(cfg["P"])
                    , young_modulus=float(cfg["youngModulus"])
                    , poisson_ratio=float(cfg["poissonRatio"])
                    , plane_type=cfg.get("planeType", "strain"))
    return rootNode


def sofaRun(mesh_file, P, young_modulus, poisson_ratio, plane_type="strain"):
    root = Sofa.Core.Node("root")
    _, dofs, topology = create_scene_args(root
                                    , mesh_file=mesh_file
                                    , P=P
                                    , young_modulus=young_modulus
                                    , poisson_ratio=poisson_ratio
                                    , plane_type=plane_type)

    Sofa.Simulation.init(root)
    pos0 = np.array(dofs.position.toList())
    Sofa.Simulation.animate(root, root.dt.value)
    pos_final = np.array(dofs.position.toList())

    if plane_type == "strain":
        u3 = pos_final - pos0
        u = u3[:, :2]
        x0 = pos0[:, :2]
    else:
        u = pos_final - pos0   # (N,2): ux, uy directly, no z dof to drop
        x0 = pos0

    triangles = np.array(topology.triangles.toList())

    mid_idx = np.argmin(np.abs(x0[:, 0] - L / 2) + np.abs(x0[:, 1] - H))
    print(f"[sanity check] midspan deflection uy = {u[mid_idx, 1]:.6e} "
          f"(plane_type={plane_type}, expected: negative)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"sofa_beam3pt_{plane_type}_results.txt")
    with open(out_path, 'w') as f:
        f.write(f"{'x0':>12}  {'y0':>12}  {'ux':>12}  {'uy':>12}\n")
        f.write("-" * 54 + "\n")
        for (xi, yi), (uxi, uyi) in zip(x0, u):
            f.write(f"{xi:12.6f}  {yi:12.6f}  {uxi:12.6f}  {uyi:12.6f}\n")

    return x0, u, triangles


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else _default_params_path()
    with open(config_file) as f:
        cfg = json.load(f)

    sofaRun(mesh_file=_default_mesh_path()
            , P=float(cfg["P"])
            , young_modulus=float(cfg["youngModulus"])
            , poisson_ratio=float(cfg["poissonRatio"])
            , plane_type=cfg.get("planeType", "strain"))