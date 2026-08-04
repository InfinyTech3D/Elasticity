"""
3D Beam Simulation - Three-Point Bending - P1 Tetrahedra
Cross-validation SOFA vs FreeFEM

Boundary conditions (all identified by exact node-coordinate matching,
since this mesh happens to have exact nodes on x=0, x=L, and x=L/2):
  - Left support line  (x=0, y=0): pin      -> ux = uy = uz = 0
  - Right support line  (x=L, y=0): roller  -> uy = 0 only
  - Midspan load line   (x=L/2, y=H): distributed line load along -y,
    split across the nodes on that line using trapezoidal (tributary
    length) weights so that the total applied force equals P.
"""
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
W = 0.2
TOL = 1e-6


def _line_load_weights(z_values):
    """Trapezoidal tributary-length weights along a 1D line (weights sum to 1)."""
    z_values = np.asarray(z_values)
    order = np.argsort(z_values)
    z_sorted = z_values[order]
    n = len(z_sorted)
    w_sorted = np.zeros(n)
    for k in range(n):
        left = z_sorted[k] - z_sorted[k - 1] if k > 0 else 0.0
        right = z_sorted[k + 1] - z_sorted[k] if k < n - 1 else 0.0
        w_sorted[k] = 0.5 * (left + right)
    w_sorted /= w_sorted.sum()
    w = np.empty(n)
    w[order] = w_sorted
    return w


def _default_mesh_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "beam3d_tet.msh")


def _default_params_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "params_beam3d_threept.json")


def create_scene_args(rootNode, mesh_file, P, young_modulus, poisson_ratio, tol=TOL):
    requiredPlugins = [
        "Elasticity",
        "Sofa.Component.Constraint.Projective",
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

    template = "Vec3d"

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

        loader = Beam.addObject('MeshGmshLoader', name="loader", filename=mesh_file)

        nodes = np.array(loader.position.value)
        N = len(nodes)
        x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]

        # ---- Left support line (x=0, y=0): pin (ux=uy=uz=0) ----
        left_idx = np.where(np.isclose(x, 0.0, atol=tol) & np.isclose(y, 0.0, atol=tol))[0]

        # ---- Right support line (x=L, y=0): roller (uy=0 only) ----
        right_idx = np.where(np.isclose(x, L, atol=tol) & np.isclose(y, 0.0, atol=tol))[0]

        # ---- Midspan load line (x=L/2, y=H): distributed line load ----
        load_idx = np.where(np.isclose(x, L / 2.0, atol=tol) & np.isclose(y, H, atol=tol))[0]

        if len(left_idx) == 0 or len(right_idx) == 0 or len(load_idx) == 0:
            raise RuntimeError(
                f"BC node search failed: left={len(left_idx)}, "
                f"right={len(right_idx)}, load={len(load_idx)}. "
                f"Check that this mesh really has exact nodes at x=0, x=L, x=L/2."
            )

        weights = _line_load_weights(z[load_idx])
        forces_y = -P * weights  # applied along -y

        dofs = Beam.addObject('MechanicalObject'
                            , name="dofs"
                            , template=template
                            , position="@loader.position"
                            , showObject=True
                            , showObjectScale=0.01)

        Beam.addObject('TetrahedronSetTopologyContainer'
                    , name="topology"
                    , src="@loader")
        Beam.addObject('TetrahedronSetTopologyModifier')

        Beam.addObject('LinearSmallStrainFEMForceField'
                    , name="FEM"
                    , template=template
                    , youngModulus=young_modulus
                    , poissonRatio=poisson_ratio
                    , topology="@topology")

        Beam.addObject('FixedProjectiveConstraint'
                    , name="leftSupport"
                    , indices=left_idx.tolist())

        Beam.addObject('PartialFixedProjectiveConstraint'
                    , name="rightSupport"
                    , fixedDirections=[0, 1, 0]
                    , indices=right_idx.tolist())

        Beam.addObject('ConstantForceField'
                    , name="MidspanLoad"
                    , indices=load_idx.tolist()
                    , forces=[[0.0, fy, 0.0] for fy in forces_y])

    return rootNode, dofs, nodes.copy()


def createScene(rootNode):
    with open(_default_params_path()) as f:
        cfg = json.load(f)
    create_scene_args(rootNode
                    , mesh_file=_default_mesh_path()
                    , P=float(cfg["P"])
                    , young_modulus=float(cfg["youngModulus"])
                    , poisson_ratio=float(cfg["poissonRatio"]))
    return rootNode


def sofaRun(mesh_file, P, young_modulus, poisson_ratio):
    root = Sofa.Core.Node("root")
    _, dofs, pos0 = create_scene_args(root
                                    , mesh_file=mesh_file
                                    , P=P
                                    , young_modulus=young_modulus
                                    , poisson_ratio=poisson_ratio)
    Sofa.Simulation.init(root)
    Sofa.Simulation.animate(root, root.dt.value)

    pos_final = np.array(dofs.position.toList())
    u = pos_final[:, :3] - pos0[:, :3]
    x0 = pos0[:, :3]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "sofa_beam3d_threept_results.txt")
    with open(out_path, 'w') as f:
        f.write(f"{'x0':>12}  {'y0':>12}  {'z0':>12}  {'ux':>12}  {'uy':>12}  {'uz':>12}\n")
        f.write("-" * 78 + "\n")
        for (xi, yi, zi), (uxi, uyi, uzi) in zip(x0, u):
            f.write(f"{xi:12.6f}  {yi:12.6f}  {zi:12.6f}  {uxi:12.6f}  {uyi:12.6f}  {uzi:12.6f}\n")

    return x0, u


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else _default_params_path()
    with open(config_file) as f:
        cfg = json.load(f)

    sofaRun(mesh_file=_default_mesh_path()
            , P=float(cfg["P"])
            , young_modulus=float(cfg["youngModulus"])
            , poisson_ratio=float(cfg["poissonRatio"]))