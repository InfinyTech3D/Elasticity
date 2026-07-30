import json
import os
import sys
import numpy as np
import Sofa
import Sofa.Core
import Sofa.Simulation

RESULTS_DIR = "results"
MESH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh")
DEFAULT_MESH_FILENAME = "beam3d_circular_tet.msh"


def torsion_consistent_forces(nodes, end_faces, T, J, yc, zc): 
    N = len(nodes)
    F = np.zeros((N, 3))
    for tri in end_faces:
        pts = nodes[tri, :]
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        area = 0.5 * np.linalg.norm(np.cross(v1, v2))
        for nid in tri:
            y, z = nodes[nid, 1], nodes[nid, 2]
            dy, dz = y - yc, z - zc
            ty = -(T / J) * dz
            tz = (T / J) * dy
            F[nid, 1] += ty * area / 3.0
            F[nid, 2] += tz * area / 3.0
    return F


def _check_small_strain(T, radius, young_modulus, poisson_ratio, length,
                         theta_length_limit=0.1): 
    J = np.pi * radius**4 / 2.0
    G = young_modulus / (2.0 * (1.0 + poisson_ratio))
    theta = T / (G * J)         
    theta_total = theta * length  
    if abs(theta_total) > theta_length_limit:
        print(
            f"estimated total Torsion's Angle  = {theta_total:.3g} rad "
            f"(> {theta_length_limit} rad).  small-strain linear modal it's not validated",
            file=sys.stderr,
        )
    return theta, theta_total


def _verify_torsion(x0, u, radius, yc, zc, theta_total, tol_x=1e-6,
                     radius_rel_tol=0.05, angle_abs_tol=0.05):
    x = x0[:, 0]
    x_max = x.max()
    end_mask = np.isclose(x, x_max, atol=tol_x)

    y0e, z0e = x0[end_mask, 1], x0[end_mask, 2]
    uxe, uye, uze = u[end_mask, 0], u[end_mask, 1], u[end_mask, 2]

    y1e, z1e = y0e + uye, z0e + uze

    r0 = np.sqrt((y0e - yc) ** 2 + (z0e - zc) ** 2)
    r1 = np.sqrt((y1e - yc) ** 2 + (z1e - zc) ** 2)
    valid = r0 > 0.1 * radius

    r_rel_err = np.abs(r1[valid] - r0[valid]) / r0[valid]

    angle0 = np.arctan2(z0e[valid] - zc, y0e[valid] - yc)
    angle1 = np.arctan2(z1e[valid] - zc, y1e[valid] - yc)
    dangle = np.mod(angle1 - angle0 + np.pi, 2 * np.pi) - np.pi  
 

    ok_radius = r_rel_err.max() < radius_rel_tol
    ok_angle = abs(dangle.mean() - theta_total) < angle_abs_tol

    if ok_radius and ok_angle:
        print(" It's a real torsion ")
    else:
        print(
            "  The deformation it's not a torsion ===> verify the T & youngModulus ",
            file=sys.stderr,
        )

    return {
        "r0": r0, "r1": r1, "r_rel_err": r_rel_err,
        "dangle_mean": dangle.mean(), "dangle_std": dangle.std(),
        "ux_mean": uxe.mean(), "ux_max_abs": np.abs(uxe).max(),
        "ok": ok_radius and ok_angle,
    }
 
def _default_mesh_path():
    return os.path.join(MESH_DIR, DEFAULT_MESH_FILENAME)


def _default_params_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "params_beam3d_torsion.json")


def create_scene_args(rootNode, mesh_file, T, radius, young_modulus, poisson_ratio, tol=1e-6):
    if not os.path.isfile(mesh_file):
        raise FileNotFoundError()

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
        tets = np.array(loader.tetrahedra.value)
        tris = np.array(loader.triangles.value)
        N = len(nodes)

        x_min = nodes[:, 0].min()
        x_max = nodes[:, 0].max()
        length = x_max - x_min
        fixed_idx = np.where(np.isclose(nodes[:, 0], x_min, atol=tol))[0].tolist()

        end_mask = np.all(np.isclose(nodes[tris, 0], x_max, atol=tol), axis=1)
        end_faces = tris[end_mask]

        if len(fixed_idx) == 0:
            raise RuntimeError()
        if len(end_faces) == 0:
            raise RuntimeError()

        _check_small_strain(T, radius, young_modulus, poisson_ratio, length)

        yc = 0.5 * (nodes[:, 1].min() + nodes[:, 1].max())
        zc = 0.5 * (nodes[:, 2].min() + nodes[:, 2].max())
        J = np.pi * radius**4 / 2.0

        F_nodal = torsion_consistent_forces(nodes, end_faces, T, J, yc, zc)
        forces_list = F_nodal.tolist()

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
                    , name="dirichlet"
                    , indices=fixed_idx)
 
        Beam.addObject('ConstantForceField'
                    , name="TorqueTraction"
                    , indices=list(range(N))
                    , forces=forces_list
                    , showArrowSize=0.0
                    , showColor=[1.0, 0.2, 0.0, 1.0])
 

    return rootNode, dofs, nodes.copy()


def createScene(rootNode):
    with open(_default_params_path()) as f:
        cfg = json.load(f)
    create_scene_args(rootNode
                    , mesh_file=_default_mesh_path()
                    , T=float(cfg["T"])
                    , radius=float(cfg["radius"])
                    , young_modulus=float(cfg["youngModulus"])
                    , poisson_ratio=float(cfg["poissonRatio"]))
    return rootNode


def sofaRun(mesh_file, T, radius, young_modulus, poisson_ratio):
    root = Sofa.Core.Node("root")
    _, dofs, pos0 = create_scene_args(root
                                    , mesh_file=mesh_file
                                    , T=T
                                    , radius=radius
                                    , young_modulus=young_modulus
                                    , poisson_ratio=poisson_ratio)
    Sofa.Simulation.init(root)
    Sofa.Simulation.animate(root, root.dt.value)

    pos_final = np.array(dofs.position.toList())
    u = pos_final[:, :3] - pos0[:, :3]
    x0 = pos0[:, :3]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "sofa_beam3d_torsion_results.txt")
    with open(out_path, 'w') as f:
        f.write(f"{'x0':>12}  {'y0':>12}  {'z0':>12}  {'ux':>12}  {'uy':>12}  {'uz':>12}\n")
        f.write("-" * 78 + "\n")
        for (xi, yi, zi), (uxi, uyi, uzi) in zip(x0, u):
            f.write(f"{xi:12.6f}  {yi:12.6f}  {zi:12.6f}  {uxi:12.6f}  {uyi:12.6f}  {uzi:12.6f}\n")

    yc = 0.5 * (x0[:, 1].min() + x0[:, 1].max())
    zc = 0.5 * (x0[:, 2].min() + x0[:, 2].max())
    length = x0[:, 0].max() - x0[:, 0].min()
    _, theta_total = _check_small_strain(T, radius, young_modulus, poisson_ratio, length)
    _verify_torsion(x0, u, radius, yc, zc, theta_total)

    return x0, u


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else _default_params_path()
    with open(config_file) as f:
        cfg = json.load(f)

    sofaRun(mesh_file=_default_mesh_path()
            , T=float(cfg["T"])
            , radius=float(cfg["radius"])
            , young_modulus=float(cfg["youngModulus"])
            , poisson_ratio=float(cfg["poissonRatio"]))