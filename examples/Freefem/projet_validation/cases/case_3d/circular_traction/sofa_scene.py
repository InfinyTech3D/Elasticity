import json
import os
import sys
import numpy as np
import Sofa
import Sofa.Core
import Sofa.Simulation

RESULTS_DIR = "results"


def consistent_traction_forces(nodes, loaded_faces, q):
    
    N = len(nodes)
    F = np.zeros((N, 3))
    for tri in loaded_faces:
        pts = nodes[tri, :]
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        area = 0.5 * np.linalg.norm(np.cross(v1, v2))
        for nid in tri:
            F[nid, 0] += q * area / 3.0
    return F


def _default_mesh_path(): 
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         RESULTS_DIR, "beam3d_circle_tet.msh")


def _default_params_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "params_beam3d_circle_traction.json")


def create_scene_args(rootNode, mesh_file, q, young_modulus, poisson_ratio, tol=1e-6):
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

        if not os.path.isfile(mesh_file):
            raise FileNotFoundError(
                f"Maillage introuvable : {mesh_file}\n"
                f"-> generate it first  : python beam3d_circular_tet.py params.json"
            )

        loader = Beam.addObject('MeshGmshLoader', name="loader", filename=mesh_file)

        nodes = np.array(loader.position.value)
        tets = np.array(loader.tetrahedra.value)
        tris = np.array(loader.triangles.value)
        N = len(nodes)
 
        x_min = nodes[:, 0].min()
        x_max = nodes[:, 0].max()
        fixed_idx = np.where(np.isclose(nodes[:, 0], x_min, atol=tol))[0].tolist()
        loaded_mask = np.all(np.isclose(nodes[tris, 0], x_max, atol=tol), axis=1)
        loaded_faces = tris[loaded_mask]

        assert len(fixed_idx) > 0, "error "
        assert len(loaded_faces) > 0, " (x_max) not found  "

        F_nodal = consistent_traction_forces(nodes, loaded_faces, q)
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
                    , name="LoadedTraction"
                    , indices=list(range(N))
                    , forces=forces_list)

    return rootNode, dofs, nodes.copy()


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
    u = pos_final[:, :3] - pos0[:, :3]
    x0 = pos0[:, :3]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "sofa_beam3d_circle_traction_results.txt")
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
            , q=float(cfg["q"])
            , young_modulus=float(cfg["youngModulus"])
            , poisson_ratio=float(cfg["poissonRatio"]))