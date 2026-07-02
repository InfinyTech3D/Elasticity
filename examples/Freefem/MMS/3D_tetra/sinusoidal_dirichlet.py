import numpy as np
import Sofa
import Sofa.Core
import Sofa.Simulation

from solid import (get_nodes_3d, element_tet, load_params,
                   plot_solution_profile, plot_solution_slices, RESULTS_DIR)
from solid_solution import SolidSolution3D
from scene import NodalForceAssembler
from fem import assemble_nodal_forces
from output import write_solution_table
from sinusoidal import mms   # réutilise l'instance SinusNeumann déjà définie là-bas


# ---------------------------------------------------------------------------
# Boundary helpers
# ---------------------------------------------------------------------------

def _boundary_node_indices(nx, ny, nz):
    """Tous les indices de nœuds sur une des 6 faces de la grille structurée."""
    def idx(i, j, k):
        return i + j * nx + k * nx * ny

    idxs = set()
    for j in range(ny):
        for k in range(nz):
            idxs.add(idx(0, j, k)); idxs.add(idx(nx - 1, j, k))
    for i in range(nx):
        for k in range(nz):
            idxs.add(idx(i, 0, k)); idxs.add(idx(i, ny - 1, k))
    for i in range(nx):
        for j in range(ny):
            idxs.add(idx(i, j, 0)); idxs.add(idx(i, j, nz - 1))
    return sorted(idxs)


def _body_force_only(element, mms, L, E, nu, nx, ny, nz):
    """Forces nodales : uniquement le terme source volumique, pas de traction."""
    def compute(nodes, topology):
        conn = element.read_connectivity(topology)
        xyz  = nodes[:, :3]
        return assemble_nodal_forces(
            lambda x, y, z: mms.source(x, y, z, E, nu, L),
            xyz, conn, element._source_rule(mms))
    return compute


# ---------------------------------------------------------------------------
# Controller: pushes the Dirichlet boundary positions at first animate step
# ---------------------------------------------------------------------------

class DirichletApplier(Sofa.Core.Controller):
    """Ecrit u_ex sur les DOFs de bord au premier pas d'animation.

    La scène charge donc plate (position == rest_position partout au GUI
    init), et la déformation de bord n'apparaît qu'au clic sur Play/Step,
    juste avant que le NewtonRaphsonSolver ne résolve l'intérieur.
    """

    def __init__(self, dofs, boundary_idx, boundary_pos, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dofs         = dofs
        self.boundary_idx = boundary_idx
        self.boundary_pos = boundary_pos
        self._applied     = False

    def onAnimateBeginEvent(self, event):
        if self._applied:
            return
        with self.dofs.position.writeable() as pos:
            for i, p in zip(self.boundary_idx, self.boundary_pos):
                pos[i] = p
        self._applied = True


# ---------------------------------------------------------------------------
# Scene builder
# ---------------------------------------------------------------------------

def build_dirichlet_scene(rootNode, mms, element, force_field, linear_solver,
                          L=1.0, E=1e6, nu=0.3, nx=6, ny=6, nz=6,
                          with_visual=False):
    rootNode.addObject("RequiredPlugin", pluginName=[
        "Elasticity",
        "Sofa.Component.Constraint.Projective",
        "Sofa.Component.Engine.Select",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.LinearSolver.Iterative",
        "Sofa.Component.MechanicalLoad",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Grid",
        "Sofa.Component.Topology.Container.Dynamic",
        "Sofa.Component.Topology.Mapping",
        "Sofa.Component.Visual",
    ])
    rootNode.addObject("DefaultAnimationLoop")
    if with_visual:
        rootNode.addObject("VisualStyle",
                           displayFlags="showBehaviorModels showForceFields")

    nodes_3d = get_nodes_3d(L, nx, ny, nz)

    Grid = rootNode.addChild("Grid")
    Grid.addObject("RegularGridTopology", name="grid",
                   nx=nx, ny=ny, nz=nz,
                   min=[0.0, 0.0, 0.0], max=[L, L, L])

    Solid = rootNode.addChild("Solid")
    Solid.addObject("StaticSolver", name="staticSolver", printLog=False)
    Solid.addObject("NewtonRaphsonSolver", name="newtonSolver",
                    maxNbIterationsNewton=1,
                    absoluteResidualStoppingThreshold=1e-10,
                    printLog=False)
    Solid.addObject(linear_solver["type"], name="linearSolver",
                    **linear_solver["parameters"])

    # Position/valeur cible du Dirichlet non-homogène sur le bord.
    boundary_idx = _boundary_node_indices(nx, ny, nz)
    init_pos = nodes_3d.copy()
    for i in boundary_idx:
        x, y, z = nodes_3d[i]
        ux, uy, uz = mms.u_ex(x, y, z, L)
        init_pos[i] = [x + ux, y + uy, z + uz]
    boundary_pos = [init_pos[i].tolist() for i in boundary_idx]

    # MechanicalObject démarre à rest partout (rendu "plat" au chargement).
    dofs = Solid.addObject("MechanicalObject", name="dofs", template="Vec3d",
                           position=nodes_3d.tolist(),
                           rest_position=nodes_3d.tolist(),
                           showObject=with_visual, showObjectScale=0.005 * L)

    topology = element.add_topology(Solid)

    Solid.addObject(force_field, name="FEM", template="Vec3d",
                    youngModulus=E, poissonRatio=nu, topology="@topology")

    Solid.addObject("FixedProjectiveConstraint", name="fix_boundary",
                    indices=boundary_idx)

    n_nodes = len(nodes_3d)
    ff = Solid.addObject("ConstantForceField", name="MMS_forces",
                         template="Vec3d",
                         indices=list(range(n_nodes)),
                         forces=[[0.0, 0.0, 0.0]] * n_nodes)

    Solid.addObject(NodalForceAssembler(
        dofs=dofs, topology=topology, force_field=ff,
        compute_forces=_body_force_only(element, mms, L, E, nu, nx, ny, nz),
        name="nodalForceAssembler"))

    Solid.addObject(DirichletApplier(
        dofs=dofs, boundary_idx=boundary_idx, boundary_pos=boundary_pos,
        name="dirichletApplier"))

    return dofs, topology


# ---------------------------------------------------------------------------
# Headless solve (used by run_convergence_dirichlet.py)
# ---------------------------------------------------------------------------

def solve_dirichlet(elem, mms, L, E, nu, nx, ny, nz, force_field, linear_solver):
    """Comme solve_solid, mais avec la baseline = rest_position explicitement
    (pas pos0), car ici pos initiale != rest pour les nœuds de bord une fois
    l'animation lancée."""
    root = Sofa.Core.Node("root")
    dofs, topology = build_dirichlet_scene(
        root, mms, elem, L=L, E=E, nu=nu,
        nx=nx, ny=ny, nz=nz, with_visual=False,
        force_field=force_field, linear_solver=linear_solver
    )
    Sofa.Simulation.init(root)
    nodes_3d = dofs.rest_position.array().copy()
    conn     = elem.read_connectivity(topology)
    Sofa.Simulation.animate(root, root.dt.value)
    pos1     = dofs.position.array().copy()
    Sofa.Simulation.unload(root)
    ux = pos1[:, 0] - nodes_3d[:, 0]
    uy = pos1[:, 1] - nodes_3d[:, 1]
    uz = pos1[:, 2] - nodes_3d[:, 2]
    return SolidSolution3D(nodes=nodes_3d, conn=conn, ux=ux, uy=uy, uz=uz)


# ---------------------------------------------------------------------------
# runSofa entry point (scène visuelle, params.json -> reference block)
# ---------------------------------------------------------------------------

def createScene(rootNode):
    cfg = load_params()
    ref = cfg["reference"]
    build_dirichlet_scene(rootNode, mms, element_tet,
                          L=cfg["length"], E=cfg["youngModulus"],
                          nu=ref["nu"],
                          nx=ref["nx"], ny=ref["nx"], nz=ref["nx"],
                          with_visual=True,
                          force_field=cfg["forceField"],
                          linear_solver=cfg["linearSolver"])
    return rootNode


# ---------------------------------------------------------------------------
# Reference-mesh driver: solve + table + plots (mirrors run_reference_scene)
# ---------------------------------------------------------------------------

def run_reference_dirichlet(elem, mms):
    """Solve au maillage de reference, ecrit la table + les plots PNG
    (profil 1D + slices 3D) dans results/, comme run_reference_scene mais
    pour le pipeline Dirichlet-only."""
    cfg = load_params()
    ref = cfg["reference"]
    L, E = cfg["length"], cfg["youngModulus"]
    nu   = ref["nu"]
    nx = ny = nz = ref["nx"]
    ff   = cfg["forceField"]
    ls   = cfg["linearSolver"]

    sol = solve_dirichlet(elem, mms, L, E, nu, nx, ny, nz,
                          force_field=ff, linear_solver=ls)
    l2  = elem.compute_l2(sol, mms, L)
    h1  = elem.compute_h1(sol, mms, L)

    label = elem.LABEL + " DIRICHLET"
    tag   = label.replace(" ", "_")
    stem  = f"{mms.name}_{tag}_nu{nu}_nx{nx}"

    xyz = sol.nodes[:, :3]
    write_solution_table(f"solution_{stem}", xyz,
                         np.column_stack([sol.ux, sol.uy, sol.uz]),
                         lambda xi, yi, zi: mms.u_ex(xi, yi, zi, L),
                         RESULTS_DIR, {"L2": l2, "H1_semi": h1})
    plot_solution_profile(f"solution_{stem}", sol, mms, L, nx, ny, nz,
                          label, nu, l2, h1)
    plot_solution_slices(f"fields3D_{stem}", sol, mms, L, nx, ny, nz,
                         label, nu)
    print(f"Dirichlet-only  nx={nx}  L2={l2:.6e}  H1={h1:.6e}")


if __name__ == "__main__":
    run_reference_dirichlet(element_tet, mms)
