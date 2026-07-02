"""Rigid-rotation test for the corotational FEM force field.

With a pure rigid-body rotation x = R·X the strain is zero.
so the internal elastic force must disappear. 
We expect a corotational force field to give zero force but for a linear small-strain field 
to yield f != 0, because it sees the rotation as strain.

Run:  python test_rigid_rotation.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from beam import load_params, element_quad, RESULTS_DIR

ANGLE_DEG   = 90.0
TRANSLATION = [0.0, 0.0, 0.0]
RATIO_TOL   = 1e-6          # corotational |f| must be < RATIO_TOL * linear |f|
N_STEPS     = 18            # only used by the commented-out solve-based path


def Rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def build_scene(root, L, E, nu, nx, force_field):
    root.addObject("RequiredPlugin", pluginName=[
        "Elasticity",
        "Sofa.Component.Mass",
        "Sofa.Component.ODESolver.Forward",        # EulerExplicitSolver
        "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Grid",
        "Sofa.Component.Topology.Container.Dynamic",
        # TODO: Revisit when testing equilibrium
        # "Sofa.Component.Constraint.Projective",
        # "Sofa.Component.Engine.Select",
        # "Sofa.Component.LinearSolver.Iterative",
        # "Sofa.Component.ODESolver.Backward",
    ])
    root.addObject("DefaultAnimationLoop")
    root.gravity = [0.0, 0.0, 0.0]

    grid = root.addChild("Grid")
    grid.addObject("RegularGridTopology", name="grid", nx=nx, ny=nx, nz=1,
                   min=[0.0, 0.0, 0.0], max=[L, L, 0.0])

    with root.addChild("Beam") as beam:
        # An ODE solver. TODO: Revisit when testing equilibrium
        beam.addObject("EulerExplicitSolver", name="odeSolver")

        # TODO: An attempt to demonstrate inconsistencies through a solver 
        #       Cannot work for now. It should be revisited. 
        # beam.addObject("StaticSolver", name="staticSolver", printLog=False)
        # beam.addObject("NewtonRaphsonSolver", name="newton",
        #                maxNbIterationsNewton=25,
        #                absoluteResidualStoppingThreshold=1e-10, printLog=False)
        # beam.addObject("CGLinearSolver", name="linearSolver",
        #                iterations=5000, tolerance=1e-12, threshold=1e-12)
        # -------------------------------------------------------------------------

        dofs = beam.addObject("MechanicalObject", name="dofs", template="Vec3d",
                          position="@../Grid/grid.position")
        beam.addObject("UniformMass", name="mass", totalMass=1.0)
        element_quad.add_topology(beam)
        beam.addObject(force_field, name="FEM", template="Vec3d",
                   youngModulus=E, poissonRatio=nu, topology="@topology")

        # TODO: An attempt to demonstrate inconsistencies through a solver. These are the BCs.
        #       Cannot work for now. It should be revisited. 
        # eps = 1e-5
        # beam.addObject("BoxROI", name="boundary", template="Vec3d", drawBoxes=False,
        #                box=[[-eps,   -eps,   -1.0, eps,   L + eps, 1.0],   # x = 0
        #                     [L - eps, -eps,  -1.0, L + eps, L + eps, 1.0],  # x = L
        #                     [-eps,   -eps,   -1.0, L + eps, eps,   1.0],    # y = 0
        #                     [-eps,   L - eps, -1.0, L + eps, L + eps, 1.0]])# y = L
        # beam.addObject("AffineMovementProjectiveConstraint",
        #                meshIndices=list(range(nx * nx)),
        #                indices="@boundary.indices",
        #                beginConstraintTime=0.0, endConstraintTime=float(N_STEPS),
        #                rotation=Rz(np.radians(ANGLE_DEG)).tolist(),
        #                translation=TRANSLATION)
        return dofs


def run(force_field):
    """Impose x = R·X and return (rest X, rotated config, max internal force magnitude)."""
    cfg = load_params()
    L, E   = cfg["length"], cfg["youngModulus"]
    nu, nx = cfg["reference"]["nu"], cfg["reference"]["nx"]

    root = Sofa.Core.Node("root")
    dofs = build_scene(root, L, E, nu, nx, force_field)
    Sofa.Simulation.init(root)

    # Take the rest positions and apply x = R @ X
    # This will apply the rigid-body rotation to all the dofs
    X  = dofs.rest_position.array().copy()
    Xr = (Rz(np.radians(ANGLE_DEG)) @ X.T).T + np.array(TRANSLATION)
    with dofs.position.writeable() as p:
        p[:] = Xr

    # Animate the scene once. What this does is call addForce, evaluated at x = R·X at the start of the step. 
    # force is read right after and is the internal force of the pure rotation.
    # In the case of LinearSmallStrainFEMForceField, we expect to see an emergent force, resulting 
    # from the force field understanding the rotation as deformation.
    # In the case of CorotationalFEMForceField, we expect to see zero force, resulting from the 
    # force field understanding the displacement as a rigid-body rotation causing zero strain.
    Sofa.Simulation.animate(root, 1e-8)
    f = dofs.force.array().copy()
    Sofa.Simulation.unload(root)

    return X, Xr, float(np.max(np.linalg.norm(f, axis=1)))


def write_metrics(forces, path):
    ''' Write the measured forces to a file'''
    f_cr  = forces["CorotationalFEMForceField"]
    f_lin = forces["LinearSmallStrainFEMForceField"]
    with open(path, "w") as f:
        f.write(f"Rigid-rotation internal force   angle={ANGLE_DEG} deg   "
                f"translation={TRANSLATION}\n")
        f.write("=" * 74 + "\n")
        for ff, fmax in forces.items():
            f.write(f"  {ff:32s} max |internal force| = {fmax:.6e}\n")
        ratio = f_cr / f_lin if f_lin != 0 else float("nan")
        f.write(f"\n  corotational / linear ratio = {ratio:.3e}  "
                f"(rigid-body property holds if << 1)\n")


def plot_forces(X, Xr, forces, path):
    ''' Plot the forces throughout the 2D plane'''
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 5))

    ax0.scatter(X[:, 0], X[:, 1], s=12, c="lightgray", label="rest X")
    ax0.scatter(Xr[:, 0], Xr[:, 1], s=12, c="tab:red", label="imposed R·X")
    ax0.set_aspect("equal"); ax0.set_xlabel("x"); ax0.set_ylabel("y")
    ax0.set_title(f"imposed rigid rotation {ANGLE_DEG}°")
    ax0.legend(loc="best"); ax0.grid(True, alpha=0.3)

    names = list(forces.keys())
    vals  = [max(forces[n], 1e-30) for n in names]     # floor for log scale
    ax1.bar(range(len(names)), vals, color=["tab:green", "tab:red"])
    ax1.set_yscale("log")
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels([n.replace("FEMForceField", "") for n in names])
    ax1.set_ylabel("max |internal force|")
    ax1.set_title("internal force under rigid rotation\n(corotational must be ~0)")
    ax1.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    forces = {}
    geom = None
    for ff in ("CorotationalFEMForceField", "LinearSmallStrainFEMForceField"):
        X, Xr, fmax = run(ff)
        forces[ff] = fmax
        geom = (X, Xr)
        print(f"{ff:32s} max |internal force| = {fmax:.6e}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    write_metrics(forces, os.path.join(RESULTS_DIR, "rigid_rotation_metrics.txt"))
    plot_forces(*geom, forces, os.path.join(RESULTS_DIR, "rigid_rotation_force.png"))
