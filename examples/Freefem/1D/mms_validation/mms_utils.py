"""Shared utilities for 1D MMS validation."""

import numpy as np
import Sofa
import Sofa.Core
import Sofa.Simulation


# ---------------------------------------------------------------------------
# Quadrature rules
# Approximation: integral_{x_i}^{x_{i+1}} g(x) dx
#                ~= (h/2) * sum_k( w_k * g(x_k) )
# where x_k = (x_i + x_{i+1})/2 + (h/2)*xi_k
# ---------------------------------------------------------------------------

def midpoint_quadrature(g, x1, x2):
    """1-point Gauss rule: n_g=1, xi=[0], w=[2]."""
    h   = x2 - x1
    x_k = 0.5 * (x1 + x2)
    return (h / 2.0) * 2.0 * g(x_k)


def gauss2_quadrature(g, x1, x2):
    """2-point Gauss rule: n_g=2, xi=[-1/sqrt(3), +1/sqrt(3)], w=[1, 1]."""
    h     = x2 - x1
    x_mid = 0.5 * (x1 + x2)
    xi    = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    x_k   = x_mid + (h / 2.0) * xi
    return (h / 2.0) * (g(x_k[0]) + g(x_k[1]))


# ---------------------------------------------------------------------------
# FEM assembly
# ---------------------------------------------------------------------------

def assemble_nodal_forces(f_body, nodes, quadrature):
    """
    Assemble the consistent nodal force vector F_i = integral f_body(x) phi_i(x) dx.

    f_body    : callable x -> float
    nodes     : 1-D array of node coordinates
    quadrature: midpoint_quadrature or gauss2_quadrature
    """
    forces = np.zeros(len(nodes))
    for i in range(len(nodes) - 1):
        x1, x2 = nodes[i], nodes[i + 1]
        h = x2 - x1
        forces[i]     += quadrature(lambda x, x1=x1, x2=x2, h=h: f_body(x) * (x2 - x) / h, x1, x2)
        forces[i + 1] += quadrature(lambda x, x1=x1, x2=x2, h=h: f_body(x) * (x - x1) / h, x1, x2)
    return forces


# ---------------------------------------------------------------------------
# Error norms
# ---------------------------------------------------------------------------

def l2_error(nodes, u_h, u_ex, quadrature):
    """L2 error norm: sqrt( integral (u_h - u_ex)^2 dx ) over the mesh."""
    total = 0.0
    for i in range(len(nodes) - 1):
        x1, x2 = nodes[i], nodes[i + 1]
        h = x2 - x1
        u_a, u_b = u_h[i], u_h[i + 1]
        u_interp = lambda x, x1=x1, h=h, u_a=u_a, u_b=u_b: u_a + (u_b - u_a) * (x - x1) / h
        total += quadrature(lambda x: (u_interp(x) - u_ex(x)) ** 2, x1, x2)
    return np.sqrt(total)


def h1_semi_error(nodes, u_h, du_ex, quadrature):
    """H1 semi-norm error: sqrt( integral (du_h - du_ex)^2 dx ) over the mesh."""
    total = 0.0
    for i in range(len(nodes) - 1):
        x1, x2 = nodes[i], nodes[i + 1]
        h = x2 - x1
        # du_h = sum_a u_a * dphi_a/dx, with dphi_a/dx = -1/h, dphi_b/dx = +1/h
        du_h = u_h[i] * (-1.0 / h) + u_h[i + 1] * (1.0 / h)
        total += quadrature(lambda x, du_h=du_h: (du_h - du_ex(x)) ** 2, x1, x2)
    return np.sqrt(total)


# ---------------------------------------------------------------------------
# SOFA runner
# ---------------------------------------------------------------------------

def run_bar_mms(length, young_modulus, poisson_ratio, nx, nodal_forces, apply_bcs):
    """
    Build and run a static 1D bar simulation.

    nodal_forces : array of length nx, assembled body force vector
    apply_bcs    : callable(Bar, nx) that adds all boundary conditions
                   (FixedProjectiveConstraint + Neumann ConstantForceField)

    Returns (node_positions, displacements).
    """
    root = Sofa.Core.Node("root")

    root.addObject('RequiredPlugin', pluginName=[
        "Elasticity",
        "Sofa.Component.Constraint.Projective",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.MechanicalLoad",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Dynamic",
    ])
    root.addObject('DefaultAnimationLoop')

    h         = length / (nx - 1)
    positions = [[i * h] for i in range(nx)]
    edges     = [[i, i + 1] for i in range(nx - 1)]

    Bar = root.addChild('Bar')
    Bar.addObject('NewtonRaphsonSolver',
                  name="newtonSolver",
                  maxNbIterationsNewton=10,
                  absoluteResidualStoppingThreshold=1e-10,
                  printLog=False)
    Bar.addObject('SparseLDLSolver',
                  name="linearSolver",
                  template="CompressedRowSparseMatrixd")
    Bar.addObject('StaticSolver',
                  name="staticSolver",
                  newtonSolver="@newtonSolver",
                  linearSolver="@linearSolver")

    dofs = Bar.addObject('MechanicalObject',
                         name="dofs",
                         template="Vec1d",
                         position=positions)
    Bar.addObject('EdgeSetTopologyContainer', name="topology", edges=edges)
    Bar.addObject('LinearSmallStrainFEMForceField',
                  name="FEM",
                  template="Vec1d",
                  youngModulus=young_modulus,
                  poissonRatio=poisson_ratio,
                  topology="@topology")

    Bar.addObject('ConstantForceField',
                  name="BodyForce",
                  indices=list(range(nx)),
                  forces=nodal_forces)

    apply_bcs(Bar, nx)

    Sofa.Simulation.init(root)
    x0 = dofs.position.array().copy().flatten()
    Sofa.Simulation.animate(root, root.dt.value)
    u  = dofs.position.array().flatten() - x0
    Sofa.Simulation.unload(root)
    return x0, u
