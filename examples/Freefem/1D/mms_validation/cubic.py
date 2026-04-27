"""
Cubic MMS: û_ex(ξ) = L·ξ²(1-ξ),  ξ = x/L ∈ [0,1]

Non-dimensional problem:  Ê·û'' + f̂ = 0  =>  f̂(ξ) = E(6ξ - 2)  (Ê = E/L)

BC:
    û(0)  = 0    (Dirichlet)
    û'(1) = -L   (Neumann)  =>  F̂_N = Ê·(-L) = -E
"""

import numpy as np

from mms_utils import (
    load_params,
    gauss2_quadrature,
    assemble_nodal_forces,
    build_bar_scene,
    run_bar_mms,
    l2_error,
    write_solution_table,
    plot_solution,
    convergence_study,
)

CASE_NAME = "cubic"


def u_ex(xi, L):
    return L * xi**2 * (1.0 - xi)


def f_body(xi, E):
    return E * (6.0 * xi - 2.0)


def make_apply_bcs(E):
    def apply_bcs(Bar, nx):
        Bar.addObject('FixedProjectiveConstraint', indices=0)
        Bar.addObject('ConstantForceField',
                      name="NeumannTip",
                      indices=nx - 1,
                      forces=-E)
    return apply_bcs


def _run(L, E, nx):
    E_eff        = E / L
    nodes        = np.linspace(0, 1, nx)
    nodal_forces = assemble_nodal_forces(lambda xi: f_body(xi, E), nodes, gauss2_quadrature)
    return run_bar_mms(E_eff, nx, nodal_forces, make_apply_bcs(E))


def createScene(rootNode):
    cfg  = load_params()
    L, E = cfg["length"], cfg["youngModulus"]
    nx   = cfg["nx"]
    E_eff        = E / L
    nodes        = np.linspace(0, 1, nx)
    nodal_forces = assemble_nodal_forces(lambda xi: f_body(xi, E), nodes, gauss2_quadrature)
    build_bar_scene(rootNode, E_eff, nx, nodal_forces, make_apply_bcs(E))
    return rootNode


if __name__ == "__main__":
    cfg     = load_params()
    L, E    = cfg["length"], cfg["youngModulus"]
    nx      = cfg["nx"]
    nx_list = cfg["nxConvergence"][CASE_NAME]

    u_exact = lambda xi: u_ex(xi, L)

    # Single solution
    x0, u_h = _run(L, E, nx)
    err      = l2_error(x0, u_h, u_exact, gauss2_quadrature)
    write_solution_table(CASE_NAME, x0, u_h, u_exact, {"L2": err})
    plot_solution(CASE_NAME, x0, u_h, u_exact, r"$Lx^2(1-x)$")

    # Convergence study
    convergence_study(CASE_NAME, nx_list,
        run_fn     = lambda nx: _run(L, E, nx),
        error_fns  = {"L2": lambda x, u: l2_error(x, u, u_exact, gauss2_quadrature)},
        ref_slopes = {r"O(h$^2$)": ("L2", 2)})
