"""
Quadratic MMS: û_ex(ξ) = ξ(1-ξ),  ξ = x/L ∈ [0,1]

Non-dimensional problem:  Ê·û'' + f̂ = 0  =>  f̂(ξ) = 2Ê  (Ê = E/L)

BC:
    û(0)  = 0      (Dirichlet)
    û'(1) = -1     (Neumann)  =>  F̂_N = -Ê
"""

import numpy as np

from mms_utils import (
    load_params,
    midpoint_quadrature,
    gauss2_quadrature,
    assemble_nodal_forces,
    build_bar_scene,
    run_bar_mms,
    l2_error,
    h1_semi_error,
    write_solution_table,
    plot_solution,
    convergence_study,
)

CASE_NAME = "quadratic"


def u_ex(xi):
    return xi * (1.0 - xi)


def du_ex(xi):
    return 1.0 - 2.0 * xi


def f_body(xi, E_eff):
    return 2.0 * E_eff


def make_apply_bcs(E_eff):
    def apply_bcs(Bar, nx):
        Bar.addObject('FixedProjectiveConstraint', indices=0)
        Bar.addObject('ConstantForceField',
                      name="NeumannTip",
                      indices=nx - 1,
                      forces=-E_eff)
    return apply_bcs


def _run(L, E, nx):
    E_eff        = E / L
    nodes        = np.linspace(0, 1, nx)
    nodal_forces = assemble_nodal_forces(lambda xi: f_body(xi, E_eff), nodes, midpoint_quadrature)
    return run_bar_mms(E_eff, nx, nodal_forces, make_apply_bcs(E_eff))


def createScene(rootNode):
    cfg  = load_params()
    L, E = cfg["length"], cfg["youngModulus"]
    nx   = cfg["nx"]
    E_eff        = E / L
    nodes        = np.linspace(0, 1, nx)
    nodal_forces = assemble_nodal_forces(lambda xi: f_body(xi, E_eff), nodes, midpoint_quadrature)
    build_bar_scene(rootNode, E_eff, nx, nodal_forces, make_apply_bcs(E_eff))
    return rootNode


if __name__ == "__main__":
    cfg     = load_params()
    L, E    = cfg["length"], cfg["youngModulus"]
    nx      = cfg["nx"]
    nx_list = cfg["nxConvergence"][CASE_NAME]

    # Single solution
    x0, u_h = _run(L, E, nx)
    l2  = l2_error(x0, u_h, u_ex, midpoint_quadrature)
    h1  = h1_semi_error(x0, u_h, du_ex, gauss2_quadrature)
    write_solution_table(CASE_NAME, x0, u_h, u_ex, {"L2": l2, "H1_semi": h1})
    plot_solution(CASE_NAME, x0, u_h, u_ex, r"$x(1-x)$")

    # Convergence study
    convergence_study(CASE_NAME, nx_list,
        run_fn     = lambda nx: _run(L, E, nx),
        error_fns  = {"L2":      lambda x, u: l2_error(x, u, u_ex, midpoint_quadrature),
                      "H1_semi": lambda x, u: h1_semi_error(x, u, du_ex, gauss2_quadrature)},
        ref_slopes = {r"O(h$^2$)": ("L2", 2), r"O(h$^1$)": ("H1_semi", 1)})
