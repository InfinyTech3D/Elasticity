"""
Quadratic MMS: u_ex(x) = x*(L-x)/L^2

Equilibrium E*u'' + f = 0  =>  f(x) = 2*E/L^2

BC:
    u(0)  = 0      (Dirichlet)
    u'(L) = -1/L   (Neumann)  =>  F_N = -E/L
"""

import numpy as np

from mms_utils import (
    load_params,
    midpoint_quadrature,
    assemble_nodal_forces,
    build_bar_scene,
    run_bar_mms,
    l2_error,
    write_solution_table,
    plot_solution,
    convergence_study,
)

CASE_NAME = "quadratic"


def u_ex(x, L):
    return x * (L - x) / L**2


def f_body(x, E, L):
    return 2.0 * E / L**2


def make_apply_bcs(E, L):
    def apply_bcs(Bar, nx):
        Bar.addObject('FixedProjectiveConstraint', indices=0)
        Bar.addObject('ConstantForceField',
                      name="NeumannTip",
                      indices=nx - 1,
                      forces=-E / L)
    return apply_bcs


def _run(L, E, nu, nx):
    nodes        = np.linspace(0, L, nx)
    nodal_forces = assemble_nodal_forces(lambda x: f_body(x, E, L), nodes, midpoint_quadrature)
    return run_bar_mms(L, E, nu, nx, nodal_forces, make_apply_bcs(E, L))


def createScene(rootNode):
    cfg  = load_params()
    L, E = cfg["length"], cfg["youngModulus"]
    nu   = cfg["poissonRatio"]
    nx   = cfg["nx"]
    nodes        = np.linspace(0, L, nx)
    nodal_forces = assemble_nodal_forces(lambda x: f_body(x, E, L), nodes, midpoint_quadrature)
    build_bar_scene(rootNode, L, E, nu, nx, nodal_forces, make_apply_bcs(E, L))
    return rootNode


if __name__ == "__main__":
    cfg     = load_params()
    L, E    = cfg["length"], cfg["youngModulus"]
    nu, nx  = cfg["poissonRatio"], cfg["nx"]
    nx_list = cfg["nxConvergence"][CASE_NAME]

    u_exact = lambda x: u_ex(x, L)

    # Single solution
    x0, u_h = _run(L, E, nu, nx)
    err      = l2_error(x0, u_h, u_exact, midpoint_quadrature)
    write_solution_table(CASE_NAME, x0, u_h, u_exact, {"L2": err})
    plot_solution(CASE_NAME, x0, u_h, u_exact, r"$x(L-x)/L^2$")

    # Convergence study
    convergence_study(CASE_NAME, L, nx_list,
        run_fn     = lambda nx: _run(L, E, nu, nx),
        error_fns  = {"L2": lambda x, u: l2_error(x, u, u_exact, midpoint_quadrature)},
        ref_slopes = {r"O(h$^2$)": ("L2", 2)})
