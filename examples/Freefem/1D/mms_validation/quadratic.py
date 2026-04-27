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
    write_convergence_table,
    plot_solution,
    plot_convergence,
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


def main():
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
    hs, errors, rows = [], [], []
    for k, nx_k in enumerate(nx_list):
        h_k          = L / (nx_k - 1)
        x0_k, u_h_k  = _run(L, E, nu, nx_k)
        e_k          = l2_error(x0_k, u_h_k, u_exact, midpoint_quadrature)
        rate         = f"{np.log(e_k / errors[-1]) / np.log(h_k / hs[-1]):.2f}" if k > 0 else ""
        hs.append(h_k)
        errors.append(e_k)
        rows.append({"nx": nx_k, "h": h_k, "L2": e_k, "rate": rate})

    write_convergence_table(CASE_NAME, rows)
    plot_convergence(CASE_NAME, hs, {"L2": errors}, {r"O(h$^2$)": (errors, 2)})


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
    main()
