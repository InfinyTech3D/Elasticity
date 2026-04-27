"""
Sinusoidal MMS: u_ex(x) = sin(pi*x)

Equilibrium E*u'' + f = 0  =>  f(x) = E*pi^2*sin(pi*x)

BC:
    u(0)  = 0                    (Dirichlet)
    u'(L) = pi*cos(pi*L)         (Neumann)  =>  F_N = E*pi*cos(pi*L)

Note: f(x) is transcendental so 2-point Gauss quadrature is used for assembly.
The H1 error must also be evaluated with gauss2_quadrature to correctly recover
O(h^1) convergence — the element midpoint is a superconvergence point for the
gradient of P1 elements, so midpoint_quadrature would artificially give O(h^2).
"""

import numpy as np

from mms_utils import (
    load_params,
    gauss2_quadrature,
    assemble_nodal_forces,
    build_bar_scene,
    run_bar_mms,
    l2_error,
    h1_semi_error,
    write_solution_table,
    write_convergence_table,
    plot_solution,
    plot_convergence,
)

CASE_NAME = "sinusoidal"


def u_ex(x, L=None):
    return np.sin(np.pi * x)


def du_ex(x):
    return np.pi * np.cos(np.pi * x)


def f_body(x, E, L=None):
    return E * np.pi**2 * np.sin(np.pi * x)


def make_apply_bcs(E, L):
    def apply_bcs(Bar, nx):
        Bar.addObject('FixedProjectiveConstraint', indices=0)
        Bar.addObject('ConstantForceField',
                      name="NeumannTip",
                      indices=nx - 1,
                      forces=E * np.pi * np.cos(np.pi * L))
    return apply_bcs


def _run(L, E, nu, nx):
    nodes        = np.linspace(0, L, nx)
    nodal_forces = assemble_nodal_forces(lambda x: f_body(x, E), nodes, gauss2_quadrature)
    return run_bar_mms(L, E, nu, nx, nodal_forces, make_apply_bcs(E, L))


def main():
    cfg     = load_params()
    L, E    = cfg["length"], cfg["youngModulus"]
    nu, nx  = cfg["poissonRatio"], cfg["nx"]
    nx_list = cfg["nxConvergence"][CASE_NAME]

    # Single solution
    x0, u_h = _run(L, E, nu, nx)
    l2  = l2_error(x0, u_h, u_ex, gauss2_quadrature)
    h1  = h1_semi_error(x0, u_h, du_ex, gauss2_quadrature)
    write_solution_table(CASE_NAME, x0, u_h, u_ex, {"L2": l2, "H1_semi": h1})
    plot_solution(CASE_NAME, x0, u_h, u_ex, r"$\sin(\pi x)$")

    # Convergence study
    hs, l2_list, h1_list, rows = [], [], [], []
    for k, nx_k in enumerate(nx_list):
        h_k          = L / (nx_k - 1)
        x0_k, u_h_k  = _run(L, E, nu, nx_k)
        l2_k         = l2_error(x0_k, u_h_k, u_ex, gauss2_quadrature)
        h1_k         = h1_semi_error(x0_k, u_h_k, du_ex, gauss2_quadrature)
        log_h        = np.log(h_k / hs[-1]) if k > 0 else None
        rate_l2      = f"{np.log(l2_k / l2_list[-1]) / log_h:.2f}" if k > 0 else ""
        rate_h1      = f"{np.log(h1_k / h1_list[-1]) / log_h:.2f}" if k > 0 else ""
        hs.append(h_k)
        l2_list.append(l2_k)
        h1_list.append(h1_k)
        rows.append({"nx": nx_k, "h": h_k, "L2": l2_k, "rate_L2": rate_l2,
                     "H1_semi": h1_k, "rate_H1": rate_h1})

    write_convergence_table(CASE_NAME, rows)
    plot_convergence(CASE_NAME, hs,
                     {"L2": l2_list, "H1 semi": h1_list},
                     {r"O(h$^2$)": (l2_list, 2), r"O(h$^1$)": (h1_list, 1)})


def createScene(rootNode):
    cfg  = load_params()
    L, E = cfg["length"], cfg["youngModulus"]
    nu   = cfg["poissonRatio"]
    nx   = cfg["nx"]
    nodes        = np.linspace(0, L, nx)
    nodal_forces = assemble_nodal_forces(lambda x: f_body(x, E), nodes, gauss2_quadrature)
    build_bar_scene(rootNode, L, E, nu, nx, nodal_forces, make_apply_bcs(E, L))
    return rootNode


if __name__ == "__main__":
    main()
