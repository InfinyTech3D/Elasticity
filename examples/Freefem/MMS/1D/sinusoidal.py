"""
Sinusoidal MMS (non-dimensional, x = x_dim/L ∈ [0,1], E = E_dim/L):

    u_ex(x) = sin(2πx)
    f(x)    = 4π²E·sin(2πx)

BC:
    u(0)   = 0                       (Dirichlet)
    u'(1)  = 2π·cos(2π) = 2π         (Neumann)  =>  F_N = E·u'(1) = 2πE

Note: f is transcendental so the 2-point Gauss rule is used for assembly.
The H1 error must also be evaluated with the 2-point rule to correctly recover
O(h^1) convergence — the element midpoint is a superconvergence point for the
gradient of P1 elements, so the 1-point rule would artificially give O(h^2).
"""

import numpy as np

from manufactured_solution import MMSCase1D
from bar import load_params, line_quadrature, build_bar_scene, run_scene


class Sinusoidal(MMSCase1D):
    name       = "sinusoidal"
    plot_label = r"$\sin(2\pi x)$"
    quadrature = staticmethod(line_quadrature(2))

    def u_ex(self, xi):
        return np.sin(2.0 * np.pi * xi)

    def du_ex(self, xi):
        return 2.0 * np.pi * np.cos(2.0 * np.pi * xi)

    def source(self, xi, E):
        return 4.0 * np.pi**2 * E * np.sin(2.0 * np.pi * xi)


mms = Sinusoidal()


def createScene(rootNode):
    cfg  = load_params()
    L, E = cfg["length"], cfg["youngModulus"]
    build_bar_scene(rootNode, mms, E / L, cfg["nx"])
    return rootNode


if __name__ == "__main__":
    run_scene(mms)
