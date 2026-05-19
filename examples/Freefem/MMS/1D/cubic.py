"""
Cubic MMS (non-dimensional, x = x_dim/L ∈ [0,1], E = E_dim/L):

    u_ex(x) = x²(1-x)
    f(x)    = E(6x - 2)

BC:
    u(0)   = 0     (Dirichlet)
    u'(1)  = -1    (Neumann)  =>  F_N = E·u'(1) = -E
"""

from manufactured_solution import MMSCase1D
from bar import load_params, line_quadrature, build_bar_scene, run_scene


class Cubic(MMSCase1D):
    name       = "cubic"
    plot_label = r"$x^2(1-x)$"
    quadrature = staticmethod(line_quadrature(2))

    def u_ex(self, xi):
        return xi**2 * (1.0 - xi)

    def du_ex(self, xi):
        return 2.0 * xi - 3.0 * xi**2

    def source(self, xi, E):
        return E * (6.0 * xi - 2.0)


mms = Cubic()


def createScene(rootNode):
    cfg  = load_params()
    L, E = cfg["length"], cfg["youngModulus"]
    build_bar_scene(rootNode, mms, E / L, cfg["nx"])
    return rootNode


if __name__ == "__main__":
    run_scene(mms)
