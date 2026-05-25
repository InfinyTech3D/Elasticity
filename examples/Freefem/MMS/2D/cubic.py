"""
Cubic 2D MMS on [0,L]^2 with linear-elasticity constitutive law:

    u_ex(x, y) = ( x (L - x)(x + y),
                   y (L - y)(y - x) )

    sigma = lambda tr(eps) I + 2 mu eps,
    with (lambda, mu) selected per dim (plane stress vs plane strain).
"""

import numpy as np

from manufactured_solution import MMSCase2D, lame


class Cubic(MMSCase2D):
    name       = "cubic"
    plot_label = r"$u_x = x(L-x)(x+y),\ u_y = y(L-y)(y-x)$"

    def u_ex(self, x, y, L):
        return (x * (L - x) * (x + y),
                y * (L - y) * (y - x))

    def grad_u_ex(self, x, y, L):
        dux_dx = (L - 2*x) * (x + y) + x * (L - x)
        dux_dy = x * (L - x)
        duy_dx = -y * (L - y)
        duy_dy = (L - 2*y) * (y - x) + y * (L - y)
        return np.array([[dux_dx, dux_dy],
                         [duy_dx, duy_dy]])

    def source(self, x, y, E, nu, L, dim):
        lam, mu = lame(E, nu, dim)
        d2ux_dxx = 2*L - 6*x - 2*y
        d2ux_dxy = L - 2*x
        d2ux_dyy = np.zeros_like(np.asarray(x, float))
        d2uy_dxx = np.zeros_like(np.asarray(x, float))
        d2uy_dxy = -(L - 2*y)
        d2uy_dyy = 2*L - 6*y + 2*x
        fx = -((lam + 2*mu) * d2ux_dxx + lam * d2uy_dxy
             + mu * (d2ux_dyy + d2uy_dxy))
        fy = -(mu * (d2ux_dxy + d2uy_dxx) + lam * d2ux_dxy
             + (lam + 2*mu) * d2uy_dyy)
        return (fx, fy)


from beam import case_scene, element_quad, element_tri

mms         = Cubic()
createScene = case_scene(mms, element_quad)


if __name__ == "__main__":
    from beam import run_reference_scene

    run_reference_scene(element_quad, mms)
