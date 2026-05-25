"""
Trigonometric 2D MMS on [0,L]^2 with linear-elasticity constitutive law:

    u_ex(x, y) = ( sin(pi x / L) cos(pi y / L),
                   cos(pi x / L) sin(pi y / L) )

    sigma = lambda tr(eps) I + 2 mu eps,
    with (lambda, mu) selected per dim (plane stress vs plane strain).
"""

import numpy as np

from manufactured_solution import MMSCase2D, lame


class Trigonometric(MMSCase2D):
    name       = "trigonometric"
    plot_label = (r"$u_x = \sin(\pi x/L)\cos(\pi y/L),\ "
                  r"u_y = \cos(\pi x/L)\sin(\pi y/L)$")

    def u_ex(self, x, y, L):
        k = np.pi / L
        return (np.sin(k * x) * np.cos(k * y),
                np.cos(k * x) * np.sin(k * y))

    def grad_u_ex(self, x, y, L):
        k = np.pi / L
        dux_dx =  k * np.cos(k * x) * np.cos(k * y)
        dux_dy = -k * np.sin(k * x) * np.sin(k * y)
        duy_dx = -k * np.sin(k * x) * np.sin(k * y)
        duy_dy =  k * np.cos(k * x) * np.cos(k * y)
        return np.array([[dux_dx, dux_dy],
                         [duy_dx, duy_dy]])

    def source(self, x, y, E, nu, L, dim):
        lam, mu = lame(E, nu, dim)
        k  = np.pi / L
        ux = np.sin(k * x) * np.cos(k * y)
        uy = np.cos(k * x) * np.sin(k * y)
        d2ux_dxx = -k**2 * ux
        d2ux_dyy = -k**2 * ux
        d2ux_dxy = -k**2 * np.cos(k * x) * np.sin(k * y)
        d2uy_dxx = -k**2 * uy
        d2uy_dyy = -k**2 * uy
        d2uy_dxy = -k**2 * np.sin(k * x) * np.cos(k * y)
        fx = -((lam + 2*mu) * d2ux_dxx + lam * d2uy_dxy
             + mu * (d2ux_dyy + d2uy_dxy))
        fy = -(mu * (d2ux_dxy + d2uy_dxx) + lam * d2ux_dxy
             + (lam + 2*mu) * d2uy_dyy)
        return (fx, fy)


from beam import case_scene, element_quad, element_tri

mms         = Trigonometric()
createScene = case_scene(mms, element_quad)


if __name__ == "__main__":
    from beam import run_reference_scene

    run_reference_scene(element_quad, mms)
