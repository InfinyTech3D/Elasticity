import numpy as np

from manufactured_solution import MMSCase3D, lame
from solidCoro import (case_scene, run_reference_scene,
                       element_hex, hex_q1_rule)

SINUS_AMPLITUDE = 1.0
THETA = np.radians(45.0)


def _Rz(theta):
    """Rotation matrix about z-axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, -s, 0],
                     [ s,  c, 0],
                     [ 0,  0, 1]])


class SinusCorotational(MMSCase3D):
    name = "sinus_corotational"
    source_quadrature_hex = staticmethod(hex_q1_rule(2))

    _Rtheta = _Rz(THETA)

    def _u_d(self, x, y, z, L):
        """Deformational part only (small sinusoidal field)."""
        k = np.pi / L
        A = SINUS_AMPLITUDE
        return np.array([A * np.sin(k*x) * np.sin(k*y),
                         A * np.sin(k*y) * np.sin(k*z),
                         A * np.sin(k*z) * np.sin(k*x)])

    def _grad_u_d(self, x, y, z, L):
        """∇u_d (3x3)."""
        k = np.pi / L
        A = SINUS_AMPLITUDE
        zero = 0.0 if np.isscalar(x) else np.zeros_like(np.asarray(x, float))
        return np.array([
            [A*k*np.cos(k*x)*np.sin(k*y),  A*k*np.sin(k*x)*np.cos(k*y), zero],
            [zero, A*k*np.cos(k*y)*np.sin(k*z),  A*k*np.sin(k*y)*np.cos(k*z)],
            [A*k*np.cos(k*x)*np.sin(k*z),  zero,  A*k*np.cos(k*z)*np.sin(k*x)],
        ])

    def u_ex(self, x, y, z, L):
        R = self._Rtheta
        x_vec = np.array([x, y, z])
        rbm = (R - np.eye(3)) @ x_vec   # composante rigide
        ud  = self._u_d(x, y, z, L)
        total = rbm + ud
        return (total[0], total[1], total[2])

    def grad_u_ex(self, x, y, z, L):
        R = self._Rtheta
        grad_rbm = R - np.eye(3)
        grad_ud  = self._grad_u_d(x, y, z, L)
        return grad_rbm + grad_ud

    def source(self, x, y, z, E, nu, L):
        lam, mu = lame(E, nu)
        R = self._Rtheta

        A = SINUS_AMPLITUDE
        p = np.pi / L
        sx, sy, sz = np.sin(p*x), np.sin(p*y), np.sin(p*z)
        cx, cy, cz = np.cos(p*x), np.cos(p*y), np.cos(p*z)

        lap_ux = -2 * p**2 * sx * sy
        lap_uy = -2 * p**2 * sy * sz
        lap_uz = -2 * p**2 * sz * sx

        d_divu_dx = p**2 * (-sx*sy + cz*cx)
        d_divu_dy = p**2 * ( cx*cy - sy*sz)
        d_divu_dz = p**2 * ( cy*cz - sz*sx)

        div_sigma_local = A * np.array([
            (lam + mu) * d_divu_dx + mu * lap_ux,
            (lam + mu) * d_divu_dy + mu * lap_uy,
            (lam + mu) * d_divu_dz + mu * lap_uz,
        ])

        f = R @ (-div_sigma_local)
        return (f[0], f[1], f[2])

    def traction(self, x, y, z, nx, ny, nz, E, nu, L):
        """t = R_theta * sigma_bar . n  sur ∂Ω_N."""
        lam, mu = lame(E, nu)
        R = self._Rtheta
        grad_ud = self._grad_u_d(x, y, z, L)
        eps_bar = 0.5 * (grad_ud + grad_ud.T)
        tr_eps  = eps_bar[0,0] + eps_bar[1,1] + eps_bar[2,2]
        sigma_bar = lam * tr_eps * np.eye(3) + 2 * mu * eps_bar
        n = np.array([nx, ny, nz])
        t = R @ (sigma_bar @ n)
        return (t[0], t[1], t[2])
 

    def apply_bcs(self, Solid, nodes_3d, L):
        eps = 1e-10
        xyz = nodes_3d[:, :3]
        dofs = Solid.dofs

    # ── Shift trick sur TOUS les nœuds ────────────────────────────────
    # → Newton part d'un état cohérent, proche de la solution
        with dofs.position.writeable() as pos:
             for i, (x, y, z) in enumerate(xyz):
                ux, uy, uz = self.u_ex(x, y, z, L)
                pos[i] = [x + ux, y + uy, z + uz]

    
        boundary_indices = [
            i for i, (x, y, z) in enumerate(xyz)
            if x < eps or x > L - eps
            or y < eps or y > L - eps
            or z < eps or z > L - eps
    ]
        Solid.addObject("FixedProjectiveConstraint",
                    name="fix_boundary",
                    indices=boundary_indices)




mms = SinusCorotational()

createScene = case_scene(mms, element_hex)


if __name__ == "__main__":
    run_reference_scene(element_hex, mms)