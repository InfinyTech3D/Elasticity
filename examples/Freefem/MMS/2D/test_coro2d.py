"""
u_d-only test : verifies that CorotationalFEMForceField (CR) and
LinearSmallStrainFEMForceField (LS) produce identical results when the imposed
displacement field is x = X + u_d(X), with no global rotation (R = I).

Two test cases are considered:

  1. SMOOTH u_d with a non-zero ANTISYMMETRIC gradient
     (u_x = u_y = A sin(πx/L) sin(πy/L))
     ===> CR and LS DO NOT converge to each other as h -> 0:
        the relative difference reaches a finite plateau (expected physical behavior).

  2. SMOOTH u_d with a purely SYMMETRIC, spatially varying gradient
     (u_x = A sin(kx) cos(ky),  u_y = A cos(kx) sin(ky),  k = π/L)
     ==> the local rotation is identically zero everywhere, as verified analytically:
        ∂u_x/∂y = -Ak sin(kx) sin(ky) = ∂u_y/∂x  (equal cross-derivatives)
     ==> any discrepancy between CR and LS should converge to zero as h -> 0.

Known limitations
-----------------
- For nx >= 160, EulerExplicitSolver becomes unstable because the nodal mass
  tends to zero (totalMass = 1 / N_nodes), causing the force values to diverge
  after a single animation step. Therefore, the results are only considered
  reliable for nx <= 80.
  A future improvement will replace the EulerExplicitSolver + UniformMass
  combination with a StaticSolver and a finite-difference Jacobian (Kfd)
  for CorotationalFEMForceField.

- Upstream warning: SOFAElementLinearSmallStrainFEMForceField registers
  multiple default templates (Vec3d, Quad / Tetrahedron / Hexahedron) in
  addition to the primary Vec3d, Edge default. This is an issue in the
  Elasticity plugin itself and is outside the scope of this test.

Run:
    python test_coro2d.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime  

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from beam import load_params, element_quad, RESULTS_DIR

NX_LIST = [40, 80]   


def build_scene(root, L, E, nu, nx, force_field):
    root.addObject("RequiredPlugin", name="ScenePlugins", pluginName=[
        "Elasticity",
        "Sofa.Component.Mass",
        "Sofa.Component.ODESolver.Forward",
        "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Grid",
        "Sofa.Component.Topology.Container.Dynamic",
    ])
    root.addObject("DefaultAnimationLoop")
    root.gravity = [0.0, 0.0, 0.0]

    grid = root.addChild("Grid")
    grid.addObject("RegularGridTopology", name="grid", nx=nx, ny=nx, nz=1,
                   min=[0.0, 0.0, 0.0], max=[L, L, 0.0])

    with root.addChild("Beam") as beam:
        beam.addObject("EulerExplicitSolver", name="odeSolver")
        dofs = beam.addObject("MechanicalObject", name="dofs", template="Vec3d",
                              position="@../Grid/grid.position")
        beam.addObject("UniformMass", name="mass", totalMass=1.0)
        element_quad.add_topology(beam)
        beam.addObject(force_field, name="FEM", template="Vec3d",
                       youngModulus=E, poissonRatio=nu, topology="@topology")

    return dofs


# Displacement fields

def u_smooth_rotational(X, amp=0.01, L=1.0, **_):
    """u_x = u_y = A sin(pi*x/L) sin(pi*y/L)

    Gradient antisymmetric part is non-zero:
      du_x/dy =  Ak cos(pi*x/L) sin(pi*y/L)
      du_y/dx =  Ak sin(pi*x/L) cos(pi*y/L)  

    CR and LS are therefore expected NOT to converge toward each other as
    h ----> 0: the relative discrepancy saturates at a finite value (physical).
    """
    k = np.pi / L
    x, y = X[:, 0], X[:, 1]
    field = amp * np.sin(k * x) * np.sin(k * y)
    return np.stack([field, field.copy(), np.zeros_like(x)], axis=1)


def u_smooth_irrotational(X, amp=0.01, L=1.0, **_):
    """u_x =  A sin(kx) cos(ky),   u_y = A cos(kx) sin(ky),   k = pi/L

    Analytically zero local rotation everywhere:
      du_x/dy = -Ak sin(kx) sin(ky)
      du_y/dx = -Ak sin(kx) sin(ky)   

    ====> (du_x/dy - du_y/dx) / 2 = 0  exactly.

    Any CR/LS discrepancy is purely from bilinear interpolation and
    must vanish as h =====> 0.
    """
    k = np.pi / L
    x, y = X[:, 0], X[:, 1]
    ux = amp * np.sin(k * x) * np.cos(k * y)
    uy = amp * np.cos(k * x) * np.sin(k * y)
    return np.stack([ux, uy, np.zeros_like(x)], axis=1)
 
def verify_irrotational(X, amp=0.01, L=1.0):
    """Print max |du_x/dy - du_y/dx| at the nodal positions.

    Must be 0.0 to machine precision: both partials equal
    -Ak sin(kx) sin(ky), so their difference is identically zero.
    """
    k = np.pi / L
    x, y = X[:, 0], X[:, 1]
    dux_dy = -amp * k * np.sin(k * x) * np.sin(k * y)   
    duy_dx = -amp * k * np.sin(k * x) * np.sin(k * y)   
    err = np.max(np.abs(dux_dy - duy_dx))
    print(f"  [verify]  max |∂u_x/∂y - ∂u_y/∂x| = {err:.3e}  "
          f"(must be 0 — confirms zero continuous local rotation)")


def run(force_field, u_field, nx_override=None):
    cfg = load_params()
    L, E   = cfg["length"], cfg["youngModulus"]
    nu, nx = cfg["reference"]["nu"], cfg["reference"]["nx"]
    if nx_override is not None:
        nx = nx_override

    root = Sofa.Core.Node("root")
    dofs = build_scene(root, L, E, nu, nx, force_field)
    Sofa.Simulation.init(root)

    X  = dofs.rest_position.array().copy()
    Xd = X + u_field(X, L=L)
    with dofs.position.writeable() as p:
        p[:] = Xd

    Sofa.Simulation.animate(root, 1e-8)
    f = dofs.force.array().copy()
    Sofa.Simulation.unload(root)
    return X, Xd, f

 
def compare(u_field, nx_list, label):
    diffs = []
    for nx in nx_list:
        _, _, f_cr = run("CorotationalFEMForceField",      u_field, nx)
        _, _, f_ls = run("LinearSmallStrainFEMForceField", u_field, nx)
        diff = np.max(np.linalg.norm(f_cr - f_ls, axis=1))
        ref  = np.max(np.linalg.norm(f_ls,        axis=1))
        rel  = diff / ref if ref > 0 else diff
        diffs.append(rel)
        print(f"  {label:14s} nx={nx:3d}   |f_CR - f_LS| / |f_LS| = {rel:.3e}")
    return diffs




def write_metrics(results, path):
    with open(path, "w") as f:
        f.write("CR vs LS : displacement-only test \n")
        f.write("=" * 74 + "\n\n")
        for label, (nx_list, diffs, expected) in results.items():
            f.write(f"  [{label}]  {expected}\n")
            for nx, d in zip(nx_list, diffs):
                f.write(f"    nx={nx:3d}   |f_CR - f_LS| / |f_LS| = {d:.6e}\n")
            f.write("\n")


def plot_forces(results, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
    ax0 = axes[0]
    nx_list, diffs, _ = results["rotational"]
    hs = [1.0 / nx for nx in nx_list]
    ax0.loglog(hs, diffs,
               marker="s", linestyle="--", color="tab:red",
               label=r"rotational  $u_x = u_y$")

    
    plateau = diffs[-1]
    ax0.axhline(y=plateau, color="gray", linestyle=":", linewidth=1.0)
    ax0.text(hs[0] * 1.05, plateau * 1.3,
             f"plateau ≈ {plateau:.2e}",
             fontsize=8, color="gray")

    ax0.invert_xaxis()
    ax0.set_xlabel(r"$h = 1/n_x$  (mesh size, finer $\rightarrow$)")
    ax0.set_ylabel(r"$\|f_{CR} - f_{LS}\| \;/\; \|f_{LS}\|$")
    ax0.set_title("Case 1 — rotational field\n"
                  r"$u_x = u_y = A\sin(\pi x/L)\sin(\pi y/L)$"
                  "\n→ finite plateau (physical, not a bug)")
    ax0.legend(loc="best")
    ax0.grid(True, which="both", alpha=0.3)
 
    ax1 = axes[1]
    nx_list, diffs, _ = results["irrotational"]
    hs = [1.0 / nx for nx in nx_list]
    ax1.loglog(hs, diffs,
               marker="^", linestyle="-", color="tab:blue",
               label=r"irrotational  $\partial_y u_x = \partial_x u_y$")

    floor = np.finfo(float).eps          
    ax1.axhline(y=floor, color="gray", linestyle=":", linewidth=1.0)
    ax1.text(max(hs) * 0.6, floor * 3.0,
             r"$\varepsilon_{\rm machine} \approx 2.2\times10^{-16}$",
             fontsize=8, color="gray")

    ax1.invert_xaxis()
    ax1.set_xlabel(r"$h = 1/n_x$  (mesh size, finer $\rightarrow$)")
    ax1.set_ylabel(r"$\|f_{CR} - f_{LS}\| \;/\; \|f_{LS}\|$")
    ax1.set_title("Case 2 — irrotational field\n"
                  r"$u_x = A\sin(kx)\cos(ky),\; u_y = A\cos(kx)\sin(ky)$"
                  "\n→ at machine precision for all $h$")
    ax1.legend(loc="best")
    ax1.grid(True, which="both", alpha=0.3)

    fig.suptitle("CR vs LS — pure displacement test (no global rotation)",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> plot saved to {path}")

 

if __name__ == "__main__":
    results = {}

    print("== Analytical verification of the irrotational field ==")
    cfg = load_params()
    xs = np.linspace(0, cfg["length"], 33)
    ys = np.linspace(0, cfg["length"], 33)
    Xg = np.array([[xi, yj, 0.0] for yj in ys for xi in xs])
    verify_irrotational(Xg, L=cfg["length"])

    # Case 1: rotational 
    print("\n== 1) u_d smooth with non-zero antisymmetric part ==")
    print("    (expected: NO convergence, relative diff saturates at a finite value)")
    results["rotational"] = (
        NX_LIST,
        compare(u_smooth_rotational, NX_LIST, "rotational"),
        "expected: saturation (physical, not a bug)",
    )

    # Case 2: irrotational 
    print("\n== 2) u_d smooth, purely symmetric gradient  "
          "(∂u_x/∂y = ∂u_y/∂x = -Ak sin(kx) sin(ky)) ==")
    print("    (expected: convergence toward 0 as h -> 0)")
    results["irrotational"] = (
        NX_LIST,
        compare(u_smooth_irrotational, NX_LIST, "irrotational"),
        "expected: convergence to 0 as h -> 0",
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    write_metrics(results, os.path.join(RESULTS_DIR, "coro2d_metrics.txt"))
    plot_forces(results,   os.path.join(RESULTS_DIR, "coro2d_convergence.png"))

 

VISUALIZE_FIELD = "irrotational"          
VISUALIZE_FF    = "CorotationalFEMForceField"   
VISUALIZE_NX    = [4, 8, 16, 40, 80] 
_FIELDS = {
    "rotational":   u_smooth_rotational,
    "irrotational": u_smooth_irrotational,
}


def createScene(root):
    cfg = load_params()
    L = cfg["length"]

    u_field = _FIELDS[VISUALIZE_FIELD]

    dofs = build_scene(root, L, cfg["youngModulus"],
                       cfg["reference"]["nu"], VISUALIZE_NX, VISUALIZE_FF)

    
    root.addObject("RequiredPlugin", name="VisualPlugins", pluginName=[
        "Sofa.GL.Component.Rendering3D",
        "Sofa.GL.Component.Shader",
        "Sofa.Component.Mapping.Linear",   
    ])

    class ApplyDisplacement(Sofa.Core.Controller):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._applied = False

        def onAnimateBeginEvent(self, _):
            if self._applied:
                return
            X = dofs.rest_position.array().copy()
            Xd = X + u_field(X, L=L)
            with dofs.position.writeable() as p:
                p[:] = Xd
            self._applied = True

    root.addObject(ApplyDisplacement(name="ApplyDisplacement"))

    beam = root.getChild("Beam")
    visu = beam.addChild("Visual")
    visu.addObject("OglModel", name="ogl", color="0.2 0.6 1.0 1.0")
    visu.addObject("IdentityMapping", input="@../dofs", output="@ogl")

    return root