"""TEST 2 — CR on combined field x = R·X + u_d(X)

Verifies that CorotationalFEMForceField correctly isolates u_d
even in the presence of a global rigid-body rotation R.

Frame-invariance check: f_CR(R·X + u_d) ≈ f_CR(X + u_d)

Three cases:
  A) x = X + u_d          (no rotation   — reference, from TEST 1)
  B) x = R·X + u_d        (rotation + u_d in reference frame)
  C) x = R·(X + u_d)      (rotation applied to full deformed config)
     → encadrant's suggestion: tests full objectivity of CR

For a truly frame-invariant formulation:
  ‖f_CR(B) − f_CR(A)‖ / ‖f_CR(A)‖  << 1
  ‖f_CR(C) − f_CR(A)‖ / ‖f_CR(A)‖  << 1

Run:  python test_cr_rotation_plus_ud.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from beam import load_params, element_quad, RESULTS_DIR

ANGLE_DEG = 45.0   # rotation angle


def Rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


def u_irrotational(X, amp=0.01, L=1.0):
    """Irrotational field — same as TEST 1 case 2.
    u_x = A sin(kx) cos(ky),  u_y = A cos(kx) sin(ky),  k = π/L
    Zero local rotation analytically.
    """
    k = np.pi / L
    x, y = X[:, 0], X[:, 1]
    ux = amp * np.sin(k * x) * np.cos(k * y)
    uy = amp * np.cos(k * x) * np.sin(k * y)
    return np.stack([ux, uy, np.zeros_like(x)], axis=1)


def build_scene(root, L, E, nu, nx):
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
        dofs = beam.addObject("MechanicalObject", name="dofs",
                              template="Vec3d",
                              position="@../Grid/grid.position")
        beam.addObject("UniformMass", name="mass", massDensity=1.0)
        element_quad.add_topology(beam)
        beam.addObject("CorotationalFEMForceField", name="FEM",
                       template="Vec3d", youngModulus=E,
                       poissonRatio=nu, topology="@topology")
    return dofs


def run_with_positions(Xd, L, E, nu, nx):
    """Build scene, impose positions Xd, animate once, return forces."""
    root = Sofa.Core.Node("root")
    dofs = build_scene(root, L, E, nu, nx)
    Sofa.Simulation.init(root)
    with dofs.position.writeable() as p:
        p[:] = Xd
    Sofa.Simulation.animate(root, 1e-8)
    f = dofs.force.array().copy()
    Sofa.Simulation.unload(root)
    return f


def compare_cases(nx, L, E, nu):
    R = Rz(np.radians(ANGLE_DEG))

    # Build reference grid once
    root_tmp = Sofa.Core.Node("root_tmp")
    dofs_tmp = build_scene(root_tmp, L, E, nu, nx)
    Sofa.Simulation.init(root_tmp)
    X = dofs_tmp.rest_position.array().copy()
    Sofa.Simulation.unload(root_tmp)

    ud = u_irrotational(X, L=L)

    # Case A: x = X + u_d  (reference, no rotation)
    Xa = X + ud
    fa = run_with_positions(Xa, L, E, nu, nx)

    # Case B: x = R·X + u_d  (rotation on X only)
    Xb = (R @ X.T).T + ud
    fb = run_with_positions(Xb, L, E, nu, nx)

    # Case C: x = R·(X + u_d)  (rotation on full deformed config)
    # encadrant's suggestion — tests full objectivity
    Xc = (R @ (X + ud).T).T
    fc = run_with_positions(Xc, L, E, nu, nx)

    ref = np.max(np.linalg.norm(fa, axis=1))

    fb_ref = (R.T @ fb.T).T
    fc_ref = (R.T @ fc.T).T

    diff_B = np.max(np.linalg.norm(fb_ref - fa, axis=1)) / ref
    diff_C = np.max(np.linalg.norm(fc_ref - fa, axis=1)) / ref

    return ref, diff_B, diff_C


def write_metrics(results, path):
    with open(path, "w") as fout:
        fout.write(f"CR frame-invariance test — angle = {ANGLE_DEG} deg\n")
        fout.write("=" * 74 + "\n\n")
        fout.write("  Case A : x = X + u_d          (reference)\n")
        fout.write("  Case B : x = R·X + u_d        (rotation on X)\n")
        fout.write("  Case C : x = R·(X + u_d)      (encadrant suggestion)\n\n")
        for nx, (ref, dB, dC) in results.items():
            fout.write(f"  nx={nx:3d}   |f_A| = {ref:.4e}"
                       f"   |f_B-f_A|/|f_A| = {dB:.4e}"
                       f"   |f_C-f_A|/|f_A| = {dC:.4e}\n")
    print(f"  -> metrics saved to {path}")


def plot_results(results, path):
    nx_list = list(results.keys())
    hs      = [1.0 / nx for nx in nx_list]
    diffs_B = [results[nx][1] for nx in nx_list]
    diffs_C = [results[nx][2] for nx in nx_list]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(hs, diffs_B, marker="s", linestyle="--", color="tab:orange",
              label=r"Case B: $x = R\cdot X + u_d$")
    ax.loglog(hs, diffs_C, marker="^", linestyle="-",  color="tab:purple",
              label=r"Case C: $x = R\cdot(X + u_d)$")

    floor = np.finfo(float).eps
    ax.axhline(y=floor, color="gray", linestyle=":", linewidth=1.0)
    ax.text(max(hs) * 0.6, floor * 3,
            r"$\varepsilon_{\rm machine}$", fontsize=8, color="gray")

    ax.invert_xaxis()
    ax.set_xlabel(r"$h = 1/n_x$  (finer mesh $\rightarrow$)")
    ax.set_ylabel(r"$\|f_{CR}(\cdot) - f_{CR}(A)\| \;/\; \|f_{CR}(A)\|$")
    ax.set_title(f"CR frame-invariance — rotation {ANGLE_DEG}°\n"
                 r"Both curves must stay $\ll 1$ (ideally $\approx \varepsilon_{\rm machine}$)")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> plot saved to {path}")


if __name__ == "__main__":
    cfg = load_params()
    L, E = cfg["length"], cfg["youngModulus"]
    nu   = cfg["reference"]["nu"]

    NX_LIST = [40, 80]
    results = {}

    print(f"== CR frame-invariance test  (angle = {ANGLE_DEG}°) ==\n")
    print(f"  {'nx':>4}   {'|f_A|':>12}   "
          f"{'|f_B-f_A|/|f_A|':>18}   {'|f_C-f_A|/|f_A|':>18}")
    print("  " + "-" * 60)

    for nx in NX_LIST:
        ref, dB, dC = compare_cases(nx, L, E, nu)
        results[nx] = (ref, dB, dC)
        print(f"  {nx:4d}   {ref:12.4e}   {dB:18.4e}   {dC:18.4e}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    write_metrics(results, os.path.join(RESULTS_DIR, "cr_frame_invariance_metrics.txt"))
    plot_results(results,  os.path.join(RESULTS_DIR, "cr_frame_invariance.png"))