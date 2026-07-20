import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from sofa_beam3pt import sofaRun, L, H
from pyfreefem import FreeFemRunner


def _rms(a, b):
    return np.linalg.norm(a - b) / np.sqrt(a.size)


def _default_params_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "params_beam3pt.json")


def _default_mesh_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "beam2d_tri.msh")


def _pair_by_coordinates(x_a, y_a, x_b, y_b, tol=1e-6):
    order_a = np.lexsort((y_a, x_a))
    order_b = np.lexsort((y_b, x_b))
    if not (np.allclose(x_a[order_a], x_b[order_b], atol=tol)
            and np.allclose(y_a[order_a], y_b[order_b], atol=tol)):
        raise ValueError("Node coordinates don't match between SOFA and FreeFEM meshes.")
    perm = np.empty_like(order_b)
    perm[order_b] = order_a
    return perm


def eb_midspan_deflection(P, E, nu, plane="strain"):
    """Indicative Euler-Bernoulli deflection for a simply-supported beam
    under a central point load: w_max = -P*L^3/(48*E*I).
    NOT a strict validation target  a gap vs plane-strain/stress """
    I = H**3 / 12.0  
    E_eff = E / (1.0 - nu**2) if plane == "strain" else E
    return -P * L**3 / (48.0 * E_eff * I)


if __name__ == "__main__":

    config_file = sys.argv[1] if len(sys.argv) > 1 else _default_params_path()
    with open(config_file) as f:
        cfg = json.load(f)

    P             = float(cfg["P"])
    young_modulus = float(cfg["youngModulus"])
    poisson_ratio = float(cfg["poissonRatio"])
    plane_type    = cfg.get("planeType", "strain")
    mesh_file     = _default_mesh_path()

    assert plane_type in ("strain", "stress"), "planeType must be 'strain' or 'stress'"

    # --- Run FreeFEM ---
    runner = FreeFemRunner("freefem_beam3pt.edp")
    exports = runner.execute({
        'P': P,
        'youngModulus': young_modulus,
        'poissonRatio': poisson_ratio,
        'meshFile': mesh_file,
        'planeStress': 1.0 if plane_type == "stress" else 0.0,
    })
    x_ff  = exports['xcoords']
    y_ff  = exports['ycoords']
    ux_ff = exports['uxOut']
    uy_ff = exports['uyOut']

    # --- Run SOFA ---
    pos0_sofa, u_sofa, triangles = sofaRun(mesh_file=mesh_file, P=P,
                                            young_modulus=young_modulus,
                                            poisson_ratio=poisson_ratio,
                                            plane_type=plane_type)
    x_sofa, y_sofa   = pos0_sofa[:, 0], pos0_sofa[:, 1]
    ux_sofa, uy_sofa = u_sofa[:, 0], u_sofa[:, 1]

    perm = _pair_by_coordinates(x_sofa, y_sofa, x_ff, y_ff)
    ux_ff_p = ux_ff[perm]
    uy_ff_p = uy_ff[perm]

    rms_ux = _rms(ux_sofa, ux_ff_p)
    rms_uy = _rms(uy_sofa, uy_ff_p)

    mid_idx = np.argmin(np.abs(x_sofa - L / 2) + np.abs(y_sofa - H))
    w_sofa = uy_sofa[mid_idx]
    w_eb = eb_midspan_deflection(P, young_modulus, poisson_ratio, plane=plane_type)

    # --- Write results ---
    os.makedirs("results", exist_ok=True)
    results_path = f"results/comparison_beam3pt_{plane_type}_results.txt"
    with open(results_path, 'w') as f:
        header = f"{'x':>10}  {'y':>10}  {'ux_sofa':>12}  {'ux_ff':>12}  {'uy_sofa':>12}  {'uy_ff':>12}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for x, y, uxs, uxf, uys, uyf in zip(x_sofa, y_sofa, ux_sofa, ux_ff_p, uy_sofa, uy_ff_p):
            f.write(f"{x:10.4f}  {y:10.4f}  {uxs:12.6e}  {uxf:12.6e}  {uys:12.6e}  {uyf:12.6e}\n")

        f.write("\n")
        f.write(f"RMS norms (SOFA vs FreeFEM, plane {plane_type}, discrete, sqrt(n)-normalized)\n")
        f.write("-" * 55 + "\n")
        f.write(f"  RMS_ux (SOFA vs FF) = {rms_ux:.6e}\n")
        f.write(f"  RMS_uy (SOFA vs FF) = {rms_uy:.6e}\n")
        f.write("\n")
        f.write("Indicative Euler-Bernoulli check (midspan deflection only)\n")
        f.write("-" * 55 + "\n")
        f.write(f"  w_sofa (midspan) = {w_sofa:.6e}\n")
        f.write(f"  w_ff   (midspan) = {uy_ff_p[mid_idx]:.6e}\n")
        f.write(f"  w_EB   (analytical, indicative only) = {w_eb:.6e}\n")
        f.write(f"  ratio SOFA/EB = {w_sofa / w_eb:.4f}  "
                f"(a gap here is EXPECTED - not a validation failure )")

    print(f"[plane_type={plane_type}]")
    print(f"RMS_ux (SOFA vs FF) = {rms_ux:.6e}")
    print(f"RMS_uy (SOFA vs FF) = {rms_uy:.6e}")
    print(f"w_sofa = {w_sofa:.6e}  |  w_EB (indicative) = {w_eb:.6e}  "
          f"|  ratio = {w_sofa / w_eb:.4f}")

    
    triang = mtri.Triangulation(x_sofa, y_sofa, triangles)

    def _plot_field(ax, vals, title, cmap='viridis'):
        tc = ax.tripcolor(triang, vals, shading='gouraud', cmap=cmap)
        plt.colorbar(tc, ax=ax)
        ax.set_title(title)
        ax.set_aspect('equal')

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(f"2D Beam — 3-Point Bending — SOFA vs FreeFEM (plane {plane_type})", fontsize=14)
    _plot_field(axes[0, 0], ux_sofa,           'SOFA : u_x')
    _plot_field(axes[0, 1], ux_ff_p,           'FreeFEM : u_x')
    _plot_field(axes[0, 2], ux_sofa - ux_ff_p, 'Diff u_x', 'RdBu')
    _plot_field(axes[1, 0], uy_sofa,           'SOFA : u_y')
    _plot_field(axes[1, 1], uy_ff_p,           'FreeFEM : u_y')
    _plot_field(axes[1, 2], uy_sofa - uy_ff_p, 'Diff u_y', 'RdBu')
    plt.tight_layout()
    fig.savefig(f"results/comparison_beam3pt_{plane_type}_fields.png", dpi=150)
    plt.close(fig)