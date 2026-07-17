import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from sofa_beam2d_compression import sofaRun, read_gmsh_2d
from pyfreefem import FreeFemRunner

# Saint-Venant boundary layer: exact clamping at x=0 perturbs the uniform
# compression solution over a distance on the order of the beam height H.
# Analytical comparison is only meaningful for x >= SAINT_VENANT_FACTOR * H.
SAINT_VENANT_FACTOR = 2.0


def _rms(a, b):
    return np.linalg.norm(a - b) / np.sqrt(a.size)


def _default_params_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "params_beam2d_compression.json")


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


def analytical_far_field(x, y, q, E, nu):
    
    ux = -q * (1.0 - nu**2) / E * x
    uy = q * nu * (1.0 + nu) / E * y
    return ux, uy


if __name__ == "__main__":

    config_file = sys.argv[1] if len(sys.argv) > 1 else _default_params_path()
    with open(config_file) as f:
        cfg = json.load(f)

    q             = float(cfg["q"])
    young_modulus = float(cfg["youngModulus"])
    poisson_ratio = float(cfg["poissonRatio"])
    mesh_file     = _default_mesh_path()

    # --- Run FreeFEM ---
    runner = FreeFemRunner("freefem_beam2d_compression.edp")
    exports = runner.execute({
        'q': q,
        'youngModulus': young_modulus,
        'poissonRatio': poisson_ratio,
        'meshFile': mesh_file,
    })
    x_ff  = exports['xcoords']
    y_ff  = exports['ycoords']
    ux_ff = exports['ux[]']
    uy_ff = exports['uy[]']

    # --- Run SOFA ---
    pos0_sofa, u_sofa = sofaRun(mesh_file=mesh_file, q=q,
                                 young_modulus=young_modulus,
                                 poisson_ratio=poisson_ratio)
    x_sofa, y_sofa   = pos0_sofa[:, 0], pos0_sofa[:, 1]
    ux_sofa, uy_sofa = u_sofa[:, 0], u_sofa[:, 1]

    perm = _pair_by_coordinates(x_sofa, y_sofa, x_ff, y_ff)
    ux_ff_p = ux_ff[perm]
    uy_ff_p = uy_ff[perm]

    rms_ux = _rms(ux_sofa, ux_ff_p)
    rms_uy = _rms(uy_sofa, uy_ff_p)

    
    H = y_sofa.max() - y_sofa.min()
    far_mask = x_sofa >= SAINT_VENANT_FACTOR * H
    ux_an, uy_an = analytical_far_field(x_sofa, y_sofa, q, young_modulus, poisson_ratio)
    if far_mask.any():
        rms_ux_an = _rms(ux_sofa[far_mask], ux_an[far_mask])
        rms_uy_an = _rms(uy_sofa[far_mask] - uy_sofa[far_mask].mean(),
                          uy_an[far_mask] - uy_an[far_mask].mean())
    else:
        rms_ux_an = rms_uy_an = float('nan')

    # Write results 
    os.makedirs("results", exist_ok=True)
    with open("results/comparison_beam2d_compression_results.txt", 'w') as f:
        header = f"{'x':>10}  {'y':>10}  {'ux_sofa':>12}  {'ux_ff':>12}  {'uy_sofa':>12}  {'uy_ff':>12}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for x, y, uxs, uxf, uys, uyf in zip(x_sofa, y_sofa, ux_sofa, ux_ff_p, uy_sofa, uy_ff_p):
            f.write(f"{x:10.4f}  {y:10.4f}  {uxs:12.6e}  {uxf:12.6e}  {uys:12.6e}  {uyf:12.6e}\n")

        f.write("\n")
        f.write("RMS norms (SOFA vs FreeFEM, discrete, sqrt(n)-normalized)\n")
        f.write("-" * 55 + "\n")
        f.write(f"  RMS_ux (SOFA vs FF) = {rms_ux:.6e}\n")
        f.write(f"  RMS_uy (SOFA vs FF) = {rms_uy:.6e}\n")
        f.write("\n")
        f.write(f"Far-field analytical check (x >= {SAINT_VENANT_FACTOR:.1f}*H = {SAINT_VENANT_FACTOR*H:.3f})\n")
        f.write("-" * 55 + "\n")
        f.write(f"  RMS_ux (SOFA vs analytical) = {rms_ux_an:.6e}\n")
        f.write(f"  RMS_uy (SOFA vs analytical, mean-shifted) = {rms_uy_an:.6e}\n")

    print(f"RMS_ux (SOFA vs FF) = {rms_ux:.6e}")
    print(f"RMS_uy (SOFA vs FF) = {rms_uy:.6e}")
    print(f"RMS_ux (SOFA vs analytical, far field) = {rms_ux_an:.6e}")
    print(f"RMS_uy (SOFA vs analytical, far field, mean-shifted) = {rms_uy_an:.6e}")


    _, triangles, _, _ = read_gmsh_2d(mesh_file)
    triang = mtri.Triangulation(x_sofa, y_sofa, triangles)

    def _plot_field(ax, vals, title, cmap='viridis'):
        tc = ax.tripcolor(triang, vals, shading='gouraud', cmap=cmap)
        plt.colorbar(tc, ax=ax)
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.axvline(SAINT_VENANT_FACTOR * H, color='k', ls='--', lw=0.8)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("2D Beam — Compression — SOFA vs FreeFEM (plane strain)", fontsize=14)
    _plot_field(axes[0, 0], ux_sofa,           'SOFA : u_x')
    _plot_field(axes[0, 1], ux_ff_p,           'FreeFEM : u_x')
    _plot_field(axes[0, 2], ux_sofa - ux_ff_p, 'Diff u_x', 'RdBu')
    _plot_field(axes[1, 0], uy_sofa,           'SOFA : u_y')
    _plot_field(axes[1, 1], uy_ff_p,           'FreeFEM : u_y')
    _plot_field(axes[1, 2], uy_sofa - uy_ff_p, 'Diff u_y', 'RdBu')
    plt.tight_layout()
    fig.savefig("results/comparison_beam2d_compression_fields.png", dpi=150)
    plt.close(fig)