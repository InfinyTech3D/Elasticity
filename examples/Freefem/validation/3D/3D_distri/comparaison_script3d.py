"""
3D Beam - Distributed Load on Top Face - Comparison File

No analytical solution available for this case -> pure cross-validation
SOFA vs FreeFEM, RMS computed separately on ux, uy, uz.
"""
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from sofa_beam3d_distributed import sofaRun
from pyfreefem import FreeFemRunner


def _rms(a, b):
    """Discrete RMS, normalized by sqrt(n)."""
    return np.linalg.norm(a - b) / np.sqrt(a.size)


def _default_params_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "params_beam3d_distributed.json")


def _default_mesh_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "beam3d_tet.msh")


def _pair_by_coordinates(x_a, y_a, z_a, x_b, y_b, z_b, tol=1e-6): 
    order_a = np.lexsort((z_a, y_a, x_a))
    order_b = np.lexsort((z_b, y_b, x_b))
    if not (np.allclose(x_a[order_a], x_b[order_b], atol=tol)
            and np.allclose(y_a[order_a], y_b[order_b], atol=tol)
            and np.allclose(z_a[order_a], z_b[order_b], atol=tol)):
        raise ValueError("Node coordinates don't match between SOFA and FreeFEM meshes.")
    perm = np.empty_like(order_b)
    perm[order_b] = order_a
    return perm


if __name__ == "__main__":

    config_file = sys.argv[1] if len(sys.argv) > 1 else _default_params_path()
    with open(config_file) as f:
        cfg = json.load(f)

    q             = float(cfg["q"])
    young_modulus = float(cfg["youngModulus"])
    poisson_ratio = float(cfg["poissonRatio"])
    mesh_file     = _default_mesh_path()

    # Run FreeFEM 
    runner = FreeFemRunner("freefem_beam3d_distributed.edp")
    exports = runner.execute({
        'q': q,
        'youngModulus': young_modulus,
        'poissonRatio': poisson_ratio,
        'meshFile': mesh_file,
    })
    x_ff  = exports['xcoords']
    y_ff  = exports['ycoords']
    z_ff  = exports['zcoords']
    ux_ff = exports['ux[]']
    uy_ff = exports['uy[]']
    uz_ff = exports['uz[]']

    
    pos0_sofa, u_sofa = sofaRun(mesh_file=mesh_file, q=q,
                                 young_modulus=young_modulus,
                                 poisson_ratio=poisson_ratio)
    x_sofa, y_sofa, z_sofa = pos0_sofa[:, 0], pos0_sofa[:, 1], pos0_sofa[:, 2]
    ux_sofa, uy_sofa, uz_sofa = u_sofa[:, 0], u_sofa[:, 1], u_sofa[:, 2]

    
    perm = _pair_by_coordinates(x_sofa, y_sofa, z_sofa, x_ff, y_ff, z_ff)
    ux_ff_p = ux_ff[perm]
    uy_ff_p = uy_ff[perm]
    uz_ff_p = uz_ff[perm]

    rms_ux = _rms(ux_sofa, ux_ff_p)
    rms_uy = _rms(uy_sofa, uy_ff_p)
    rms_uz = _rms(uz_sofa, uz_ff_p)

    # --- Write results ---
    os.makedirs("results", exist_ok=True)
    with open("results/comparison_beam3d_distributed_results.txt", 'w') as f:
        header = (f"{'x':>10}  {'y':>10}  {'z':>10}  "
                  f"{'ux_sofa':>12}  {'ux_ff':>12}  "
                  f"{'uy_sofa':>12}  {'uy_ff':>12}  "
                  f"{'uz_sofa':>12}  {'uz_ff':>12}")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for x, y, z, uxs, uxf, uys, uyf, uzs, uzf in zip(
                x_sofa, y_sofa, z_sofa, ux_sofa, ux_ff_p, uy_sofa, uy_ff_p, uz_sofa, uz_ff_p):
            f.write(f"{x:10.4f}  {y:10.4f}  {z:10.4f}  "
                    f"{uxs:12.6e}  {uxf:12.6e}  "
                    f"{uys:12.6e}  {uyf:12.6e}  "
                    f"{uzs:12.6e}  {uzf:12.6e}\n")

        f.write("\n")
        f.write("RMS norms (SOFA vs FreeFEM, discrete, sqrt(n)-normalized)\n")
        f.write("-" * 55 + "\n")
        f.write(f"  RMS_ux (SOFA vs FF) = {rms_ux:.6e}\n")
        f.write(f"  RMS_uy (SOFA vs FF) = {rms_uy:.6e}\n")
        f.write(f"  RMS_uz (SOFA vs FF) = {rms_uz:.6e}\n")

    print(f"RMS_ux (SOFA vs FF) = {rms_ux:.6e}")
    print(f"RMS_uy (SOFA vs FF) = {rms_uy:.6e}")
    print(f"RMS_uz (SOFA vs FF) = {rms_uz:.6e}")
 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("3D Beam — Distributed Load on Top — SOFA vs FreeFEM (parity)", fontsize=14)

    def _parity(ax, a, b, label):
        ax.scatter(a, b, s=8, alpha=0.6)
        lims = [min(a.min(), b.min()), max(a.max(), b.max())]
        ax.plot(lims, lims, 'r--', linewidth=1)
        ax.set_xlabel(f"{label}_sofa")
        ax.set_ylabel(f"{label}_ff")
        ax.set_title(label)
        ax.set_aspect('equal')

    _parity(axes[0], ux_sofa, ux_ff_p, "ux")
    _parity(axes[1], uy_sofa, uy_ff_p, "uy")
    _parity(axes[2], uz_sofa, uz_ff_p, "uz")
    plt.tight_layout()
    fig.savefig("results/comparison_beam3d_distributed_fields.png", dpi=150)
    plt.close(fig)
 