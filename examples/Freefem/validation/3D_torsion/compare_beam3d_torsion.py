import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from sofa_beam3d_torsion import sofaRun, MESH_DIR, DEFAULT_MESH_FILENAME
from pyfreefem import FreeFemRunner


def _rms(a, b):
    return np.linalg.norm(a - b) / np.sqrt(a.size)


def _rel_rms(u_ref, u_test):
    denom = np.linalg.norm(u_ref)
    return float(np.linalg.norm(u_test - u_ref) / denom) if denom > 0 else float("nan")


def _default_params_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "params_beam3d_torsion.json")


def _default_mesh_path():
    return os.path.join(MESH_DIR, DEFAULT_MESH_FILENAME)


def _to_freefem_path(path):
    return path.replace(os.sep, "/")


def _rewrite_freefem_output_in_place(raw, out_path): 
    header = (f"{'x':>12}  {'y':>12}  {'z':>12}  "
              f"{'ux':>14}  {'uy':>14}  {'uz':>14}")
    with open(out_path, 'w') as f:
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for x, y, z, ux, uy, uz in raw:
            f.write(f"{x:12.6f}  {y:12.6f}  {z:12.6f}  "
                    f"{ux:+14.6e}  {uy:+14.6e}  {uz:+14.6e}\n")


def _pair_by_coordinates(x_a, y_a, z_a, x_b, y_b, z_b, tol=1e-6, snap=1e-6):
    
    x_a, y_a, z_a = map(np.asarray, (x_a, y_a, z_a))
    x_b, y_b, z_b = map(np.asarray, (x_b, y_b, z_b))

    print(f"[diag] n_sofa={x_a.size}  n_freefem={x_b.size}")

    if x_a.size != x_b.size:
        raise ValueError(
            f"Node COUNT mismatch: SOFA has {x_a.size} nodes, "
            f"FreeFEM has {x_b.size} nodes. "
        )

    def snap_(v):
        return np.round(v / snap) * snap

    xs_a, ys_a, zs_a = snap_(x_a), snap_(y_a), snap_(z_a)
    xs_b, ys_b, zs_b = snap_(x_b), snap_(y_b), snap_(z_b)

    order_a = np.lexsort((zs_a, ys_a, xs_a))
    order_b = np.lexsort((zs_b, ys_b, xs_b))

    da = np.stack([x_a[order_a], y_a[order_a], z_a[order_a]], axis=1)
    db = np.stack([x_b[order_b], y_b[order_b], z_b[order_b]], axis=1)
    diff = np.linalg.norm(da - db, axis=1)

    print(f"[diag] max sorted-coordinate discrepancy = {diff.max():.6e} (tol={tol:.1e})")
    if diff.max() >= tol:
        raise ValueError(
            "Node coordinates don't match between SOFA and FreeFEM meshes."
        )

    perm = np.empty_like(order_b)
    perm[order_b] = order_a
    return perm


def _analytical_displacement(x0, torque, radius, young_modulus, poisson_ratio, yc, zc):
   
    G = young_modulus / (2.0 * (1.0 + poisson_ratio))
    J = np.pi * radius ** 4 / 2.0
    theta_prime = torque / (G * J)

    x, y, z = x0[:, 0], x0[:, 1], x0[:, 2]
    ux = np.zeros_like(x)
    uy = -theta_prime * x * (z - zc)
    uz = theta_prime * x * (y - yc)
    return np.column_stack([ux, uy, uz]), theta_prime


if __name__ == "__main__":

    config_file = sys.argv[1] if len(sys.argv) > 1 else _default_params_path()
    with open(config_file) as f:
        cfg = json.load(f)

    T             = float(cfg["T"])
    radius        = float(cfg["radius"])
    young_modulus = float(cfg["youngModulus"])
    poisson_ratio = float(cfg["poissonRatio"])
    mesh_file     = _default_mesh_path()

    os.makedirs("results", exist_ok=True)
    ff_out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "results", "freefem_beam3d_torsion_raw.txt")

    runner = FreeFemRunner("freefem_beam3d_torsion.edp")
    runner.execute({
        'T': T,
        'radius': radius,
        'youngModulus': young_modulus,
        'poissonRatio': poisson_ratio,
        'meshFile': _to_freefem_path(mesh_file),
        'outFile': _to_freefem_path(ff_out_path),
    })

    if not os.path.isfile(ff_out_path):
        raise RuntimeError( 
        )

    raw = np.loadtxt(ff_out_path)
    _rewrite_freefem_output_in_place(raw, ff_out_path)

    x_ff, y_ff, z_ff = raw[:, 0], raw[:, 1], raw[:, 2]
    ux_ff, uy_ff, uz_ff = raw[:, 3], raw[:, 4], raw[:, 5]

    #  ========== Run SOFA ===========
    pos0_sofa, u_sofa = sofaRun(mesh_file=mesh_file, T=T, radius=radius,
                                 young_modulus=young_modulus,
                                 poisson_ratio=poisson_ratio)
    x_sofa, y_sofa, z_sofa    = pos0_sofa[:, 0], pos0_sofa[:, 1], pos0_sofa[:, 2]
    ux_sofa, uy_sofa, uz_sofa = u_sofa[:, 0], u_sofa[:, 1], u_sofa[:, 2]

    perm = _pair_by_coordinates(x_sofa, y_sofa, z_sofa, x_ff, y_ff, z_ff)
    ux_ff_p = ux_ff[perm]
    uy_ff_p = uy_ff[perm]
    uz_ff_p = uz_ff[perm]
    u_ff_p = np.column_stack([ux_ff_p, uy_ff_p, uz_ff_p])
    u_sofa_full = np.column_stack([ux_sofa, uy_sofa, uz_sofa])

    yc = 0.5 * (y_sofa.min() + y_sofa.max())
    zc = 0.5 * (z_sofa.min() + z_sofa.max())
    u_ana, theta_prime = _analytical_displacement(
        pos0_sofa, T, radius, young_modulus, poisson_ratio, yc, zc
    )

    rms_ux = _rms(ux_sofa, ux_ff_p)
    rms_uy = _rms(uy_sofa, uy_ff_p)
    rms_uz = _rms(uz_sofa, uz_ff_p)

    rel_sofa_ff  = _rel_rms(u_sofa_full, u_ff_p)
    rel_sofa_ana = _rel_rms(u_ana, u_sofa_full)
    rel_ff_ana   = _rel_rms(u_ana, u_ff_p)

    with open("results/comparison_beam3d_torsion_results.txt", 'w') as f:
        header = (f"{'x':>10}  {'y':>10}  {'z':>10}  {'ux_sofa':>14}  {'ux_ff':>14}  "
                  f"{'uy_sofa':>14}  {'uy_ff':>14}  {'uz_sofa':>14}  {'uz_ff':>14}")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for x, y, z, uxs, uxf, uys, uyf, uzs, uzf in zip(
                x_sofa, y_sofa, z_sofa, ux_sofa, ux_ff_p, uy_sofa, uy_ff_p, uz_sofa, uz_ff_p):
            f.write(f"{x:10.4f}  {y:10.4f}  {z:10.4f}  {uxs:+14.6e}  {uxf:+14.6e}  "
                    f"{uys:+14.6e}  {uyf:+14.6e}  {uzs:+14.6e}  {uzf:+14.6e}\n")

        f.write("\n")
        f.write(f"theta' (analytical) = {theta_prime:.6g} rad/m\n")
        f.write("RMS norms\n")
        f.write("-" * 40 + "\n")
        f.write(f"  RMS_ux (SOFA vs FF) = {rms_ux:.6e}\n")
        f.write(f"  RMS_uy (SOFA vs FF) = {rms_uy:.6e}\n")
        f.write(f"  RMS_uz (SOFA vs FF) = {rms_uz:.6e}\n")
        f.write(f"  Relatif SOFA vs FF         = {rel_sofa_ff:.3%}\n")
        f.write(f"  Relatif SOFA vs Analytique = {rel_sofa_ana:.3%}\n")
        f.write(f"  Relatif FF   vs Analytique = {rel_ff_ana:.3%}\n")

    print("=" * 70)
    print("Validation : poutre 3D section circulaire, torsion pure")
    print("=" * 70)
    print(f"theta' (analytical) = {theta_prime:.6g} rad/m")
    print(f"RMS_ux (SOFA vs FF) = {rms_ux:.6e}")
    print(f"RMS_uy (SOFA vs FF) = {rms_uy:.6e}")
    print(f"RMS_uz (SOFA vs FF) = {rms_uz:.6e}")
    print("-" * 70)
    print(f"Relatif SOFA vs FF         = {rel_sofa_ff:.3%}")
    print(f"Relatif SOFA vs Analytique = {rel_sofa_ana:.3%}")
    print(f"Relatif FF   vs Analytique = {rel_ff_ana:.3%}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (u_s, u_f, label) in zip(axes, [
            (ux_sofa, ux_ff_p, 'ux'), (uy_sofa, uy_ff_p, 'uy'), (uz_sofa, uz_ff_p, 'uz')]):
        ax.scatter(u_s, u_f, s=15, alpha=0.8)
        lo = min(u_s.min(), u_f.min())
        hi = max(u_s.max(), u_f.max())
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1)
        ax.set_xlabel(f'{label}_sofa')
        ax.set_ylabel(f'{label}_ff')
        ax.set_title(label)

    fig.suptitle("3D Beam - Torsion- SOFA vs FreeFEM ", fontsize=14)
    plt.tight_layout()
    fig.savefig("results/comparison_beam3d_torsion_fields.png", dpi=150)
    plt.close(fig)