import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from sofa_beam3d_threept import sofaRun, L, H, W
from pyfreefem import FreeFemRunner


def _rms(a, b):
    """Discrete RMS, normalized by sqrt(n)."""
    return np.linalg.norm(a - b) / np.sqrt(a.size)


def _default_params_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "params_beam3d_threept.json")


def _default_mesh_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "beam3d_tet.msh")


def _to_freefem_path(path):
    """FreeFEM string literals on Windows can choke on backslashes; use forward slashes."""
    return path.replace(os.sep, "/")


def _pair_by_coordinates(x_a, y_a, z_a, x_b, y_b, z_b, tol=1e-6, snap=1e-6):
    """
    Pair nodes between two independently-loaded copies of "the same" mesh
    by (rounded) coordinates.

    The raw mesh file carries sub-tolerance floating noise from mesh
    generation (e.g. x=0.8999999999997362 instead of exactly 0.9), and
    this noise differs slightly across nodes that are conceptually at the
    same coordinate. SOFA keeps that raw noise; FreeFEM's writer appears
    to clean/round it on output. Sorting on the RAW values is therefore
    unsafe: a primary sort key (x) that is only "almost tied" can order
    differently between the two sources, which then contaminates the
    secondary/tertiary keys (y, z) and silently mispairs otherwise
    identical points. Snapping to a grid well above the noise floor but
    well below the real geometric spacing fixes this.
    """
    x_a, y_a, z_a = map(np.asarray, (x_a, y_a, z_a))
    x_b, y_b, z_b = map(np.asarray, (x_b, y_b, z_b))

    print(f"[diag] n_sofa={x_a.size}  n_freefem={x_b.size}")
    print(f"[diag] SOFA    bbox: x[{x_a.min():.6f},{x_a.max():.6f}] "
          f"y[{y_a.min():.6f},{y_a.max():.6f}] z[{z_a.min():.6f},{z_a.max():.6f}]")
    print(f"[diag] FreeFEM bbox: x[{x_b.min():.6f},{x_b.max():.6f}] "
          f"y[{y_b.min():.6f},{y_b.max():.6f}] z[{z_b.min():.6f},{z_b.max():.6f}]")

    if x_a.size != x_b.size:
        raise ValueError(
            f"Node COUNT mismatch: SOFA has {x_a.size} nodes, "
            f"FreeFEM has {x_b.size} nodes. Check that both loaded the "
            f"exact same mesh file (same path, no stale results.txt)."
        )

    # Snap to a grid well above float noise (~1e-13) but well below the
    # real mesh spacing, so ties sort consistently on both sides.
    def snap_(v):
        return np.round(v / snap) * snap

    xs_a, ys_a, zs_a = snap_(x_a), snap_(y_a), snap_(z_a)
    xs_b, ys_b, zs_b = snap_(x_b), snap_(y_b), snap_(z_b)

    order_a = np.lexsort((zs_a, ys_a, xs_a))
    order_b = np.lexsort((zs_b, ys_b, xs_b))

    da = np.stack([x_a[order_a], y_a[order_a], z_a[order_a]], axis=1)
    db = np.stack([x_b[order_b], y_b[order_b], z_b[order_b]], axis=1)
    diff = np.linalg.norm(da - db, axis=1)
    worst = np.argsort(diff)[::-1][:10]

    print(f"[diag] max sorted-coordinate discrepancy = {diff.max():.6e} (tol={tol:.1e})")
    if diff.max() >= tol:
        print("[diag] worst offenders (sorted rank, SOFA xyz, FreeFEM xyz, dist):")
        for i in worst:
            print(f"    rank={i:4d}  sofa={tuple(np.round(da[i], 8))}  "
                  f"ff={tuple(np.round(db[i], 8))}  dist={diff[i]:.3e}")

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

    P             = float(cfg["P"])
    young_modulus = float(cfg["youngModulus"])
    poisson_ratio = float(cfg["poissonRatio"])
    mesh_file     = _default_mesh_path()

    os.makedirs("results", exist_ok=True)
    ff_out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "results", "freefem_beam3d_threept_raw.txt")

    #  ====== Run FreeFEM (writes plain-text results to ff_out_path) =====
    runner = FreeFemRunner("freefem_beam3d_threept.edp")
    runner.execute({
        'P': P,
        'youngModulus': young_modulus,
        'poissonRatio': poisson_ratio,
        'meshFile': _to_freefem_path(mesh_file),
        'outFile': _to_freefem_path(ff_out_path),
    })

    if not os.path.isfile(ff_out_path):
        raise RuntimeError(
            f"FreeFEM did not produce the expected output file: {ff_out_path}\n"
            f"Check the FreeFEM console output above for compile/runtime errors."
        )

    raw = np.loadtxt(ff_out_path)
    x_ff, y_ff, z_ff = raw[:, 0], raw[:, 1], raw[:, 2]
    ux_ff, uy_ff, uz_ff = raw[:, 3], raw[:, 4], raw[:, 5]

    #  ========== Run SOFA ===========
    pos0_sofa, u_sofa = sofaRun(mesh_file=mesh_file, P=P,
                                 young_modulus=young_modulus,
                                 poisson_ratio=poisson_ratio)
    x_sofa, y_sofa, z_sofa    = pos0_sofa[:, 0], pos0_sofa[:, 1], pos0_sofa[:, 2]
    ux_sofa, uy_sofa, uz_sofa = u_sofa[:, 0], u_sofa[:, 1], u_sofa[:, 2]

    perm = _pair_by_coordinates(x_sofa, y_sofa, z_sofa, x_ff, y_ff, z_ff)
    ux_ff_p = ux_ff[perm]
    uy_ff_p = uy_ff[perm]
    uz_ff_p = uz_ff[perm]

    rms_ux = _rms(ux_sofa, ux_ff_p)
    rms_uy = _rms(uy_sofa, uy_ff_p)
    rms_uz = _rms(uz_sofa, uz_ff_p)

    # Midspan deflection: node on the loaded top edge closest to (L/2, H, W/2)
    mid_idx = np.argmin(np.abs(x_sofa - L / 2) + np.abs(y_sofa - H) + np.abs(z_sofa - W / 2))
    w_sofa = uy_sofa[mid_idx]
    w_ff   = uy_ff_p[mid_idx]

    with open("results/comparison_beam3d_threept_results.txt", 'w') as f:
        header = (f"{'x':>10}  {'y':>10}  {'z':>10}  {'ux_sofa':>12}  {'ux_ff':>12}  "
                  f"{'uy_sofa':>12}  {'uy_ff':>12}  {'uz_sofa':>12}  {'uz_ff':>12}")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for x, y, z, uxs, uxf, uys, uyf, uzs, uzf in zip(
                x_sofa, y_sofa, z_sofa, ux_sofa, ux_ff_p, uy_sofa, uy_ff_p, uz_sofa, uz_ff_p):
            f.write(f"{x:10.4f}  {y:10.4f}  {z:10.4f}  {uxs:12.6e}  {uxf:12.6e}  "
                    f"{uys:12.6e}  {uyf:12.6e}  {uzs:12.6e}  {uzf:12.6e}\n")

        f.write("\n")
        f.write("RMS norms (SOFA vs FreeFEM, discrete, sqrt(n)-normalized)\n")
        f.write("-" * 55 + "\n")
        f.write(f"  RMS_ux (SOFA vs FF) = {rms_ux:.6e}\n")
        f.write(f"  RMS_uy (SOFA vs FF) = {rms_uy:.6e}\n")
        f.write(f"  RMS_uz (SOFA vs FF) = {rms_uz:.6e}\n")
        f.write("\n")
        f.write(f"  w_sofa (near midspan) = {w_sofa:.6e}\n")
        f.write(f"  w_ff   (near midspan) = {w_ff:.6e}\n")

    print(f"RMS_ux (SOFA vs FF) = {rms_ux:.6e}")
    print(f"RMS_uy (SOFA vs FF) = {rms_uy:.6e}")
    print(f"RMS_uz (SOFA vs FF) = {rms_uz:.6e}")
    print(f"w_sofa = {w_sofa:.6e}  |  w_ff = {w_ff:.6e}")

    # ---- Visualization: parity plots (u_sofa vs u_ff), one per component ----
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

    fig.suptitle("3D Beam — 3-Point Bending — SOFA vs FreeFEM (parity)", fontsize=14)
    plt.tight_layout()
    fig.savefig("results/comparison_beam3d_threept_fields.png", dpi=150)
    plt.close(fig)