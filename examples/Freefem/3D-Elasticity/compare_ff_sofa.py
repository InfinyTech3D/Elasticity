"""
3D Beam Simulation - Comparison FreeFEM vs SOFA
"""
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gmsh_generate_beam3D import generate_beam3D
from sofa_beam3d import sofaRun
from pyfreefem import FreeFemRunner


def _l2(a, b):
    return np.linalg.norm(a - b) / a.size


def _write_freefem_results(path, x, y, z, u_x, u_y, u_z):
    with open(path, 'w') as f:
        f.write(f"{'x':>12} {'y':>12} {'z':>12} {'ux':>12} {'uy':>12} {'uz':>12}\n")
        f.write("-" * 80 + "\n")
        for xi, yi, zi, uxi, uyi, uzi in zip(x, y, z, u_x, u_y, u_z):
            f.write(f"{xi:12.6f} {yi:12.6f} {zi:12.6f} {uxi:12.6f} {uyi:12.6f} {uzi:12.6f}\n")


if __name__ == "__main__":

    ## --- Read parameters from JSON file ---
    config_file = sys.argv[1] if len(sys.argv) > 1 else "params.json"
    with open(config_file) as f:
        cfg = json.load(f)

    length        = float(cfg["length"])
    height        = float(cfg["height"])
    width         = float(cfg["width"])
    nx            = int(cfg["nx"])
    ny            = int(cfg["ny"])
    nz            = int(cfg["nz"])
    rho           = float(cfg["rho"])
    gravity       = float(cfg["gravity"])
    young_modulus = float(cfg["youngModulus"])
    poisson_ratio = float(cfg["poissonRatio"])
    mesh_filename = cfg.get("meshfile", "beam3d.msh")

    # --- Run Gmsh ---
    path_before_gmsh = os.environ.get('PATH', '')
    msh_path, x_positions, y_positions, z_positions = generate_beam3D(
        length=length,
        height=height,
        width=width,
        nx=nx,
        ny=ny,
        nz=nz,
        filename=mesh_filename
    )

    # --- Run FreeFEM ---
    os.environ['PATH'] = path_before_gmsh
    runner = FreeFemRunner("3d-beam.edp")
    results = runner.execute({
        'youngModulus' : young_modulus,
        'poissonRatio' : poisson_ratio,
        'rhoMat'       : rho,
        'grav'         : gravity,
        'meshFile'     : os.path.abspath(msh_path).replace('\\', '/'),
    })
    x_ff  = results['coordX']
    y_ff  = results['coordY']
    z_ff  = results['coordZ']
    u_x_ff = results['dispX']
    u_y_ff = results['dispY']
    u_z_ff = results['dispZ']

    os.makedirs("results", exist_ok=True)
    _write_freefem_results(
        os.path.join("results", "freefem_3d_results.txt"),
        x_ff, y_ff, z_ff, u_x_ff, u_y_ff, u_z_ff
    )

    # --- Run SOFA ---
    pos_initial, u_sofa = sofaRun(
        height        = height,
        width         = width,
        young_modulus = young_modulus,
        poisson_ratio = poisson_ratio,
        rho           = rho,
        gravity       = gravity,
        mesh_file     = os.path.abspath(msh_path).replace('\\', '/'),
    )
    x_sofa   = pos_initial[:, 0]
    y_sofa   = pos_initial[:, 1]
    z_sofa   = pos_initial[:, 2]
    u_x_sofa = u_sofa[:, 0]
    u_y_sofa = u_sofa[:, 1]
    u_z_sofa = u_sofa[:, 2]

    ## --- Compare Results ---
    err_ux = u_x_sofa - u_x_ff
    err_uy = u_y_sofa - u_y_ff
    err_uz = u_z_sofa - u_z_ff

    with open("results/comparison_3d_results.txt", 'w') as f:
        header = (f"{'x':>12}  {'y':>12}  {'z':>12}  "
                  f"{'ux_ff':>12}  {'ux_sofa':>12}  "
                  f"{'uy_ff':>12}  {'uy_sofa':>12}  "
                  f"{'uz_ff':>12}  {'uz_sofa':>12}  "
                  f"{'err_ux':>12}  {'err_uy':>12}  {'err_uz':>12}")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for vals in zip(x_ff, y_ff, z_ff,
                        u_x_ff, u_x_sofa,
                        u_y_ff, u_y_sofa,
                        u_z_ff, u_z_sofa,
                        err_ux, err_uy, err_uz):
            f.write("  ".join(f"{v:12.6f}" for v in vals) + "\n")
        f.write("\n")
        f.write("L2 error norms\n")
        f.write("-" * 40 + "\n")
        f.write(f"  ||ux_sofa - ux_ff||_2 = {_l2(u_x_sofa, u_x_ff):.6e}\n")
        f.write(f"  ||uy_sofa - uy_ff||_2 = {_l2(u_y_sofa, u_y_ff):.6e}\n")
        f.write(f"  ||uz_sofa - uz_ff||_2 = {_l2(u_z_sofa, u_z_ff):.6e}\n")

    ## --- Plots ---
    

    tol = min(height, width) * 0.1
    mask_ff   = (np.abs(y_ff   - height/2) < tol) & (np.abs(z_ff   - width/2) < tol)
    mask_sofa = (np.abs(y_sofa - height/2) < tol) & (np.abs(z_sofa - width/2) < tol)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    fig.suptitle("Displacement along beam axis — FreeFEM vs SOFA")

    for ax, comp, uff, usofa, label in zip(
        axes,
        ['ux', 'uy', 'uz'],
        [u_x_ff,   u_y_ff,   u_z_ff],
        [u_x_sofa, u_y_sofa, u_z_sofa],
        ['$u_x$',  '$u_y$',  '$u_z$'],
    ):
        ax.scatter(x_ff[mask_ff],     uff[mask_ff],     s=10, label="FreeFEM", alpha=0.8)
        ax.scatter(x_sofa[mask_sofa], usofa[mask_sofa], s=10, label="SOFA",    alpha=0.8, marker='x')
        ax.set_xlabel("x (m)")
        ax.set_ylabel(f"{label} (m)")
        ax.set_title(f"{label} along beam axis")
        ax.legend()

    fig.savefig("results/plot_3d_displacement_axis.png", dpi=150)
    plt.close(fig)

    # Fig 2 — deformed shape (projection XZ — plan de flexion)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(x_ff   + u_x_ff,   z_ff   + u_z_ff,   s=4, label="FreeFEM", alpha=0.7)
    ax.scatter(x_sofa + u_x_sofa, z_sofa + u_z_sofa, s=4, label="SOFA",    alpha=0.7, marker='x')
    ax.set_aspect('equal')
    ax.set_title("Deformed beam (XZ projection) — FreeFEM vs SOFA")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.legend()
    fig.savefig("results/plot_3d_deformed_beam_XZ.png", dpi=150)
    plt.close(fig)

    