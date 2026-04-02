"""
2D Beam Simulation - Deformed Under Gravity - Comparison File
"""
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from gmsh_generate_beam2D import generate_beam2D
from sofa_beam2d import sofaRun
from pyfreefem import FreeFemRunner


def _l2(a, b):
    return np.linalg.norm(a - b) / a.size


def _write_freefem_results(path, x, y, u_x, u_y):
    x_final = x + u_x
    y_final = y + u_y
    with open(path, 'w') as f:
        f.write(f"{'x_initial':>12} {'y_initial':>12} {'x_final':>12} {'y_final':>12} {'u_x':>12} {'u_y':>12}\n")
        f.write("-" * 80 + "\n")
        for xi, yi, xf, yf, uxi, uyi in zip(x, y, x_final, y_final, u_x, u_y):
            f.write(f"{xi:12.6f} {yi:12.6f} {xf:12.6f} {yf:12.6f} {uxi:12.6f} {uyi:12.6f}\n")


if __name__ == "__main__":

    ## --- Read parameters from JSON file ---
    config_file = sys.argv[1] if len(sys.argv) > 1 else "params.json"
    with open(config_file) as f:
        cfg = json.load(f)

    length          = float(cfg["length"])
    height          = float(cfg["height"])
    nx              = int(cfg["nx"])
    ny              = int(cfg["ny"])
    force           = float(cfg["force"])
    rho             = float(cfg["rho"])
    gravity         = float(cfg["gravity"])
    young_modulus   = float(cfg["youngModulus"])
    poisson_ratio   = float(cfg["poissonRatio"])
    mesh_filename   = cfg["meshfile"]

    # --- Run Gmsh ---
    path_before_gmsh = os.environ.get('PATH', '')
    msh_path, x_positions, y_positions = generate_beam2D(
        length=length,
        height=height,
        nx=nx,
        ny=ny,
        filename=mesh_filename
    )

    for formulation2d in ["planeStrain", "planeStress"]:
        print(f"Running comparison simulations for {formulation2d}")
        # --- Run FreeFEM ---
        # gmsh.initialize() corrupts PATH, breaking pyfreefem's stdbuf call
        os.environ['PATH'] = path_before_gmsh
        runner = FreeFemRunner("freefem_beam2d.edp")
        results = runner.execute({
            'youngModulus': young_modulus,
            'poissonRatio': poisson_ratio,
            'rhoMat': rho,
            'gravity': gravity,
            'meshFile': os.path.abspath(msh_path).replace('\\', '/'),
            'formulation' : formulation2d
        })
        x_ff = results['xcoords']
        y_ff = results['ycoords']
        dofs = results['uu[]']  # interleaved [uu_0, vv_0, uu_1, vv_1, ...]
        u_x_ff = dofs[0::2]
        u_y_ff = dofs[1::2]

        os.makedirs("results", exist_ok=True)
        _write_freefem_results(os.path.join("results", f"freefem_{formulation2d}_results.txt")
                            , x_ff, y_ff, u_x_ff, u_y_ff)

        ## --- Run SOFA ---
        pos_initial, u_sofa = sofaRun(
            height=height,
            young_modulus=young_modulus,
            poisson_ratio=poisson_ratio,
            rho=rho,
            gravity=gravity,
            mesh_file=os.path.abspath(msh_path).replace('\\', '/'),
            formulation=formulation2d
        )
        x_sofa   = pos_initial[:, 0]
        y_sofa   = pos_initial[:, 1]
        u_x_sofa = u_sofa[:, 0]
        u_y_sofa = u_sofa[:, 1]

        ## --- Compare Results ---
        err_ux = u_x_sofa - u_x_ff
        err_uy = u_y_sofa - u_y_ff

        with open(f"results/comparison_{formulation2d}_results.txt", 'w') as f:
            header = (f"{'x':>12}  {'y':>12}  {'ux_ff':>12}  {'ux_sofa':>12}"
                      f"  {'uy_ff':>12}  {'uy_sofa':>12}  {'err_ux':>12}  {'err_uy':>12}")
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            for vals in zip(x_ff, y_ff, u_x_ff, u_x_sofa, u_y_ff, u_y_sofa, err_ux, err_uy):
                f.write("  ".join(f"{v:12.6f}" for v in vals) + "\n")
            f.write("\n")
            f.write("L2 error norms\n")
            f.write("-" * 40 + "\n")
            f.write(f"  ||ux_sofa - ux_ff||_2 = {_l2(u_x_sofa, u_x_ff):.6e}\n")
            f.write(f"  ||uy_sofa - uy_ff||_2 = {_l2(u_y_sofa, u_y_ff):.6e}\n")

        ## --- Plots ---
        triang = mtri.Triangulation(x_ff, y_ff)

        # Shared colour ranges so SOFA and FreeFEM panels are directly comparable
        vmin_ux = min(u_x_ff.min(), u_x_sofa.min())
        vmax_ux = max(u_x_ff.max(), u_x_sofa.max())
        vmin_uy = min(u_y_ff.min(), u_y_sofa.min())
        vmax_uy = max(u_y_ff.max(), u_y_sofa.max())

        # Fig 1 — displacement field: FreeFEM (top row) vs SOFA (bottom row)
        fig, axes = plt.subplots(2, 2, figsize=(14, 6), constrained_layout=True)
        fig.suptitle("Displacement field — FreeFEM (top) vs SOFA (bottom)")
        for row, (label, ux, uy) in enumerate([
            ("FreeFEM", u_x_ff,   u_y_ff),
            ("SOFA",    u_x_sofa, u_y_sofa),
        ]):
            tcf = axes[row, 0].tricontourf(triang, ux, levels=20, vmin=vmin_ux, vmax=vmax_ux)
            axes[row, 0].set_title(f"{label} — $u_x$")
            axes[row, 0].set_aspect('equal')
            fig.colorbar(tcf, ax=axes[row, 0])

            tcf = axes[row, 1].tricontourf(triang, uy, levels=20, vmin=vmin_uy, vmax=vmax_uy)
            axes[row, 1].set_title(f"{label} — $u_y$")
            axes[row, 1].set_aspect('equal')
            fig.colorbar(tcf, ax=axes[row, 1])
        fig.savefig(f"results/plot_{formulation2d}_displacement_field.png", dpi=150)
        plt.close(fig)

        # Fig 2 — deformed beams overlaid
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.scatter(x_ff   + u_x_ff,   y_ff   + u_y_ff,   s=4, label="FreeFEM", alpha=0.7)
        ax.scatter(x_sofa + u_x_sofa, y_sofa + u_y_sofa, s=4, label="SOFA",    alpha=0.7, marker='x')
        ax.set_aspect('equal')
        ax.set_title("Deformed beam — FreeFEM vs SOFA")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        fig.savefig(f"results/plot_{formulation2d}_deformed_beams.png", dpi=150)
        plt.close(fig)

        # Figs 3 & 4 — nodal error in x and y
        for comp, err in [("ux", err_ux), ("uy", err_uy)]:
            fig, ax = plt.subplots(figsize=(14, 4))
            tcf = ax.tricontourf(triang, err, levels=20, cmap='RdBu_r')
            fig.colorbar(tcf, ax=ax, label=f"$u_{comp[1]}$ error (SOFA − FreeFEM)")
            ax.set_title(f"Nodal error — $u_{comp[1]}$")
            ax.set_aspect('equal')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.savefig(f"results/plot_error_{formulation2d}_{comp}.png", dpi=150)
            plt.close(fig)

