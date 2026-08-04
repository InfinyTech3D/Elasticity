# =============================================================================
# comparaison_script3d_circle_bending.py
# 3D circular beam, transverse (bending) load: SOFA vs FreeFEM
#
# Same structure as comparaison_script3d_circle.py (axial traction case):
# generate the mesh, run FreeFEM, run SOFA, match nodes by coordinates,
# compute RMS. No analytical reference here -- the Saint-Venant traction
# formulas do not apply to bending; an Euler-Bernoulli/Timoshenko reference
# would need to be added separately if/when needed.
# =============================================================================

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from pyfreefem import FreeFemRunner
import sofa_beam3d_circle_bending as sofa_case

RESULTS_DIR = "results"


def _default_edp_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "freefem_beam3d_circle_bending.edp")


def match_by_coordinates(coords_a, vals_a, coords_b, vals_b, tol=1e-6): 
    matched_a, matched_b, matched_coords = [], [], []
    for i, c in enumerate(coords_a):
        d = np.linalg.norm(coords_b - c, axis=1)
        j = np.argmin(d)
        if d[j] > tol:
            continue
        matched_coords.append(c)
        matched_a.append(vals_a[i])
        matched_b.append(vals_b[j])
    return np.array(matched_coords), np.array(matched_a), np.array(matched_b)


def rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    with open("params.json") as f:
        mesh_cfg = json.load(f)["beam3d_circle_tet"]
    with open("params_beam3d_circle_bending.json") as f:
        phys_cfg = json.load(f)

    length = mesh_cfg["length"]
    radius = mesh_cfg["radius"]
    E = phys_cfg["youngModulus"]
    nu = phys_cfg["poissonRatio"]
    q = phys_cfg["q"]
 
    msh_path = os.path.join(RESULTS_DIR, mesh_cfg["meshfile"])
    if not os.path.exists(msh_path):
        raise FileNotFoundError(
            f"{msh_path} not found -- run beam3d_circular_tet.py first "
            f"(and check_mesh_conformity.py to validate it)."
        )
    print(f"Mesh: {msh_path}")
 
    runner = FreeFemRunner(_default_edp_path())
    exports = runner.execute({
        "meshfile": os.path.abspath(msh_path),
        "E": E, "nu": nu, "F": q * (np.pi * radius**2), "radius": radius, "length": length,
    })
    ux_ff = exports["ux[]"] if "ux[]" in exports else exports["ux"]
    uy_ff = exports["uy[]"] if "uy[]" in exports else exports["uy"]
    uz_ff = exports["uz[]"] if "uz[]" in exports else exports["uz"]
    xcoords = exports["xcoords"]
    ycoords = exports["ycoords"]
    zcoords = exports["zcoords"]
    coords_ff = np.column_stack([xcoords, ycoords, zcoords])
 
    coords_sofa, u_sofa = sofa_case.sofaRun(
        mesh_file=msh_path, q=q, young_modulus=E, poisson_ratio=nu,
    )
    ux_sofa, uy_sofa, uz_sofa = u_sofa[:, 0], u_sofa[:, 1], u_sofa[:, 2] 

    tol = 1e-6 * max(length, radius)
    coords_m, ux_sofa_m, ux_ff_m = match_by_coordinates(
        coords_sofa, ux_sofa, coords_ff, ux_ff, tol=tol)
    _, uy_sofa_m, uy_ff_m = match_by_coordinates(
        coords_sofa, uy_sofa, coords_ff, uy_ff, tol=tol)
    _, uz_sofa_m, uz_ff_m = match_by_coordinates(
        coords_sofa, uz_sofa, coords_ff, uz_ff, tol=tol)

    rms_ux = rms(ux_sofa_m, ux_ff_m)
    rms_uy = rms(uy_sofa_m, uy_ff_m)
    rms_uz = rms(uz_sofa_m, uz_ff_m)

    print(f"RMS_ux (SOFA vs FF) = {rms_ux:.6e}")
    print(f"RMS_uy (SOFA vs FF) = {rms_uy:.6e}")
    print(f"RMS_uz (SOFA vs FF) = {rms_uz:.6e}")
 
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "comparison_beam3d_circle_bending_results.txt")
    with open(out_path, "w") as f:
        f.write("x y z ux_sofa ux_ff uy_sofa uy_ff uz_sofa uz_ff\n")
        f.write("-" * 100 + "\n")
        for i in range(len(coords_m)):
            f.write(f"{coords_m[i,0]:10.4f} {coords_m[i,1]:10.4f} {coords_m[i,2]:10.4f} "
                     f"{ux_sofa_m[i]:14.6e} {ux_ff_m[i]:14.6e} "
                     f"{uy_sofa_m[i]:14.6e} {uy_ff_m[i]:14.6e} "
                     f"{uz_sofa_m[i]:14.6e} {uz_ff_m[i]:14.6e}\n")
        f.write("\nRMS norms (SOFA vs FreeFEM)\n")
        f.write(f"  RMS_ux = {rms_ux:.6e}\n  RMS_uy = {rms_uy:.6e}\n  RMS_uz = {rms_uz:.6e}\n")
    print("\nWrote:", out_path)
 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (a, b, name) in zip(axes, [(ux_sofa_m, ux_ff_m, "ux"),
                                        (uy_sofa_m, uy_ff_m, "uy"),
                                        (uz_sofa_m, uz_ff_m, "uz")]):
        ax.scatter(a, b, s=15, alpha=0.6)
        lims = [min(a.min(), b.min()), max(a.max(), b.max())]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlabel(f"{name}_sofa")
        ax.set_ylabel(f"{name}_ff")
        ax.set_title(name)
    fig.suptitle("3D Circular Beam — Bending — SOFA vs FreeFEM (parity)")
    fig_path = os.path.join(RESULTS_DIR, "comparison_beam3d_circle_bending_fields.png")
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    print("Wrote:", fig_path)


if __name__ == "__main__":
    main()