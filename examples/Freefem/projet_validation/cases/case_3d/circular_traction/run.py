import json
import os
import numpy as np
from common import env_setup  # noqa: F401 -- corrige PATH pour stdbuf (Windows)
from pyfreefem import FreeFemRunner
from .mesh_gen import generate_beam3D_circular_tet
from .sofa_scene import sofaRun

HERE = os.path.dirname(os.path.abspath(__file__))

LABEL = "3D — Circular beam, axial traction"

MATH_DESCRIPTION = """
Poutre 3D a section circulaire (rayon r, longueur L), encastree sur la
face x=0 (label 1, Fixed), soumise a une traction axiale uniforme sur la
face x=L (label 2, Loaded) : contrainte q = F/A avec A = pi*r^2.

    -div(sigma) = 0                  dans Omega
    sigma.n = (q, 0, 0)              sur la face Loaded (x=L)
    u = 0                            sur la face Fixed (x=0)

Elasticite 3D complete :
    lambda = E*nu / ((1+nu)(1-2nu)), mu = E / (2(1+nu))

Solution analytique en champ lointain (traction uniaxiale simple) :
    ux(x,y,z) =  (q/E) * x
    uy(x,y,z) = -nu*(q/E) * y
    uz(x,y,z) = -nu*(q/E) * z

Valable seulement pour x >= exclusionFactor*r (effet Saint-Venant : la
face encastree perturbe la solution pres du bord). La comparaison SOFA
vs FreeFem, elle, reste valable sur tout le domaine.
"""


def _load_mesh_cfg():
    with open(os.path.join(HERE, "params_mesh.json")) as f:
        return json.load(f)["beam3d_circle_tet"]


def _load_phys_cfg():
    with open(os.path.join(HERE, "params_beam3d_circle_traction.json")) as f:
        return json.load(f)


def _mesh_path(meshfile):
    return os.path.join(HERE, "results", meshfile)


def _match_by_coordinates(coords_a, vals_a, coords_b, vals_b, tol=1e-6):
    """Appariement par plus proche voisin (fidele a comparaison_script3d_circle.py) :
    pour chaque noeud de coords_a, cherche le plus proche dans coords_b."""
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


def _analytical_far_field(coords, E, nu, q):
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    ux = (q / E) * x
    uy = -nu * (q / E) * y
    uz = -nu * (q / E) * z
    return np.column_stack([ux, uy, uz])


def run_case():
    mesh_cfg = _load_mesh_cfg()
    phys_cfg = _load_phys_cfg()

    length = float(mesh_cfg["length"])
    radius = float(mesh_cfg["radius"])
    E = float(phys_cfg["youngModulus"])
    nu = float(phys_cfg["poissonRatio"])
    q = float(phys_cfg["q"])
    exclusion_factor = float(phys_cfg.get("exclusionFactor", 2.0))

    # --- Maillage : genere s'il n'existe pas deja ---
    msh_path = _mesh_path(mesh_cfg["meshfile"])
    if not os.path.isfile(msh_path):
        os.makedirs(os.path.dirname(msh_path), exist_ok=True)
        generate_beam3D_circular_tet(
            length=length, radius=radius,
            mesh_size=float(mesh_cfg["mesh_size"]),
            filename=msh_path,   # chemin absolu -> ecrit exactement ici
        )

    # --- FreeFem++ ---
    # F = q*A calcule explicitement (le .edp attend "F", pas "q" : lui
    # passer "q" directement serait ignore silencieusement, cf. discussion).
    A = np.pi * radius**2
    F = q * A
    runner = FreeFemRunner(os.path.join(HERE, "freefem_beam3d_circle_traction.edp"))
    exports = runner.execute({
        "meshfile": os.path.abspath(msh_path),
        "E": E, "nu": nu, "F": F, "radius": radius, "length": length,
        "exclusionFactor": exclusion_factor,
    }, verbosity=0)
    ux_ff = np.asarray(exports["ux[]"])
    uy_ff = np.asarray(exports["uy[]"])
    uz_ff = np.asarray(exports["uz[]"])
    coords_ff = np.column_stack([exports["xcoords"], exports["ycoords"], exports["zcoords"]])

    # --- SOFA ---
    coords_sofa, u_sofa = sofaRun(mesh_file=msh_path, q=q,
                                   young_modulus=E, poisson_ratio=nu)

    # --- Appariement (plus proche voisin, par composante) ---
    tol = 1e-6 * max(length, radius)
    coords_m, ux_sofa_m, ux_ff_m = _match_by_coordinates(coords_sofa, u_sofa[:, 0], coords_ff, ux_ff, tol=tol)
    _, uy_sofa_m, uy_ff_m = _match_by_coordinates(coords_sofa, u_sofa[:, 1], coords_ff, uy_ff, tol=tol)
    _, uz_sofa_m, uz_ff_m = _match_by_coordinates(coords_sofa, u_sofa[:, 2], coords_ff, uz_ff, tol=tol)

    u_sofa_m = np.column_stack([ux_sofa_m, uy_sofa_m, uz_sofa_m])
    u_ff_m = np.column_stack([ux_ff_m, uy_ff_m, uz_ff_m])

    # --- Champ analytique en zone lointaine (calcule sur tous les noeuds
    # apparies ; le masque x >= exclusion_x est applique par compare_far_field) ---
    exclusion_x = exclusion_factor * radius
    u_far_field_ana = _analytical_far_field(coords_m, E, nu, q)

    return {
        "label": LABEL,
        "dim": 3,
        "coords_ff": coords_m,
        "u_ff": u_ff_m,
        "coords_sofa": coords_m,
        "u_sofa": u_sofa_m,
        "u_ana": None,
        "component_names": ["ux", "uy", "uz"],
        "u_far_field_ana": u_far_field_ana,
        "saint_venant_x": exclusion_x,
    }
