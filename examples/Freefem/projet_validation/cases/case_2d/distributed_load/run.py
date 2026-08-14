import json
import os
import numpy as np
from common import env_setup  # noqa: F401 -- corrige PATH pour stdbuf (Windows)
from pyfreefem import FreeFemRunner
from .sofa_scene import sofaRun, read_gmsh_2d

HERE = os.path.dirname(os.path.abspath(__file__))

MATH_DESCRIPTION = """
Poutre 2D encastree en x=0 (bord "Fixed"), soumise a un chargement
volumique uniformement distribue (0, -q) sur tout le domaine.

    -div(sigma) = (0, -q)   dans Omega
    sigma = lambda*tr(eps)*I + 2*mu*eps
    u = 0                    sur le bord Fixed (x=0)

mu = E / (2(1+nu))  (identique dans les deux modes)
    plane strain : lambda = E*nu / ((1+nu)(1-2nu))
    plane stress : lambda = E*nu / (1-nu^2)

Mode choisi a l'appel de run_case(mode=...) : "plane_strain" (defaut) ou
"plane_stress". Pas de solution analytique connue pour ce cas -> comparaison
SOFA vs FreeFem uniquement (cross-validation par RMS separement sur ux et uy).
"""

_LAMBDA_STRAIN = "lambda = E*nu / ((1+nu)(1-2nu))   [plane strain]"
_LAMBDA_STRESS = "lambda = E*nu / (1-nu^2)           [plane stress]"

_VALID_MODES = ("plane_strain", "plane_stress")


def _load_params():
    with open(os.path.join(HERE, "params_beam2d_distributed.json")) as f:
        return json.load(f)


def _mesh_path():
    return os.path.join(HERE, "beam2d_tri.msh")


def _pair_by_coordinates(x_a, y_a, x_b, y_b, tol=1e-6):
    """Retourne perm tel que (x_b[perm], y_b[perm]) correspond noeud a
    noeud a (x_a, y_a). Les deux maillages doivent partager le meme
    ensemble de noeuds (ici : meme beam2d_tri.msh des deux cotes)."""
    order_a = np.lexsort((y_a, x_a))
    order_b = np.lexsort((y_b, x_b))
    if not (np.allclose(x_a[order_a], x_b[order_b], atol=tol)
            and np.allclose(y_a[order_a], y_b[order_b], atol=tol)):
        raise ValueError("Node coordinates don't match between SOFA and FreeFEM meshes.")
    perm = np.empty_like(order_b)
    perm[order_b] = order_a
    return perm


def run_case(mode="plane_strain"):
    """
    Execute le cas 2D charge distribuee.

    mode : "plane_strain" (par defaut, template SOFA Vec3d, z=0 impose)
           ou "plane_stress" (template SOFA Vec2d, elasticite 2D genuine).
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"mode doit etre parmi {_VALID_MODES}, recu {mode!r}")

    cfg = _load_params()
    q = float(cfg["q"])
    young_modulus = float(cfg["youngModulus"])
    poisson_ratio = float(cfg["poissonRatio"])
    mesh_file = _mesh_path()

    # --- FreeFem++ ---
    runner = FreeFemRunner(os.path.join(HERE, "freefem_beam2d_distributed.edp"))
    exports = runner.execute({
        'q': q,
        'youngModulus': young_modulus,
        'poissonRatio': poisson_ratio,
        'meshFile': mesh_file,
        'planeStrain': 1 if mode == "plane_strain" else 0,
    }, verbosity=0)
    x_ff = np.asarray(exports['xcoords'])
    y_ff = np.asarray(exports['ycoords'])
    ux_ff = np.asarray(exports['ux[]'])
    uy_ff = np.asarray(exports['uy[]'])

    # --- SOFA (meme E, nu physiques ; seul le template Vec3d/Vec2d change) ---
    pos0_sofa, u_sofa = sofaRun(mesh_file=mesh_file, q=q,
                                 young_modulus=young_modulus,
                                 poisson_ratio=poisson_ratio,
                                 mode=mode)
    x_sofa, y_sofa = pos0_sofa[:, 0], pos0_sofa[:, 1]

    # --- Reordonner FreeFem sur l'ordre des noeuds SOFA (meme maillage) ---
    perm = _pair_by_coordinates(x_sofa, y_sofa, x_ff, y_ff)
    u_ff = np.column_stack([ux_ff[perm], uy_ff[perm]])

    # Connectivite du maillage (utile pour le trace 2D : tripcolor)
    _, triangles, _ = read_gmsh_2d(mesh_file)

    lambda_formula = _LAMBDA_STRAIN if mode == "plane_strain" else _LAMBDA_STRESS

    return {
        "label": f"2D — Distributed load ({mode.replace('_', ' ')})",
        "mode": mode,
        "dim": 2,
        "coords_ff": np.column_stack([x_sofa, y_sofa]),   # meme ordre que u_ff apres pairing
        "u_ff": u_ff,
        "coords_sofa": np.column_stack([x_sofa, y_sofa]),
        "u_sofa": u_sofa,
        "u_ana": None,
        "triangles": triangles,   # connectivite, utilisee par plot_fields_2d
        "component_names": ["ux", "uy"],
        "lambda_formula": lambda_formula,
    }
