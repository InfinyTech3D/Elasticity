import json
import os
import numpy as np
from common import env_setup  # noqa: F401 -- corrige PATH pour stdbuf (Windows)
from pyfreefem import FreeFemRunner
from .sofa_scene import sofaRun, read_gmsh_2d

HERE = os.path.dirname(os.path.abspath(__file__))

_VALID_MODES = ("plane_strain", "plane_stress")

LABEL_TEMPLATE = "2D — Compression (traction sur bord droit, {mode})"

# Couche limite de Saint-Venant : le clampage exact en x=0 perturbe la
# solution de compression uniforme sur une distance de l'ordre de la
# hauteur H de la poutre. La comparaison analytique n'a de sens que pour
# x >= SAINT_VENANT_FACTOR * H.
SAINT_VENANT_FACTOR = 2.0

MATH_DESCRIPTION = f"""
Poutre 2D encastree en x=0 (bord "Fixed"), soumise a une traction uniforme
(-q, 0) sur le bord droit (x=L, label 3) -> compression uniaxiale.

    -div(sigma) = 0                  dans Omega
    sigma.n = (-q, 0)                sur le bord Right (x=L)
    u = 0                            sur le bord Fixed (x=0)

Mode choisi a l'appel de run_case(mode=...) : "plane_strain" (defaut) ou
"plane_stress" -- change a la fois le solveur (SOFA : template Vec3d/z=0
verrouille vs Vec2d ; FreeFem : formule de lambda) ET la solution
analytique en champ lointain (les deux regimes ont une reponse elastique
differente sous chargement uniaxial) :

    plane strain : ux = -q(1-nu^2)/E * x   uy = q*nu(1+nu)/E * y
    plane stress : ux = -q/E * x           uy = q*nu/E * y

Valable seulement pour x >= {SAINT_VENANT_FACTOR:.1f}*H (effet Saint-Venant :
le clampage exact perturbe la solution pres du bord encastre). La
comparaison SOFA vs FreeFem, elle, reste valable sur tout le domaine.
"""


def _load_params():
    with open(os.path.join(HERE, "params_beam2d_compression.json")) as f:
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


def _analytical_far_field(x, y, q, E, nu, mode):
    if mode == "plane_strain":
        ux = -q * (1.0 - nu**2) / E * x
        uy = q * nu * (1.0 + nu) / E * y
    else:
        ux = -q / E * x
        uy = q * nu / E * y
    return ux, uy


def run_case(mode="plane_strain"):
    """
    Execute le cas 2D compression.

    mode : "plane_strain" (par defaut, template SOFA Vec3d, z=0 verrouille)
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
    runner = FreeFemRunner(os.path.join(HERE, "freefem_beam2d_compression.edp"))
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

    # --- SOFA ---
    pos0_sofa, u_sofa = sofaRun(mesh_file=mesh_file, q=q,
                                 young_modulus=young_modulus,
                                 poisson_ratio=poisson_ratio,
                                 mode=mode)
    x_sofa, y_sofa = pos0_sofa[:, 0], pos0_sofa[:, 1]

    # --- Reordonner FreeFem sur l'ordre des noeuds SOFA (meme maillage) ---
    perm = _pair_by_coordinates(x_sofa, y_sofa, x_ff, y_ff)
    u_ff = np.column_stack([ux_ff[perm], uy_ff[perm]])

    # --- Champ analytique en zone lointaine (Saint-Venant), formule selon le mode ---
    H = y_sofa.max() - y_sofa.min()
    saint_venant_x = SAINT_VENANT_FACTOR * H
    ux_an, uy_an = _analytical_far_field(x_sofa, y_sofa, q, young_modulus, poisson_ratio, mode)
    u_far_field_ana = np.column_stack([ux_an, uy_an])

    # Connectivite du maillage (utile pour le trace 2D : tripcolor)
    _, triangles, _, _ = read_gmsh_2d(mesh_file)

    return {
        "label": LABEL_TEMPLATE.format(mode=mode.replace("_", " ")),
        "mode": mode,
        "dim": 2,
        "coords_ff": np.column_stack([x_sofa, y_sofa]),   # meme ordre que u_ff apres pairing
        "u_ff": u_ff,
        "coords_sofa": np.column_stack([x_sofa, y_sofa]),
        "u_sofa": u_sofa,
        "u_ana": None,             # pas de solution exacte sur tout le domaine (effet Saint-Venant)
        "triangles": triangles,
        "component_names": ["ux", "uy"],
        # Specifique a ce cas : verification analytique en champ lointain
        "u_far_field_ana": u_far_field_ana,
        "saint_venant_x": saint_venant_x,
        "vline_x": saint_venant_x,   # utilise par plot_fields_2d pour tracer le seuil
    }
