import json
import os
import numpy as np
from common import env_setup  # noqa: F401 -- corrige PATH pour stdbuf (Windows)
from pyfreefem import FreeFemRunner
from .sofa_scene import sofaRun, L, H

HERE = os.path.dirname(os.path.abspath(__file__))

_VALID_MODES = ("plane_strain", "plane_stress")

LABEL_TEMPLATE = "2D — 3-point bending ({mode})"

MATH_DESCRIPTION = """
Poutre 2D simplement appuyee (appui simple a gauche : ux=uy=0, appui a
rouleau a droite : uy=0), soumise a une charge ponctuelle P au milieu de
la face superieure (flexion 3 points).

Mode choisi a l'appel de run_case(mode=...) : "plane_strain" (SOFA Vec3d,
FreeFem lambda=E*nu/((1+nu)(1-2nu))) ou "plane_stress" (SOFA Vec2d,
FreeFem lambda=E*nu/(1-nu^2)).

Pas de solution analytique exacte 2D disponible -> comparaison SOFA vs
FreeFem par RMS (global + par composante), sur tout le domaine.

Une verification Euler-Bernoulli de la fleche a mi-portee est aussi
fournie, a TITRE INDICATIF SEULEMENT :
    w_EB = -P*L^3 / (48*E_eff*I),  I = H^3/12
Ce n'est PAS une cible de validation stricte : la poutre est courte et
epaisse (L/H = 5), donc les effets de cisaillement (non captes par
Euler-Bernoulli) et les concentrations de contrainte pres de l'appui/du
point de charge (non captees par une theorie de poutre 1D) creent un
ecart attendu avec la solution FEM 2D. Un ecart SOFA/EB n'est donc pas
un signe d'erreur.
"""


def _load_params():
    with open(os.path.join(HERE, "params_beam3pt.json")) as f:
        return json.load(f)


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


def _eb_midspan_deflection(P, E, nu, mode):
    """Fleche Euler-Bernoulli a mi-portee, indicative uniquement (voir
    MATH_DESCRIPTION). Meme formule que comparaison_script2d_3pt.py."""
    I = H**3 / 12.0
    E_eff = E / (1.0 - nu**2) if mode == "plane_strain" else E
    return -P * L**3 / (48.0 * E_eff * I)


def run_case(mode="plane_strain"):
    """
    Execute le cas 2D flexion 3 points.

    mode : "plane_strain" (par defaut, template SOFA Vec3d, z=0 verrouille)
           ou "plane_stress" (template SOFA Vec2d, elasticite 2D genuine).
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"mode doit etre parmi {_VALID_MODES}, recu {mode!r}")

    plane_type = "strain" if mode == "plane_strain" else "stress"

    cfg = _load_params()
    P = float(cfg["P"])
    young_modulus = float(cfg["youngModulus"])
    poisson_ratio = float(cfg["poissonRatio"])
    mesh_file = os.path.join(HERE, "beam2d_tri.msh")

    # --- FreeFem++ ---
    runner = FreeFemRunner(os.path.join(HERE, "freefem_beam3pt.edp"))
    exports = runner.execute({
        'P': P,
        'youngModulus': young_modulus,
        'poissonRatio': poisson_ratio,
        'meshFile': mesh_file,
        'planeStress': 1.0 if mode == "plane_stress" else 0.0,
    }, verbosity=0)
    x_ff = np.asarray(exports['xcoords'])
    y_ff = np.asarray(exports['ycoords'])
    ux_ff = np.asarray(exports['uxOut'])
    uy_ff = np.asarray(exports['uyOut'])

    # --- SOFA ---
    pos0_sofa, u_sofa, triangles = sofaRun(mesh_file=mesh_file, P=P,
                                            young_modulus=young_modulus,
                                            poisson_ratio=poisson_ratio,
                                            plane_type=plane_type)
    x_sofa, y_sofa = pos0_sofa[:, 0], pos0_sofa[:, 1]

    # --- Reordonner FreeFem sur l'ordre des noeuds SOFA (meme maillage) ---
    perm = _pair_by_coordinates(x_sofa, y_sofa, x_ff, y_ff)
    u_ff = np.column_stack([ux_ff[perm], uy_ff[perm]])

    # --- Verification Euler-Bernoulli (indicative) a mi-portee ---
    mid_idx = np.argmin(np.abs(x_sofa - L / 2) + np.abs(y_sofa - H))
    w_sofa = u_sofa[mid_idx, 1]
    w_ff = u_ff[mid_idx, 1]
    w_eb = _eb_midspan_deflection(P, young_modulus, poisson_ratio, mode)

    return {
        "label": LABEL_TEMPLATE.format(mode=mode.replace("_", " ")),
        "mode": mode,
        "dim": 2,
        "coords_ff": np.column_stack([x_sofa, y_sofa]),   # meme ordre que u_ff apres pairing
        "u_ff": u_ff,
        "coords_sofa": np.column_stack([x_sofa, y_sofa]),
        "u_sofa": u_sofa,
        "u_ana": None,   # pas de solution 2D exacte, cf MATH_DESCRIPTION
        "triangles": triangles,
        "component_names": ["ux", "uy"],
        # Specifique a ce cas : verification Euler-Bernoulli indicative
        "eb_check": {
            "w_sofa": w_sofa,
            "w_ff": w_ff,
            "w_eb": w_eb,
            "ratio_sofa_eb": w_sofa / w_eb,
        },
    }
