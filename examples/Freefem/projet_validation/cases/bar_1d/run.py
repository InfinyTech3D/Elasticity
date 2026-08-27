"""
Cas 1D — Barre encastrée-libre, charge répartie.

Toute la logique (paramètres, FreeFem, SOFA, solution analytique) est
encapsulée ici. Le notebook n'appelle que run_case().
"""
import json
import os

from common import env_setup  # noqa: F401 -- corrige PATH pour stdbuf (Windows)
from pyfreefem import FreeFemRunner

from .sofa_scene import sofaRun

HERE = os.path.dirname(os.path.abspath(__file__))

LABEL = "Barre 1D — charge répartie"

MATH_DESCRIPTION = """
Équation forte : E·u''(x) + q = 0, avec u(0)=0 (encastrement) et u'(L)=0 (extrémité libre).
Solution analytique : u(x) = (q/E) · (L·x - x²/2)
"""


def _load_params():
    with open(os.path.join(HERE, "params.json")) as f:
        return json.load(f)


def _u_exact(x, q, E, L):
    return (q / E) * (L * x - x**2 / 2.0)


def run_case():
    """
    Exécute FreeFem et SOFA sur le cas 1D distribué, calcule la solution
    analytique, et renvoie un dict de résultats prêt pour common.metrics
    et common.plotting.
    """
    cfg = _load_params()
    length, nx = float(cfg["length"]), int(cfg["nx"])
    q = float(cfg["q"])
    young_modulus = float(cfg["youngModulus"])
    poisson_ratio = float(cfg["poissonRatio"])

    # --- FreeFem++ ---
    runner = FreeFemRunner(os.path.join(HERE, "freefem_bar_distributed.edp"))
    exports = runner.execute({
        "youngModulus": young_modulus,
        "q": q,
        "nx": nx,
        "length": length,
    }, verbosity=0)
    x_ff, u_ff = exports["xcoords"], exports["u[]"]

    # --- SOFA ---
    x_sofa, u_sofa = sofaRun(
        length=length, q=q,
        young_modulus=young_modulus, poisson_ratio=poisson_ratio, nx=nx,
    )

    # --- Analytique ---
    u_ana = _u_exact(x_ff, q, young_modulus, length)

    return {
        "label": LABEL,
        "params": cfg,
        "x_ff": x_ff, "u_ff": u_ff,
        "x_sofa": x_sofa, "u_sofa": u_sofa,
        "u_ana": u_ana,
    }
