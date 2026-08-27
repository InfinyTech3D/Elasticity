import json
import os
import numpy as np
from common import env_setup  # noqa: F401 -- corrige PATH pour stdbuf (Windows)
from pyfreefem import FreeFemRunner
from .sofa_scene import sofaRun

HERE = os.path.dirname(os.path.abspath(__file__))

LABEL = "3D — Distributed load (top face)"

MATH_DESCRIPTION = """
Poutre 3D encastree sur la face x=x_min (label 1), soumise a une charge
repartie uniforme (0, -q, 0) sur la face superieure y=y_max (label 4).

    -div(sigma) = 0                  dans Omega
    sigma.n = (0, -q, 0)             sur la face superieure (y=y_max)
    u = 0                            sur la face encastree (x=x_min)

Elasticite 3D complete (pas de notion plane strain/stress en 3D) :
    lambda = E*nu / ((1+nu)(1-2nu))
    mu     = E / (2(1+nu))

Pas de solution analytique connue pour ce cas -> comparaison SOFA vs
FreeFem uniquement (cross-validation par RMS separement sur ux, uy, uz).
"""


def _load_params():
    with open(os.path.join(HERE, "params_beam3d_distributed.json")) as f:
        return json.load(f)


def _mesh_path():
    return os.path.join(HERE, "beam3d_tet.msh")


def _pair_by_coordinates(x_a, y_a, z_a, x_b, y_b, z_b, tol=1e-6):
    """Retourne perm tel que (x_b[perm], y_b[perm], z_b[perm]) correspond
    noeud a noeud a (x_a, y_a, z_a). Les deux maillages doivent partager le
    meme ensemble de noeuds (ici : meme beam3d_tet.msh des deux cotes)."""
    order_a = np.lexsort((z_a, y_a, x_a))
    order_b = np.lexsort((z_b, y_b, x_b))
    if not (np.allclose(x_a[order_a], x_b[order_b], atol=tol)
            and np.allclose(y_a[order_a], y_b[order_b], atol=tol)
            and np.allclose(z_a[order_a], z_b[order_b], atol=tol)):
        raise ValueError("Node coordinates don't match between SOFA and FreeFEM meshes.")
    perm = np.empty_like(order_b)
    perm[order_b] = order_a
    return perm


def run_case():
    cfg = _load_params()
    q = float(cfg["q"])
    young_modulus = float(cfg["youngModulus"])
    poisson_ratio = float(cfg["poissonRatio"])
    mesh_file = _mesh_path()

    # --- FreeFem++ ---
    runner = FreeFemRunner(os.path.join(HERE, "freefem_beam3d_distributed.edp"))
    exports = runner.execute({
        'q': q,
        'youngModulus': young_modulus,
        'poissonRatio': poisson_ratio,
        'meshFile': mesh_file,
    }, verbosity=0)
    x_ff = np.asarray(exports['xcoords'])
    y_ff = np.asarray(exports['ycoords'])
    z_ff = np.asarray(exports['zcoords'])
    ux_ff = np.asarray(exports['ux[]'])
    uy_ff = np.asarray(exports['uy[]'])
    uz_ff = np.asarray(exports['uz[]'])

    # --- SOFA ---
    pos0_sofa, u_sofa = sofaRun(mesh_file=mesh_file, q=q,
                                 young_modulus=young_modulus,
                                 poisson_ratio=poisson_ratio)
    x_sofa, y_sofa, z_sofa = pos0_sofa[:, 0], pos0_sofa[:, 1], pos0_sofa[:, 2]

    # --- Reordonner FreeFem sur l'ordre des noeuds SOFA (meme maillage) ---
    perm = _pair_by_coordinates(x_sofa, y_sofa, z_sofa, x_ff, y_ff, z_ff)
    u_ff = np.column_stack([ux_ff[perm], uy_ff[perm], uz_ff[perm]])

    return {
        "label": LABEL,
        "dim": 3,
        "coords_ff": np.column_stack([x_sofa, y_sofa, z_sofa]),   # meme ordre que u_ff apres pairing
        "u_ff": u_ff,
        "coords_sofa": np.column_stack([x_sofa, y_sofa, z_sofa]),
        "u_sofa": u_sofa,
        "u_ana": None,
        "component_names": ["ux", "uy", "uz"],
    }
