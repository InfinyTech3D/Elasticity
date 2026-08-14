import json
import os
import numpy as np
from common import env_setup  # noqa: F401 -- corrige PATH pour stdbuf (Windows)
from pyfreefem import FreeFemRunner
from .sofa_scene import sofaRun, L, H, W

HERE = os.path.dirname(os.path.abspath(__file__))

LABEL = "3D — 3-point bending"

MATH_DESCRIPTION = """
Poutre 3D simplement appuyee : ligne d'appui gauche (x=0, y=0) en pin
(ux=uy=uz=0), ligne d'appui droite (x=L, y=0) en rouleau (uy=0 seulement),
chargee par une charge lineique repartie le long de la ligne mediane
(x=L/2, y=H), poids reparti selon des poids trapezoidaux (longueur
tributaire) pour que la somme egale P.

Elasticite 3D complete (pas de notion plane strain/stress en 3D) :
    lambda = E*nu / ((1+nu)(1-2nu)), mu = E / (2(1+nu))

Pas de solution analytique 2D/3D exacte -> comparaison SOFA vs FreeFem
uniquement (RMS separement sur ux, uy, uz), plus un releve de la fleche
au point le plus proche de (L/2, H, W/2).
"""


def _load_params():
    with open(os.path.join(HERE, "params_beam3d_threept.json")) as f:
        return json.load(f)


def _mesh_path():
    return os.path.join(HERE, "beam3d_tet.msh")


def _to_freefem_path(path):
    """FreeFEM string literals on Windows can choke on backslashes; use forward slashes."""
    return path.replace(os.sep, "/")


def _pair_by_coordinates(x_a, y_a, z_a, x_b, y_b, z_b, tol=1e-6, snap=1e-6):
    """
    Apparie les noeuds entre deux copies independamment chargees du "meme"
    maillage, par coordonnees arrondies (snap).

    Le fichier de maillage brut porte du bruit flottant sous-tolerance issu
    de la generation du maillage (ex: x=0.8999999999997362 au lieu de 0.9),
    et ce bruit differe legerement entre noeuds conceptuellement au meme
    endroit selon la source (SOFA garde le bruit brut, FreeFem semble le
    nettoyer/arrondir en sortie). Trier sur les valeurs BRUTES est donc
    dangereux : une cle de tri primaire (x) "presque a egalite" peut
    s'ordonner differemment entre les deux sources, ce qui contamine
    ensuite les cles secondaires/tertiaires (y, z) et mele silencieusement
    des points pourtant identiques. Arrondir sur une grille bien au-dessus
    du bruit mais bien en-dessous de l'espacement reel du maillage corrige
    ce probleme.
    """
    x_a, y_a, z_a = map(np.asarray, (x_a, y_a, z_a))
    x_b, y_b, z_b = map(np.asarray, (x_b, y_b, z_b))

    if x_a.size != x_b.size:
        raise ValueError(
            f"Node COUNT mismatch: SOFA has {x_a.size} nodes, "
            f"FreeFEM has {x_b.size} nodes. Check that both loaded the "
            f"exact same mesh file (same path, no stale results.txt)."
        )

    def snap_(v):
        return np.round(v / snap) * snap

    xs_a, ys_a, zs_a = snap_(x_a), snap_(y_a), snap_(z_a)
    xs_b, ys_b, zs_b = snap_(x_b), snap_(y_b), snap_(z_b)

    order_a = np.lexsort((zs_a, ys_a, xs_a))
    order_b = np.lexsort((zs_b, ys_b, xs_b))

    if not (np.allclose(x_a[order_a], x_b[order_b], atol=tol)
            and np.allclose(y_a[order_a], y_b[order_b], atol=tol)
            and np.allclose(z_a[order_a], z_b[order_b], atol=tol)):
        raise ValueError("Node coordinates don't match between SOFA and FreeFEM meshes.")
    perm = np.empty_like(order_b)
    perm[order_b] = order_a
    return perm


def run_case():
    cfg = _load_params()
    P = float(cfg["P"])
    young_modulus = float(cfg["youngModulus"])
    poisson_ratio = float(cfg["poissonRatio"])
    mesh_file = _mesh_path()

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    ff_out_path = os.path.join(HERE, "results", "freefem_beam3d_threept_raw.txt")

    # --- FreeFem++ (ecrit directement un fichier texte, pas d'exportArray) ---
    runner = FreeFemRunner(os.path.join(HERE, "freefem_beam3d_threept.edp"))
    runner.execute({
        'P': P,
        'youngModulus': young_modulus,
        'poissonRatio': poisson_ratio,
        'meshFile': _to_freefem_path(mesh_file),
        'outFile': _to_freefem_path(ff_out_path),
    }, verbosity=0)

    if not os.path.isfile(ff_out_path):
        raise RuntimeError(
            f"FreeFEM did not produce the expected output file: {ff_out_path}\n"
            f"Check the FreeFEM console output above for compile/runtime errors."
        )

    raw = np.loadtxt(ff_out_path)
    x_ff, y_ff, z_ff = raw[:, 0], raw[:, 1], raw[:, 2]
    ux_ff, uy_ff, uz_ff = raw[:, 3], raw[:, 4], raw[:, 5]

    # --- SOFA ---
    pos0_sofa, u_sofa = sofaRun(mesh_file=mesh_file, P=P,
                                 young_modulus=young_modulus,
                                 poisson_ratio=poisson_ratio)
    x_sofa, y_sofa, z_sofa = pos0_sofa[:, 0], pos0_sofa[:, 1], pos0_sofa[:, 2]

    # --- Reordonner FreeFem sur l'ordre des noeuds SOFA (meme maillage) ---
    perm = _pair_by_coordinates(x_sofa, y_sofa, z_sofa, x_ff, y_ff, z_ff)
    u_ff = np.column_stack([ux_ff[perm], uy_ff[perm], uz_ff[perm]])

    # --- Fleche au point le plus proche de (L/2, H, W/2) ---
    mid_idx = np.argmin(np.abs(x_sofa - L / 2) + np.abs(y_sofa - H) + np.abs(z_sofa - W / 2))
    w_sofa = u_sofa[mid_idx, 1]
    w_ff = u_ff[mid_idx, 1]

    return {
        "label": LABEL,
        "dim": 3,
        "coords_ff": np.column_stack([x_sofa, y_sofa, z_sofa]),   # meme ordre que u_ff apres pairing
        "u_ff": u_ff,
        "coords_sofa": np.column_stack([x_sofa, y_sofa, z_sofa]),
        "u_sofa": u_sofa,
        "u_ana": None,
        "component_names": ["ux", "uy", "uz"],
        "midspan_deflection": {"w_sofa": w_sofa, "w_ff": w_ff},
    }
