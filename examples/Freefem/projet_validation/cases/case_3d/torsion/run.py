import json
import os
import numpy as np
from common import env_setup  # noqa: F401 -- corrige PATH pour stdbuf (Windows)
from pyfreefem import FreeFemRunner
from .mesh_gen import generate_beam3D_circular_tet
from .sofa_scene import sofaRun, MESH_DIR, DEFAULT_MESH_FILENAME

HERE = os.path.dirname(os.path.abspath(__file__))

LABEL = "3D — Circular beam, pure torsion"

MATH_DESCRIPTION = """
Poutre 3D a section circulaire (rayon r), encastree en x=xmin (deplacement
nul), soumise a un couple de torsion pur T sur la face x=xmax (traction
tangentielle tau = (T/J)*(-(z-zc), (y-yc)) integree sur la face).

    J = pi*r^4/2  (moment d'inertie polaire, section circulaire)
    G = E / (2(1+nu))
    theta' = T / (G*J)   (angle de torsion par unite de longueur)

Solution analytique EXACTE (une section circulaire ne se gauchit pas sous
torsion pure -> valable sur tout le domaine, contrairement aux cas
traction/compression qui ont une couche limite de Saint-Venant) :

    ux(x,y,z) = 0
    uy(x,y,z) = -theta' * x * (z - zc)
    uz(x,y,z) =  theta' * x * (y - yc)

avec (yc, zc) le centre de la section. Hypothese petites deformations :
valide si theta'*L < 0.1 rad (verifie automatiquement par la scene SOFA).
"""


def _load_mesh_cfg():
    with open(os.path.join(HERE, "params_mesh.json")) as f:
        return json.load(f)["beam3d_circle_tet"]


def _load_phys_cfg():
    with open(os.path.join(HERE, "params_beam3d_torsion.json")) as f:
        return json.load(f)


def _to_freefem_path(path):
    """FreeFEM string literals on Windows can choke on backslashes; use forward slashes."""
    return path.replace(os.sep, "/")


def _pair_by_coordinates(x_a, y_a, z_a, x_b, y_b, z_b, tol=1e-6, snap=1e-6):
    """Appariement robuste par snapping (fidele a compare_beam3d_torsion.py,
    meme technique que cases/case_3d/three_point_bending)."""
    x_a, y_a, z_a = map(np.asarray, (x_a, y_a, z_a))
    x_b, y_b, z_b = map(np.asarray, (x_b, y_b, z_b))

    if x_a.size != x_b.size:
        raise ValueError(
            f"Node COUNT mismatch: SOFA has {x_a.size} nodes, "
            f"FreeFEM has {x_b.size} nodes."
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


def _analytical_displacement(x0, torque, radius, young_modulus, poisson_ratio, yc, zc):
    G = young_modulus / (2.0 * (1.0 + poisson_ratio))
    J = np.pi * radius**4 / 2.0
    theta_prime = torque / (G * J)

    x, y, z = x0[:, 0], x0[:, 1], x0[:, 2]
    ux = np.zeros_like(x)
    uy = -theta_prime * x * (z - zc)
    uz = theta_prime * x * (y - yc)
    return np.column_stack([ux, uy, uz]), theta_prime


def run_case():
    mesh_cfg = _load_mesh_cfg()
    phys_cfg = _load_phys_cfg()

    T = float(phys_cfg["T"])
    radius = float(phys_cfg["radius"])
    young_modulus = float(phys_cfg["youngModulus"])
    poisson_ratio = float(phys_cfg["poissonRatio"])

    # --- Maillage : genere s'il n'existe pas deja ---
    meshfile = mesh_cfg.get("meshfile", DEFAULT_MESH_FILENAME)
    mesh_path = os.path.join(MESH_DIR, meshfile)
    if not os.path.isfile(mesh_path):
        generate_beam3D_circular_tet(
            length=float(mesh_cfg["length"]),
            radius=float(mesh_cfg["radius"]),
            mesh_size=float(mesh_cfg["mesh_size"]),
            filename=meshfile,
        )

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    ff_out_path = os.path.join(HERE, "results", "freefem_beam3d_torsion_raw.txt")

    # --- FreeFem++ (ecrit directement un fichier texte, pas d'exportArray) ---
    runner = FreeFemRunner(os.path.join(HERE, "freefem_beam3d_torsion.edp"))
    runner.execute({
        'T': T,
        'radius': radius,
        'youngModulus': young_modulus,
        'poissonRatio': poisson_ratio,
        'meshFile': _to_freefem_path(mesh_path),
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

    # --- SOFA (fait aussi une verification interne "est-ce une vraie torsion") ---
    pos0_sofa, u_sofa = sofaRun(mesh_file=mesh_path, T=T, radius=radius,
                                 young_modulus=young_modulus,
                                 poisson_ratio=poisson_ratio)
    x_sofa, y_sofa, z_sofa = pos0_sofa[:, 0], pos0_sofa[:, 1], pos0_sofa[:, 2]

    # --- Reordonner FreeFem sur l'ordre des noeuds SOFA (meme maillage) ---
    perm = _pair_by_coordinates(x_sofa, y_sofa, z_sofa, x_ff, y_ff, z_ff)
    u_ff = np.column_stack([ux_ff[perm], uy_ff[perm], uz_ff[perm]])

    # --- Solution analytique EXACTE (valable partout, cf MATH_DESCRIPTION) ---
    yc = 0.5 * (y_sofa.min() + y_sofa.max())
    zc = 0.5 * (z_sofa.min() + z_sofa.max())
    u_ana, theta_prime = _analytical_displacement(
        pos0_sofa, T, radius, young_modulus, poisson_ratio, yc, zc
    )
    length = x_sofa.max() - x_sofa.min()

    return {
        "label": LABEL,
        "dim": 3,
        "coords_ff": np.column_stack([x_sofa, y_sofa, z_sofa]),   # meme ordre que u_ff apres pairing
        "u_ff": u_ff,
        "coords_sofa": np.column_stack([x_sofa, y_sofa, z_sofa]),
        "u_sofa": u_sofa,
        "u_ana": u_ana,     # exact partout, pas de masque necessaire
        "component_names": ["ux", "uy", "uz"],
        "theta_prime": theta_prime,
        "theta_total": theta_prime * length,
    }
