"""
Fonctions de tracé génériques, utilisées par tous les cas de validation
(1D, 2D, 3D...). Un seul fichier pour tous les cas :

    - plot_displacement / plot_geometry                  -> cas 1D (champ scalaire, un axe x)
    - plot_deformed_mesh_2d / plot_displacement_field_2d  -> cas 2D/3D (champ vectoriel, maillage)

Contrat attendu du dict `results` (retourné par run_case() de chaque cas) :
    - "label"        : str
    - pour le 1D  : "x_ff", "u_ff", "x_sofa", "u_sofa", "u_ana" (ou None)
    - pour 2D/3D  : "dim" (2 ou 3), "coords_sofa", "u_sofa", "coords_ff", "u_ff",
                     "u_ana" (ou None), "triangles" (optionnel, pour un rendu maillage)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


# ------------------------------------------------------------------
# Cas 1D — champ scalaire, un seul axe x
# ------------------------------------------------------------------

def plot_displacement(results):
    """
    Trace u(x) pour les 3 approches (analytique, FreeFem, SOFA) à partir
    d'un dict de résultats produit par un run_case() 1D.
    """
    x_ff = results["x_ff"]
    u_ff = results["u_ff"]
    x_sofa = results["x_sofa"]
    u_sofa = results["u_sofa"]
    u_ana = results.get("u_ana")
    title = results.get("label", "Comparaison des déplacements")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if u_ana is not None:
        ax.plot(x_ff, u_ana, label="Analytique", linestyle="--", color="black")
    ax.plot(x_ff, u_ff, label="FreeFEM", marker="o", markersize=4, linestyle="none")
    ax.plot(x_sofa, u_sofa, label="SOFA", marker="x", markersize=5, linestyle="none")
    ax.set_xlabel("x")
    ax.set_ylabel("Déplacement u(x)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_geometry(results, scale=None):
    """
    Trace la géométrie repos vs déformée (schématique 1D) à partir
    d'un dict de résultats produit par un run_case() 1D.
    """
    x0 = results["x_sofa"]
    u = results["u_sofa"]
    title = results.get("label", "Géométrie — repos vs déformée")

    if scale is None:
        span = x0[-1] - x0[0]
        max_disp = np.max(np.abs(u)) or 1e-12
        scale = 0.15 * span / max_disp

    fig, ax = plt.subplots(figsize=(9, 2.5))
    y_rest, y_def = 1.0, 0.0
    ax.plot(x0, [y_rest] * len(x0), 'o-', color="gray", label="Configuration au repos")
    ax.plot(x0 + scale * u, [y_def] * len(x0), 'o-', color="crimson",
            label=f"Déformée (SOFA, ×{scale:.1f})")
    ax.plot([0, 0], [y_def - 0.3, y_rest + 0.3], 'k-', linewidth=3)
    ax.set_yticks([y_def, y_rest])
    ax.set_yticklabels(["déformée", "repos"])
    ax.set_xlabel("x")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(y_def - 0.6, y_rest + 0.6)
    fig.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# Cas 2D/3D — champ vectoriel, maillage
# ------------------------------------------------------------------
# Fidèle à comparaison_script_2d.py : une grille [SOFA | FreeFem | Diff]
# par composante du déplacement (ux, uy, ...), tripcolor sur le maillage.

def _triangulation(results):
    """Renvoie une Triangulation matplotlib à partir de 'triangles' + 'coords_sofa'."""
    tris = results.get("triangles")
    if tris is None:
        return None
    coords = results["coords_sofa"]
    return mtri.Triangulation(coords[:, 0], coords[:, 1], tris)


def plot_fields_2d(results):
    """
    Grille [SOFA | FreeFem | Diff] x [une ligne par composante], comme dans
    comparaison_script_2d.py (ex: ux en haut, uy en bas).

    Si 'vline_x' est présent dans `results` (ex: seuil Saint-Venant du cas
    compression), un trait pointillé vertical est tracé sur chaque sous-plot.
    """
    u_sofa = np.asarray(results["u_sofa"])
    u_ff = np.asarray(results["u_ff"])
    names = results.get("component_names", [f"u{i}" for i in range(u_sofa.shape[1])])
    triang = _triangulation(results)
    coords = results["coords_sofa"]
    vline_x = results.get("vline_x")

    def _plot_field(ax, vals, title, cmap="viridis"):
        if triang is not None:
            tc = ax.tripcolor(triang, vals, shading="gouraud", cmap=cmap)
        else:
            tc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, cmap=cmap, s=10)
        plt.colorbar(tc, ax=ax)
        ax.set_title(title)
        ax.set_aspect("equal")
        if vline_x is not None:
            ax.axvline(vline_x, color="k", ls="--", lw=0.8)

    n_comp = len(names)
    fig, axes = plt.subplots(n_comp, 3, figsize=(16, 4 * n_comp), squeeze=False)
    fig.suptitle(results.get("label", "Comparaison des champs"), fontsize=14)
    for row, name in enumerate(names):
        vs = u_sofa[:, row]
        vf = u_ff[:, row]
        _plot_field(axes[row, 0], vs, f"SOFA : {name}")
        _plot_field(axes[row, 1], vf, f"FreeFEM : {name}")
        _plot_field(axes[row, 2], vs - vf, f"Diff {name}", "RdBu")
    plt.tight_layout()
    plt.show()


def plot_parity(results):
    """
    Nuage de points SOFA vs FreeFem par composante, avec la diagonale y=x
    en référence (comme comparaison_script3d.py). Utile en 3D, où une
    carte de couleur sur le maillage (tripcolor) ne s'applique pas
    directement (tétraèdres). Fonctionne pour n'importe quelle dimension.
    """
    u_sofa = np.asarray(results["u_sofa"])
    u_ff = np.asarray(results["u_ff"])
    names = results.get("component_names", [f"u{i}" for i in range(u_sofa.shape[1])])

    n_comp = len(names)
    fig, axes = plt.subplots(1, n_comp, figsize=(5 * n_comp, 5), squeeze=False)
    fig.suptitle(results.get("label", "SOFA vs FreeFem (parité)"), fontsize=14)
    for i, name in enumerate(names):
        ax = axes[0, i]
        a, b = u_sofa[:, i], u_ff[:, i]
        ax.scatter(a, b, s=8, alpha=0.6)
        lims = [min(a.min(), b.min()), max(a.max(), b.max())]
        ax.plot(lims, lims, 'r--', linewidth=1)
        ax.set_xlabel(f"{name}_sofa")
        ax.set_ylabel(f"{name}_ff")
        ax.set_title(name)
        ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()
