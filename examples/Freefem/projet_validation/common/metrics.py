"""
Métriques génériques de comparaison, utilisées par tous les cas de validation
(1D, 2D, 3D, torsion...).
"""
import numpy as np


def rms(a, b):
    """Norme RMS entre deux champs : ||a - b||_2 / sqrt(n)."""
    a = np.asarray(a)
    b = np.asarray(b)
    return np.linalg.norm(a - b) / np.sqrt(a.size)


def compare_and_report(results):
    """
    Affiche les normes RMS pertinentes à partir d'un dict de résultats
    produit par un run_case() (voir cases/bar_1d/run.py pour le format 1D,
    cases/case_2d/distributed_load/run.py pour le format vectoriel 2D/3D).

    Clés attendues : 'label', 'u_ff', 'u_sofa', et optionnellement 'u_ana'.
    Si 'u_sofa'/'u_ff' sont des tableaux (n, dim) avec dim > 1, un RMS est
    aussi affiché composante par composante (ux, uy, ... ou 'component_names'
    si fourni dans le dict), comme dans comparaison_script_2d.py.
    """
    label = results.get("label", "Cas")
    u_ff = np.asarray(results["u_ff"])
    u_sofa = np.asarray(results["u_sofa"])
    u_ana = results.get("u_ana")

    print(f"--- {label} ---")
    print(f"RMS(sofa, ff)     = {rms(u_sofa, u_ff):.6e}")
    if u_ana is not None:
        print(f"RMS(sofa, exact)  = {rms(u_sofa, u_ana):.6e}")
        print(f"RMS(ff, exact)    = {rms(u_ff, u_ana):.6e}")

    if u_sofa.ndim > 1 and u_sofa.shape[1] > 1:
        names = results.get("component_names", [f"u{i}" for i in range(u_sofa.shape[1])])
        print("RMS par composante (SOFA vs FreeFem) :")
        for i, name in enumerate(names):
            print(f"  RMS_{name} (sofa, ff) = {rms(u_sofa[:, i], u_ff[:, i]):.6e}")


def compare_far_field(results, mean_shift_components=("uy",)):
    """
    Vérification analytique en zone lointaine, pour les cas où le champ
    exact n'est valable qu'à distance du bord encastré (effet Saint-Venant,
    ex: cases/case_2d/compression/run.py).

    Clés attendues dans `results` : 'coords_sofa', 'u_sofa',
    'u_far_field_ana', 'saint_venant_x'. Ne fait rien si ces clés sont
    absentes (cas qui n'a pas de champ lointain).

    `mean_shift_components` : noms de composantes (parmi 'component_names')
    à recentrer sur leur moyenne avant comparaison (utile quand une
    translation/rotation rigide n'est pas fixée par les conditions aux
    limites, ex: uy dans le cas compression).
    """
    u_ana = results.get("u_far_field_ana")
    threshold = results.get("saint_venant_x")
    if u_ana is None or threshold is None:
        return

    coords = np.asarray(results["coords_sofa"])
    u_sofa = np.asarray(results["u_sofa"])
    u_ana = np.asarray(u_ana)
    names = results.get("component_names", [f"u{i}" for i in range(u_sofa.shape[1])])

    mask = coords[:, 0] >= threshold
    if not mask.any():
        print(f"Aucun noeud au-dela du seuil Saint-Venant (x >= {threshold:.3f}).")
        return

    print(f"Far-field analytical check (x >= {threshold:.3f}) :")
    for i, name in enumerate(names):
        a = u_sofa[mask, i]
        b = u_ana[mask, i]
        if name in mean_shift_components:
            a = a - a.mean()
            b = b - b.mean()
            print(f"  RMS_{name} (sofa, analytique, mean-shifted) = {rms(a, b):.6e}")
        else:
            print(f"  RMS_{name} (sofa, analytique)               = {rms(a, b):.6e}")


def report_eb_check(results):
    """
    Affiche la vérification Euler-Bernoulli indicative à mi-portée, pour
    les cas qui en fournissent une (ex: cases/case_2d/three_point_bending).

    Ne fait rien si 'eb_check' est absent du dict de résultats. Rappel
    explicite : ce n'est PAS une cible de validation stricte (poutre
    courte, effets de cisaillement et concentrations de contrainte non
    captés par la théorie de poutre 1D) — voir MATH_DESCRIPTION du cas.
    """
    eb = results.get("eb_check")
    if eb is None:
        return
    print("Verification Euler-Bernoulli (INDICATIVE uniquement, pas une "
          "cible de validation stricte) :")
    print(f"  w_sofa (mi-portee) = {eb['w_sofa']:.6e}")
    print(f"  w_ff   (mi-portee) = {eb['w_ff']:.6e}")
    print(f"  w_EB   (analytique, poutre 1D)   = {eb['w_eb']:.6e}")
    print(f"  ratio SOFA/EB = {eb['ratio_sofa_eb']:.4f}  "
          f"(un ecart ici est ATTENDU, pas un signe d'erreur)")


def report_midspan_deflection(results):
    """
    Affiche la fleche au point milieu (SOFA vs FreeFem), pour les cas qui
    fournissent 'midspan_deflection' (ex: cases/case_3d/three_point_bending).
    Ne fait rien si la clé est absente.
    """
    mid = results.get("midspan_deflection")
    if mid is None:
        return
    print("Fleche au point le plus proche du milieu :")
    print(f"  w_sofa = {mid['w_sofa']:.6e}")
    print(f"  w_ff   = {mid['w_ff']:.6e}")


def report_relative_error(results):
    """
    Affiche les erreurs relatives (norme globale, en %) entre SOFA,
    FreeFem et la solution analytique, quand elle est disponible.
    Complementaire de compare_and_report (qui donne le RMS absolu) --
    utile quand les deplacements sont petits et que le RMS brut est peu
    parlant seul. Ex: cases/case_3d/torsion.

    Clés attendues : 'u_sofa', 'u_ff', optionnellement 'u_ana'.
    """
    u_sofa = np.asarray(results["u_sofa"])
    u_ff = np.asarray(results["u_ff"])
    u_ana = results.get("u_ana")

    def rel(ref, test):
        denom = np.linalg.norm(ref)
        return float(np.linalg.norm(test - ref) / denom) if denom > 0 else float("nan")

    print("Erreurs relatives (norme globale) :")
    print(f"  SOFA vs FreeFem   = {rel(u_sofa, u_ff):.3%}")
    if u_ana is not None:
        u_ana = np.asarray(u_ana)
        print(f"  SOFA vs Analytique = {rel(u_ana, u_sofa):.3%}")
        print(f"  FreeFem vs Analytique = {rel(u_ana, u_ff):.3%}")


def report_torsion_angle(results):
    """
    Affiche l'angle de torsion analytique (theta' et theta_total), pour
    les cas qui le fournissent (ex: cases/case_3d/torsion). Ne fait rien
    si les clés sont absentes.
    """
    theta_prime = results.get("theta_prime")
    theta_total = results.get("theta_total")
    if theta_prime is None:
        return
    print(f"theta' (analytique) = {theta_prime:.6g} rad/m")
    if theta_total is not None:
        print(f"theta_total (analytique, sur toute la longueur) = {theta_total:.6g} rad")
