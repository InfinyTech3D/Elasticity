"""
compare_freefem_sofa_complet.py
================================
Étude complète de comparaison FreeFEM vs SOFA
Poutre 2D encastrée sous gravité

Contenu :
  1. Chargement & normalisation des fichiers
  2. Interpolation commune (grille uniforme)
  3. Fig. 1 : Visualisation des maillages (nœuds + triangulation)
  4. Fig. 2 : Cartes de déplacements ux / uy (FF | SOFA | Diff)
  5. Fig. 3 : Profils sur la ligne médiane y = 1
  6. Fig. 4 : Cartes d'erreur relative (%)
  7. Tableau récapitulatif complet

Usage :
    python compare_freefem_sofa_complet.py

Fichiers requis (même dossier) :
    freefem_results.txt  — colonnes : x y ux uy [sigmaxx sigmayy sigmaxy sigmavm]
    sofa_results.txt     — colonnes : x y ux uy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# 0. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
FILE_FF   = "freefem_results.txt"
FILE_SOFA = "sofa_results.txt"
FIELDS    = ["ux", "uy"]
LABELS    = {"ux": "Déplacement horizontal $u_x$",
             "uy": "Déplacement vertical $u_y$"}

# Grille commune d'interpolation (domaine [0,10] × [0,2])
NX_GRID, NY_GRID = 300, 80

# ══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT & NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  COMPARAISON FreeFEM vs SOFA — Poutre 2D sous gravité")
print("=" * 65)

# --- FreeFEM ---
ff = pd.read_csv(FILE_FF, sep=r'\s+')
ff.columns = [c.strip() for c in ff.columns]
# S'assurer que x,y sont bien dans [0,10]×[0,2]
ff = ff[(ff["x"] >= -0.01) & (ff["x"] <= 10.01) &
        (ff["y"] >= -0.01) & (ff["y"] <=  2.01)].copy()

# --- SOFA ---
# Tentative lecture espace-séparé puis virgule
try:
    sf = pd.read_csv(FILE_SOFA, sep=r'\s+', comment='#')
    if sf.shape[1] < 4:
        raise ValueError
except Exception:
    sf = pd.read_csv(FILE_SOFA, sep=',', comment='#')
sf.columns = [c.strip() for c in sf.columns]

# Normalisation des noms de colonnes SOFA si besoin
col_map = {}
for c in sf.columns:
    cl = c.lower().strip()
    if cl in ("x",):   col_map[c] = "x"
    if cl in ("y",):   col_map[c] = "y"
    if cl in ("ux","u_x","disp_x"): col_map[c] = "ux"
    if cl in ("uy","u_y","disp_y"): col_map[c] = "uy"
sf = sf.rename(columns=col_map)

# Filtrage domaine SOFA (certains exports incluent z≠0)
sf = sf[(sf["x"] >= -0.01) & (sf["x"] <= 10.01) &
        (sf["y"] >= -0.01) & (sf["y"] <=  2.01)].copy()

print(f"\n  FreeFEM : {len(ff):>6} nœuds  |  SOFA : {len(sf):>6} nœuds")
print(f"  Colonnes FreeFEM : {list(ff.columns)}")
print(f"  Colonnes SOFA    : {list(sf.columns)}")


#  GRILLE COMMUNE & INTERPOLATION

gx, gy = np.mgrid[0:10:NX_GRID*1j, 0:2:NY_GRID*1j]
pts_grid = np.column_stack([gx.ravel(), gy.ravel()])

def interp_field(df, col, method="linear"):
    """Interpolation griddata sur la grille commune."""
    pts  = df[["x", "y"]].values
    vals = df[col].values
    return griddata(pts, vals, (gx, gy), method=method).reshape(gx.shape)

# FreeFEM → linéaire (maillage dense non-structuré)
ff_g = {f: interp_field(ff, f, method="linear") for f in FIELDS}

# SOFA → cubique si grille sparse, linéaire sinon
sofa_method = "linear" if len(sf) > 200 else "cubic"
sf_g = {f: interp_field(sf, f, method=sofa_method) for f in FIELDS}

# Différences absolues et relatives
diff_g = {f: sf_g[f] - ff_g[f] for f in FIELDS}
rel_g  = {}
for f in FIELDS:
    ref = np.abs(ff_g[f])
    with np.errstate(invalid="ignore", divide="ignore"):
        rel_g[f] = np.where(ref > 1e-12,
                            np.abs(diff_g[f]) / ref * 100.0,
                            np.nan)

print(f"\n  Interpolation SOFA  : méthode='{sofa_method}'")
print(f"  Grille commune      : {NX_GRID}×{NY_GRID} = {NX_GRID*NY_GRID} points")

# ══════════════════════════════════════════════════════════════════════════════
# 3. FIGURE 1 — VISUALISATION DES MAILLAGES
# ══════════════════════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(1, 2, figsize=(15, 5))
fig1.suptitle("Visualisation des maillages — FreeFEM vs SOFA",
              fontsize=14, fontweight="bold")

def plot_mesh(ax, df, title, color_node, color_tri):
    x = df["x"].values
    y = df["y"].values
    # Triangulation de Delaunay pour visualiser la connectivité
    if len(x) > 3:
        tri = Delaunay(np.column_stack([x, y]))
        # Filtrer les triangles trop grands (bord de domaine)
        simp = tri.simplices
        cx = x[simp].mean(axis=1)
        cy = y[simp].mean(axis=1)
        mask = (cx >= 0) & (cx <= 10) & (cy >= 0) & (cy <= 2)
        # Taille max des arêtes
        def edge_max(s):
            pts = np.column_stack([x[s], y[s]])
            d = [np.linalg.norm(pts[i]-pts[j])
                 for i,j in [(0,1),(1,2),(2,0)]]
            return max(d)
        edge_ok = np.array([edge_max(s) < 1.5 for s in simp])
        simp_ok = simp[mask & edge_ok]
        triang = mtri.Triangulation(x, y, simp_ok)
        ax.triplot(triang, color=color_tri, lw=0.4, alpha=0.6)
    ax.scatter(x, y, s=3, c=color_node, zorder=3, alpha=0.8)
    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(-0.1, 2.1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{title}\n({len(x)} nœuds)", fontsize=11)
    ax.grid(True, alpha=0.2)
    # Annotation densité
    ax.text(0.98, 0.04,
            f"Δx̄ ≈ {10/np.sqrt(len(x)):.3f} m",
            transform=ax.transAxes, ha="right", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

plot_mesh(axes1[0], ff, "FreeFEM (maillage triangulaire non-structuré)",
          "#1a6faf", "#6ab0de")
plot_mesh(axes1[1], sf, "SOFA (grille régulière + triangulation Q1→T3)",
          "#d62728", "#f4a09a")

plt.tight_layout()
plt.savefig("fig1_maillages.png", dpi=150, bbox_inches="tight")
print("\n  ✔  fig1_maillages.png")

# ══════════════════════════════════════════════════════════════════════════════
# 4. FIGURE 2 — CARTES DE DÉPLACEMENTS (3 × 2)
# ══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(3, 2, figsize=(15, 11))
fig2.suptitle("Cartes de déplacements — FreeFEM | SOFA | Différence",
              fontsize=14, fontweight="bold")

rows = [
    ("FreeFEM", ff_g,   "#1a6faf"),
    ("SOFA",    sf_g,   "#d62728"),
    ("Différence SOFA − FreeFEM", diff_g, None),
]

for row_i, (rname, rdata, _) in enumerate(rows):
    for col_i, field in enumerate(FIELDS):
        ax  = axes2[row_i, col_i]
        dat = rdata[field]
        if row_i < 2:
            cmap = "RdBu_r"
            vmax = np.nanpercentile(np.abs(dat), 99)
            im = ax.contourf(gx, gy, dat, levels=50,
                             cmap=cmap, vmin=-vmax, vmax=vmax)
        else:
            cmap = "coolwarm"
            vmax = np.nanpercentile(np.abs(dat), 97)
            im = ax.contourf(gx, gy, dat, levels=50,
                             cmap=cmap, vmin=-vmax, vmax=vmax)
        cb = fig2.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cb.ax.tick_params(labelsize=8)
        ax.set_title(f"{rname} — {LABELS[field]}", fontsize=10)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        # Valeur extrême
        vext = np.nanmax(np.abs(dat))
        ax.text(0.01, 0.04, f"max|·| = {vext:.4e}",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75))

plt.tight_layout()
plt.savefig("fig2_deplacements.png", dpi=150, bbox_inches="tight")
print("  ✔  fig2_deplacements.png")

# ══════════════════════════════════════════════════════════════════════════════
# 5. FIGURE 3 — PROFILS LIGNE MÉDIANE y = 1
# ══════════════════════════════════════════════════════════════════════════════
# Stratégie : on lit le profil depuis la GRILLE COMMUNE (déjà interpolée)
# pour les deux codes → pas de problème de nœuds manquants.
# On superpose aussi les nœuds bruts quand ils existent près de y=1.
# ── Tolérance adaptée à la densité de chaque maillage ──────────────────────
#   FreeFEM : maillage dense (~4800 nœuds) → beaucoup de pts près de y=1
#   SOFA    : grille 11×4  → les 4 lignes y sont à 0, 0.67, 1.33, 2
#             → aucun nœud à y=1 ! On prend la tolérance suffisante
#             pour capturer y=0.67 ou y=1.33 (écart ≈ 0.33)
def tol_for(df, y_target=1.0, n_min=3):
    """Trouve la tolérance minimale qui capture au moins n_min nœuds."""
    y_unique = np.sort(np.unique(np.round(df["y"].values, 6)))
    nearest  = y_unique[np.argmin(np.abs(y_unique - y_target))]
    gap      = np.abs(nearest - y_target) + 0.05   # petite marge
    return max(gap, 0.12)

xs_line = np.linspace(0, 10, 400)

fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle("Profils de déplacement — ligne médiane y ≈ 1\n"
              "(courbes depuis grille commune ; points = nœuds bruts)",
              fontsize=12, fontweight="bold")

for ax, field in zip(axes3, FIELDS):

    # ── A. Courbes depuis la grille commune (interpolée) ─────────────────
    iy_common = np.argmin(np.abs(gy[0, :] - 1.0))
    xs_g = gx[:, iy_common]

    ax.plot(xs_g, ff_g[field][:, iy_common],
            color="#1a6faf", lw=2.2, label="FreeFEM (grille commune)")
    ax.plot(xs_g, sf_g[field][:, iy_common],
            color="#d62728", lw=2.2, ls="--", label="SOFA (grille commune)")

    # ── B. Nœuds bruts FreeFEM (nombreux → scatter léger) ────────────────
    tol_ff = tol_for(ff)
    sub_ff = ff[np.abs(ff["y"] - 1.0) < tol_ff].sort_values("x")
    ax.scatter(sub_ff["x"], sub_ff[field],
               s=6, c="#1a6faf", alpha=0.25, marker="o",
               label=f"FreeFEM nœuds (|y−1|<{tol_ff:.2f})")

    # ── C. Nœuds bruts SOFA (peu nombreux → gros marqueurs) ──────────────
    tol_sf = tol_for(sf)
    sub_sf = sf[np.abs(sf["y"] - 1.0) < tol_sf].sort_values("x")
    if len(sub_sf) > 0:
        ax.scatter(sub_sf["x"], sub_sf[field],
                   s=60, c="#d62728", alpha=0.8, marker="s", zorder=5,
                   label=f"SOFA nœuds (|y−1|<{tol_sf:.2f}, n={len(sub_sf)})")
    else:
        ax.text(0.5, 0.5, "Aucun nœud SOFA proche de y=1\n(grille trop sparse)",
                transform=ax.transAxes, ha="center", fontsize=9,
                color="#d62728", alpha=0.7)

    # ── D. Différence sur axe secondaire ─────────────────────────────────
    diff_line = diff_g[field][:, iy_common]
    ax2 = ax.twinx()
    ax2.fill_between(xs_g, 0, diff_line,
                     color="#2ca02c", alpha=0.15, label="Diff (aire)")
    ax2.plot(xs_g, diff_line, color="#2ca02c", lw=1.3,
             ls=":", label="Diff SOFA−FF")
    ax2.axhline(0, color="#2ca02c", lw=0.5, ls="-", alpha=0.4)
    ax2.set_ylabel("Δ = SOFA − FreeFEM", color="#2ca02c", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="#2ca02c")

    ax.set_xlabel("x (m)")
    ax.set_ylabel(field)
    ax.set_title(LABELS[field], fontsize=11)
    ax.axvline(0, color="gray", lw=0.8, ls=":")
    ax.grid(True, alpha=0.25)
    lines_a, labs_a = ax.get_legend_handles_labels()
    lines_b, labs_b = ax2.get_legend_handles_labels()
    ax.legend(lines_a + lines_b, labs_a + labs_b,
              fontsize=8, loc="lower left", ncol=2)

plt.tight_layout()
plt.savefig("fig3_profil_mediane.png", dpi=150, bbox_inches="tight")
print("  ✔  fig3_profil_mediane.png")

# ══════════════════════════════════════════════════════════════════════════════
# 6. FIGURE 4 — CARTES D'ERREUR RELATIVE (%)
# ══════════════════════════════════════════════════════════════════════════════
fig4, axes4 = plt.subplots(1, 2, figsize=(14, 4.5))
fig4.suptitle("Erreur relative |SOFA − FreeFEM| / |FreeFEM| × 100 (%)",
              fontsize=13, fontweight="bold")

for ax, field in zip(axes4, FIELDS):
    dat  = rel_g[field]
    vmax = np.nanpercentile(dat, 95)
    im   = ax.contourf(gx, gy, dat, levels=50,
                       cmap="hot_r", vmin=0, vmax=vmax)
    cb   = fig4.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label("%")
    ax.set_title(f"Erreur relative — {LABELS[field]}", fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    med = np.nanmedian(dat)
    mn  = np.nanmean(dat)
    ax.text(0.01, 0.04,
            f"Médiane : {med:.2f}%   Moyenne : {mn:.2f}%",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75))

plt.tight_layout()
plt.savefig("fig4_erreur_relative.png", dpi=150, bbox_inches="tight")
print("  ✔  fig4_erreur_relative.png")

# ══════════════════════════════════════════════════════════════════════════════
# 7. FIGURE 5 — HISTOGRAMMES DES ERREURS
# ══════════════════════════════════════════════════════════════════════════════
fig5, axes5 = plt.subplots(1, 2, figsize=(12, 4))
fig5.suptitle("Distribution des erreurs absolues SOFA − FreeFEM",
              fontsize=13, fontweight="bold")

for ax, field in zip(axes5, FIELDS):
    d = diff_g[field].ravel()
    d = d[~np.isnan(d)]
    ax.hist(d, bins=80, color="#5a9fd4", edgecolor="white",
            linewidth=0.3, alpha=0.85)
    ax.axvline(np.mean(d),  color="#d62728", lw=1.8, ls="-",
               label=f"Moyenne : {np.mean(d):.2e}")
    ax.axvline(np.median(d), color="#2ca02c", lw=1.8, ls="--",
               label=f"Médiane : {np.median(d):.2e}")
    ax.set_xlabel(f"Δ{field} = SOFA − FreeFEM")
    ax.set_ylabel("Nombre de points")
    ax.set_title(LABELS[field], fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    # Annotations stats
    txt = (f"σ = {np.std(d):.2e}\n"
           f"max|·| = {np.abs(d).max():.2e}\n"
           f"RMSE = {np.sqrt(np.mean(d**2)):.2e}")
    ax.text(0.97, 0.97, txt, transform=ax.transAxes,
            va="top", ha="right", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

plt.tight_layout()
plt.savefig("fig5_histogrammes.png", dpi=150, bbox_inches="tight")
print("  ✔  fig5_histogrammes.png")

# ══════════════════════════════════════════════════════════════════════════════
# 8. TABLEAU RÉCAPITULATIF COMPLET
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 72)
print(f"  {'Quantité':<30} {'FreeFEM':>12} {'SOFA':>12} "
      f"{'Δ absolu':>12} {'Δ relatif%':>11}")
print("═" * 72)

metrics = {
    # Déplacements verticaux
    "max |u_y|"         : lambda d: np.abs(d["uy"]).max(),
    "min  u_y  (flèche)": lambda d: d["uy"].min(),
    "mean u_y"          : lambda d: d["uy"].mean(),
    "RMSE u_y (grille)" : None,   # traité séparément
    # Déplacements horizontaux
    "max |u_x|"         : lambda d: np.abs(d["ux"]).max(),
    "min  u_x"          : lambda d: d["ux"].min(),
    "mean u_x"          : lambda d: d["ux"].mean(),
    "RMSE u_x (grille)" : None,
    # Erreurs relatives médianes
    "Erreur rel. méd. u_y (%)": None,
    "Erreur rel. méd. u_x (%)": None,
    "Erreur rel. moy. u_y (%)": None,
    "Erreur rel. moy. u_x (%)": None,
}

# Calcul RMSE sur grille commune
rmse = {}
for f in FIELDS:
    d = diff_g[f].ravel()
    d = d[~np.isnan(d)]
    rmse[f] = np.sqrt(np.mean(d**2))

# Erreurs relatives médianes / moyennes sur grille
rel_med = {f: np.nanmedian(rel_g[f]) for f in FIELDS}
rel_moy = {f: np.nanmean(rel_g[f])   for f in FIELDS}

def pct(vff, vsf):
    return abs(vsf - vff) / (abs(vff) + 1e-15) * 100

rows_table = [
    ("── DÉPLACEMENT VERTICAL u_y ──────────────────────────────", None, None),
    ("max |u_y|",          np.abs(ff["uy"]).max(), np.abs(sf["uy"]).max()),
    ("min  u_y  (flèche)", ff["uy"].min(),          sf["uy"].min()),
    ("mean u_y",           ff["uy"].mean(),          sf["uy"].mean()),
    ("RMSE u_y (grille)",  None,                     rmse["uy"]),
    ("── DÉPLACEMENT HORIZONTAL u_x ────────────────────────────", None, None),
    ("max |u_x|",          np.abs(ff["ux"]).max(), np.abs(sf["ux"]).max()),
    ("min  u_x",           ff["ux"].min(),           sf["ux"].min()),
    ("mean u_x",           ff["ux"].mean(),           sf["ux"].mean()),
    ("RMSE u_x (grille)",  None,                     rmse["ux"]),
    ("── ERREURS RELATIVES SUR GRILLE COMMUNE ──────────────────", None, None),
    ("Erreur rel. médiane u_y (%)", rel_med["uy"], None),
    ("Erreur rel. médiane u_x (%)", rel_med["ux"], None),
    ("Erreur rel. moyenne u_y (%)", rel_moy["uy"], None),
    ("Erreur rel. moyenne u_x (%)", rel_moy["ux"], None),
    ("── INFORMATIONS MAILLAGE ──────────────────────────────────", None, None),
    ("Nœuds FreeFEM",      len(ff),  None),
    ("Nœuds SOFA",         len(sf),  None),
    ("Rapport nœuds FF/SOFA", len(ff)/max(len(sf),1), None),
]

for row in rows_table:
    label, vff, vsf = row
    if vff is None and vsf is None:
        print(f"\n  {label}")
        continue
    if vsf is None:
        # Valeur unique (erreur relative, info maillage)
        print(f"  {label:<45} {vff:>12.4f}")
        continue
    if vff is None:
        # RMSE (pas de valeur FF individuelle)
        print(f"  {label:<45} {'—':>12} {vsf:>12.4e}")
        continue
    delta = vsf - vff
    dp    = pct(vff, vsf)
    print(f"  {label:<45} {vff:>12.6f} {vsf:>12.6f} "
          f"{delta:>12.2e} {dp:>10.2f}%")

print("═" * 72)

# ══════════════════════════════════════════════════════════════════════════════
# 9. EXPORT CSV DU TABLEAU
# ══════════════════════════════════════════════════════════════════════════════
summary_rows = []
for f in FIELDS:
    ff_max = np.abs(ff[f]).max()
    sf_max = np.abs(sf[f]).max()
    ff_min = ff[f].min()
    sf_min = sf[f].min()
    summary_rows.append({
        "champ": f,
        "FF_max_abs":   ff_max,
        "SF_max_abs":   sf_max,
        "delta_max_abs": sf_max - ff_max,
        "pct_max_abs":  pct(ff_max, sf_max),
        "FF_min":       ff_min,
        "SF_min":       sf_min,
        "delta_min":    sf_min - ff_min,
        "pct_min":      pct(ff_min, sf_min),
        "RMSE_grille":  rmse[f],
        "err_rel_med_pct": rel_med[f],
        "err_rel_moy_pct": rel_moy[f],
        "noeuds_FF":    len(ff),
        "noeuds_SF":    len(sf),
    })

pd.DataFrame(summary_rows).to_csv("comparaison_summary.csv", index=False)
print("\n  ✔  comparaison_summary.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 10. AFFICHAGE
# ══════════════════════════════════════════════════════════════════════════════
import os, subprocess, sys

output_dir = os.path.dirname(os.path.abspath(__file__))
figures = ["fig1_maillages.png", "fig2_deplacements.png",
           "fig3_profil_mediane.png", "fig4_erreur_relative.png",
           "fig5_histogrammes.png"]

print("\n  Fichiers générés :")
for fname in figures + ["comparaison_summary.csv"]:
    fpath = os.path.join(output_dir, fname)
    size  = os.path.getsize(fpath) // 1024 if os.path.exists(fpath) else 0
    print(f"    • {fname}  ({size} Ko)")

# Ouvrir automatiquement les images avec le viewer Windows/Mac/Linux
print("\n  Ouverture des figures...")
for fname in figures:
    fpath = os.path.join(output_dir, fname)
    if not os.path.exists(fpath):
        continue
    try:
        if sys.platform.startswith("win"):
            os.startfile(fpath)          # Windows : ouvre avec la visionneuse par défaut
        elif sys.platform == "darwin":
            subprocess.Popen(["open", fpath])
        else:
            subprocess.Popen(["xdg-open", fpath])
    except Exception as e:
        print(f"     Impossible d'ouvrir {fname} : {e}")


try:
    plt.show()
except Exception:
    pass






# ══════════════════════════════════════════════════════════════════════════════
# 11. FIGURE 6 — VISUALISATION DES NŒUDS AVEC LEURS NUMÉROS
# ══════════════════════════════════════════════════════════════════════════════

def plot_nodes_with_numbers(ax, df, title, color_node='blue', show_all=False, max_nodes_to_show=100):
    """
    Affiche les nœuds avec leurs numéros.
    
    Parameters:
    -----------
    ax : axes matplotlib
    df : DataFrame avec colonnes x, y
    title : str
    color_node : couleur des nœuds
    show_all : bool - si True affiche tous les nœuds, sinon limite à max_nodes_to_show
    max_nodes_to_show : int - nombre maximum de nœuds à étiqueter
    """
    x = df["x"].values
    y = df["y"].values
    n_nodes = len(x)
    
    # Tracer tous les nœuds
    ax.scatter(x, y, s=20, c=color_node, zorder=3, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Décider quels nœuds étiqueter
    if show_all or n_nodes <= max_nodes_to_show:
        # Afficher tous les nœuds
        indices = range(n_nodes)
    else:
        # Afficher seulement quelques nœuds (bords + quelques internes)
        # Identifier les nœuds uniques par ligne y
        y_unique = np.unique(np.round(y, 6))
        n_y = len(y_unique)
        n_x = n_nodes // n_y
        
        indices_to_label = []
        
        # Ajouter les coins
        corners = [0, n_x-1, n_nodes - n_x, n_nodes - 1]
        indices_to_label.extend(corners)
        
        # Ajouter quelques nœuds sur les bords
        step = max(1, n_x // 10)
        for i in range(0, n_x, step):
            indices_to_label.append(i)  # bord bas
            indices_to_label.append(n_nodes - n_x + i)  # bord haut
        
        # Ajouter quelques nœuds internes
        step_y = max(1, n_y // 5)
        step_x = max(1, n_x // 5)
        for iy in range(0, n_y, step_y):
            for ix in range(0, n_x, step_x):
                idx = iy * n_x + ix
                if idx not in indices_to_label:
                    indices_to_label.append(idx)
        
        indices = list(set(indices_to_label))  # Éliminer les doublons
    
    # Afficher les numéros
    for idx in indices:
        if idx < n_nodes:
            ax.annotate(str(idx), 
                       (x[idx], y[idx]),
                       xytext=(5, 5), 
                       textcoords='offset points',
                       fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="gray"),
                       alpha=0.8)
    
    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(-0.1, 2.1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{title}\n({n_nodes} nœuds", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Ajouter une légende sur le nombre de nœuds affichés
    if n_nodes > max_nodes_to_show:
        ax.text(0.02, 0.98, f"Affichage de {len(indices)}/{n_nodes} nœuds",
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.7))


# Créer la figure 6
fig6, axes6 = plt.subplots(2, 2, figsize=(16, 12))
fig6.suptitle("Visualisation des nœuds avec leurs numéros", 
              fontsize=14, fontweight="bold")

# Sous-figure pour FreeFEM (tous les nœuds)
plot_nodes_with_numbers(axes6[0, 0], ff, "FreeFEM - Tous les nœuds", 
                        color_node='#1a6faf', show_all=True)

# Sous-figure pour FreeFEM (limité pour lisibilité)
plot_nodes_with_numbers(axes6[0, 1], ff, "FreeFEM - Échantillon de nœuds", 
                        color_node='#1a6faf', show_all=False, max_nodes_to_show=100)

# Sous-figure pour SOFA (tous les nœuds)
plot_nodes_with_numbers(axes6[1, 0], sf, "SOFA - Tous les nœuds", 
                        color_node='#d62728', show_all=True)

# Sous-figure pour SOFA (avec détails supplémentaires)
ax = axes6[1, 1]
x_sf = sf["x"].values
y_sf = sf["y"].values
ax.scatter(x_sf, y_sf, s=50, c='#d62728', zorder=3, alpha=0.8, edgecolors='black', linewidth=0.5)

# Afficher tous les numéros pour SOFA (peu nombreux)
for idx, (xi, yi) in enumerate(zip(x_sf, y_sf)):
    ax.annotate(str(idx), 
               (xi, yi),
               xytext=(8, 8), 
               textcoords='offset points',
               fontsize=10,
               fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="red", lw=1.5),
               alpha=0.9)

# Ajouter les coordonnées pour quelques nœuds clés
for idx, (xi, yi) in enumerate(zip(x_sf, y_sf)):
    if idx % 5 == 0 or idx < 4 or idx >= len(x_sf)-4:  # Afficher les coordonnées pour certains nœuds
        ax.annotate(f"({xi:.1f}, {yi:.1f})", 
                   (xi, yi),
                   xytext=(8, -15), 
                   textcoords='offset points',
                   fontsize=7,
                   alpha=0.7)

ax.set_xlim(-0.2, 10.2)
ax.set_ylim(-0.1, 2.1)
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title(f"SOFA - Détail des nœuds\n({len(sf)} nœuds, grille régulière)", fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fig6_noeuds_avec_numeros.png", dpi=150, bbox_inches="tight")
print("  ✔  fig6_noeuds_avec_numeros.png")


# ══════════════════════════════════════════════════════════════════════════════
# 12. FIGURE 7 — TOPOLOGIE DU MAILLAGE SOFA (grille structurée)
# ══════════════════════════════════════════════════════════════════════════════

def plot_sofa_grid_structure(ax, df, nx, ny):
    """Visualise la structure de grille de SOFA avec les connexions"""
    x = df["x"].values
    y = df["y"].values
    
    # Déterminer nx, ny à partir des données si non fournis
    if nx is None or ny is None:
        y_unique = np.unique(np.round(y, 6))
        ny = len(y_unique)
        nx = len(x) // ny
    
    # Tracer les lignes de la grille
    for i in range(ny):
        row_indices = slice(i*nx, (i+1)*nx)
        ax.plot(x[row_indices], y[row_indices], 'b-', linewidth=1, alpha=0.5, color='gray')
    
    for j in range(nx):
        col_indices = list(range(j, nx*ny, nx))
        ax.plot(x[col_indices], y[col_indices], 'b-', linewidth=1, alpha=0.5, color='gray')
    
    # Tracer les nœuds
    ax.scatter(x, y, s=30, c='#d62728', zorder=3, alpha=0.8, edgecolors='black')
    
    # Numéroter les nœuds
    for idx, (xi, yi) in enumerate(zip(x, y)):
        ax.annotate(str(idx), 
                   (xi, yi),
                   xytext=(5, 5), 
                   textcoords='offset points',
                   fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))
    
    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(-0.1, 2.1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Structure de la grille SOFA\n{nx} × {ny} = {nx*ny} nœuds", fontsize=11)
    ax.grid(True, alpha=0.3)

# Créer la figure 7
fig7, ax7 = plt.subplots(1, 1, figsize=(12, 6))

# Déterminer nx, ny à partir des données SOFA
y_unique_sf = np.unique(np.round(sf["y"].values, 6))
ny_sf = len(y_unique_sf)
nx_sf = len(sf) // ny_sf

plot_sofa_grid_structure(ax7, sf, nx_sf, ny_sf)

plt.tight_layout()
plt.savefig("fig7_grille_sofa.png", dpi=150, bbox_inches="tight")
print("  ✔  fig7_grille_sofa.png")


# ══════════════════════════════════════════════════════════════════════════════
# 13. FIGURE 8 — CARTE DE CHALEUR DE LA DENSITÉ DE NŒUDS (FreeFEM)
# ══════════════════════════════════════════════════════════════════════════════

def plot_node_density_heatmap(ax, df, title, bins=50):
    """Affiche une carte de chaleur de la densité de nœuds"""
    x = df["x"].values
    y = df["y"].values
    
    # Créer l'histogramme 2D
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[0, 10], [0, 2]])
    
    # Afficher la carte de chaleur
    im = ax.imshow(H.T, origin='lower', extent=[0, 10, 0, 2], 
                   cmap='hot', aspect='auto', alpha=0.8)
    
    # Ajouter la colorbar
    plt.colorbar(im, ax=ax, label='Nombre de nœuds par cellule')
    
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)

fig8, axes8 = plt.subplots(1, 2, figsize=(14, 5))
fig8.suptitle("Densité de nœuds dans le maillage", fontsize=13, fontweight="bold")

plot_node_density_heatmap(axes8[0], ff, "FreeFEM - Distribution des nœuds", bins=40)
plot_node_density_heatmap(axes8[1], sf, "SOFA - Distribution des nœuds (grille régulière)", bins=20)

plt.tight_layout()
plt.savefig("fig8_densite_noeuds.png", dpi=150, bbox_inches="tight")
print("  ✔  fig8_densite_noeuds.png")


# ══════════════════════════════════════════════════════════════════════════════
# Mettre à jour la liste des figures
figures += ["fig6_noeuds_avec_numeros.png", "fig7_grille_sofa.png", "fig8_densite_noeuds.png"]






# prise de note 

"""
POURQUOI ON PEUT COMPARER LES DEUX CODES MALGRÉ DES MAILLAGES DIFFÉRENTS ?
===========================================================================

PROBLÈME DE DÉPART :
  • FreeFEM génère un maillage triangulaire NON-STRUCTURÉ avec ~4800 nœuds.
    Les nœuds sont placés de façon irrégulière, denses là où le gradient est
    fort (encastrement gauche), plus rares au centre.

  • SOFA utilise une RegularGridTopology 11×4 = 44 nœuds seulement,
    placés sur une grille parfaitement régulière.
    Les lignes y sont : 0.0, 0.67, 1.33, 2.0
    → aucun nœud à y = 1.0 exactement !

SOLUTION — LA GRILLE COMMUNE :
  1. On crée une grille uniforme 300×80 = 24 000 points couvrant [0,10]×[0,2].
  2. On interpole chaque champ (ux, uy) sur cette grille depuis les nœuds bruts :
       - FreeFEM : scipy.griddata(..., method='linear')
         → triangulation Delaunay + interpolation linéaire par morceaux (P1)
         → très fidèle car maillage dense
       - SOFA : scipy.griddata(..., method='cubic')
         → interpolation cubique car 44 points sont trop peu pour 'linear'
         → lisse le champ entre les nœuds réguliers

  3. Une fois les deux champs sur la même grille, on peut :
       diff[i,j] = SOFA_interpolé[i,j] - FreeFEM_interpolé[i,j]
     et calculer RMSE, erreur relative, histogrammes, etc.

POURQUOI LA COMPARAISON A DU SENS ?
  Les deux codes résolvent la MÊME équation (élasticité linéaire plane,
  même E, nu, gravité, encastrement gauche). La différence vient de :
    a) La formulation : plane_strain (Vec3, λ=Eν/((1+ν)(1-2ν))) vs
       les coefficients Lamé 3D de FreeFEM (identiques à plane_strain)
    b) La discrétisation : triangles P1 vs Q1 subdivisé en triangles
    c) Le nombre de DDL : 44 nœuds SOFA << 4814 nœuds FreeFEM
       → SOFA est moins précis, mais donne la bonne tendance physique.

  Le tableau récapitulatif vous donne l'écart % entre les deux pour
  chaque quantité d'intérêt (flèche maximale, RMSE, erreur médiane...).
"""