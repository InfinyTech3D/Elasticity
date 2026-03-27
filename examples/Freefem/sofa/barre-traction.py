import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import tempfile
import os

# ─── PARAMÈTRES
L               = 1.0
E               = 1e6
A               = 0.001
N               = 64                        # Nombre d'éléments
OUTPUT_FF       = "freefem_deplacement.txt" # Sortie FreeFEM
OUTPUT_SOFA     = "sofa_deplacement.txt"    # Fichier SOFA à exporter 


FREEFEM_CODE = """
real L = {L};
real E = {E};
real A = {A};
int  n = {N};

meshL Th = segment(n, [x*L]);
fespace Vh(Th, P1);
Vh u, v;

problem Traction(u, v) =
    int1d(Th)( E * A * dx(u) * dx(v) )
    - int1d(Th)( 1.0 * v )
    + on(1, u=0);

Traction;

// Export noeuds : on parcourt les points du maillage
ofstream file("{OUTPUT_FF}");
file << "x u" << endl;
for (int i = 0; i < Th.nv; i++) {{
    real xi = Th(i).x;
    real ui = u(xi, 0);
    file << xi << " " << ui << endl;
}}

cout << "Export OK : " << Th.nv << " noeuds ecrits dans {OUTPUT_FF}" << endl;
""".format(L=L, E=E, A=A, N=N, OUTPUT_FF=OUTPUT_FF)

#  EXÉCUTION FREEFEM 
print("Lancement FreeFEM++...")
with tempfile.NamedTemporaryFile(mode='w', suffix='.edp',
                                  delete=False, encoding='utf-8') as f:
    f.write(FREEFEM_CODE)
    temp_file = f.name

try:
    result = subprocess.run(
        ['FreeFem++', temp_file],
        capture_output=True, text=True, encoding='utf-8'
    )
    print(result.stdout)
    if result.returncode != 0:
        print("ERREUR FreeFEM :\n", result.stderr)
        exit(1)
finally:
    os.unlink(temp_file)

# ─── LECTURE FREEFEM 
if not os.path.exists(OUTPUT_FF):
    print(f"Fichier {OUTPUT_FF} non trouvé — vérifier FreeFEM.")
    exit(1)

df_ff = pd.read_csv(OUTPUT_FF, sep=r'\s+')
x_ff  = df_ff['x'].values
u_ff  = df_ff['u'].values
print(f"\nFreeFEM : {len(x_ff)} nœuds chargés depuis '{OUTPUT_FF}'")

# ─── SOLUTION EXACTE 
# -EA u'' = 1, u(0)=0, u'(L)=0  →  u(x) = (1/EA) * x*(L - x/2)
x_exact = np.linspace(0, L, 500)
u_exact = (1.0 / (E * A)) * x_exact * (L - x_exact / 2.0)

# ─── LECTURE SOFA
sofa_disponible = os.path.exists(OUTPUT_SOFA)
if sofa_disponible:
    df_sofa = pd.read_csv(OUTPUT_SOFA, sep=r'\s+')
    # Colonnes attendues : x  u  (même convention que FreeFEM)
    x_sofa = df_sofa.iloc[:, 0].values
    u_sofa = df_sofa.iloc[:, 1].values
    print(f"SOFA     : {len(x_sofa)} nœuds chargés depuis '{OUTPUT_SOFA}'")

    # Interpolation FreeFEM sur les points SOFA pour comparaison ponctuelle
    u_ff_interp = np.interp(x_sofa, x_ff, u_ff)
    diff        = u_sofa - u_ff_interp
    err_max     = np.max(np.abs(diff))
    err_rms     = np.sqrt(np.mean(diff**2))
    print(f"\n--- Comparaison SOFA vs FreeFEM ---")
    print(f"  Erreur max  : {err_max:.4e}")
    print(f"  Erreur RMS  : {err_rms:.4e}")
else:
    print(f"\n(Fichier SOFA '{OUTPUT_SOFA}' absent — comparaison désactivée)")

# ─── FIGURE 
fig, axes = plt.subplots(1, 2 if sofa_disponible else 1,
                          figsize=(13 if sofa_disponible else 7, 5))
if not sofa_disponible:
    axes = [axes]

# Panneau 1 — Champ de déplacement
ax = axes[0]
ax.plot(x_exact, u_exact, '-',  color='gray',    lw=1.5, label='Solution exacte')
ax.plot(x_ff,    u_ff,    'o-', color='#1a6faf',  lw=2,   markersize=4,
        label=f'FreeFEM (N={N})')
if sofa_disponible:
    ax.plot(x_sofa, u_sofa, 's--', color='#d62728', lw=2, markersize=4,
            label='SOFA')
ax.set_xlabel("x (m)")
ax.set_ylabel("u(x) (m)")
ax.set_title("Déplacement axial — Barre 1D (charge uniforme)")
ax.legend()
ax.grid(True, alpha=0.3)

 
