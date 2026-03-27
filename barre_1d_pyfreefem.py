import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import tempfile
import os

# PARAMÈTRES 
L         = 1.0
E         = 1e6
A         = 0.001
N         = 64
OUTPUT_FF = "freefem_deplacement.txt"


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

#  LECTURE RÉSULTATS 
if not os.path.exists(OUTPUT_FF):
    print(f"Fichier {OUTPUT_FF} non trouvé — vérifier FreeFEM.")
    exit(1)

df_ff = pd.read_csv(OUTPUT_FF, sep=r'\s+')
df_ff = df_ff.sort_values('x').reset_index(drop=True)
x_ff  = df_ff['x'].values
u_ff  = df_ff['u'].values
#print(f"\nFreeFEM : {len(x_ff)} nœuds chargés depuis '{OUTPUT_FF}'")
#print(df_ff.to_string(index=False))

# SOLUTION EXACTE 
# -EA u'' = 1, u(0)=0, u'(L)=0  →  u(x) = x*(L - x/2) / (EA)
x_exact = np.linspace(0, L, 500)
u_exact = x_exact * (L - x_exact / 2.0) / (E * A)

# ─── FIGURE 
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(x_exact, u_exact, '-',  color='gray',    lw=1.5, label='Solution exacte')
ax.plot(x_ff,    u_ff,    'o-', color='#1a6faf',  lw=2,  markersize=4,
        label=f'FreeFEM (N={N})')
ax.set_xlabel("x (m)")
ax.set_ylabel("u(x) (m)")
ax.set_title("Déplacement axial — Barre 1D ")
ax.legend()
ax.grid(True, alpha=0.3)

