import numpy as np
import pandas as pd
import subprocess
import tempfile
import os 
import time 

# ========== PARAMÈTRES À DÉFINIR ==========
E = 21.5          # Module d'Young (GPa)
NU = 0.29         # Coefficient de Poisson
GRAVITY = -0.05   # Poids propre (N/m³)
N_LONG = 140      # Nombre d'éléments sur la longueur
N_HAUT = 35       # Nombre d'éléments sur la hauteur
OUTPUT_FILE = "freefem_results.txt"   

FREEFEM_CODE = """
real E = {E};
real sigma = {NU};
real gravity = {GRAVITY};
int n = {N_LONG};
int m = {N_HAUT};

border a(t=2, 0){{x=0;   y=t;    label=1;}}
border b(t=0,10){{x=t;   y=0;    label=2;}}
border c(t=0, 2){{x=10;  y=t;    label=3;}}
border d(t=0,10){{x=10-t; y=2;   label=4;}}

mesh th = buildmesh(b(n) + c(m) + d(n) + a(m));

fespace Vh(th, [P1, P1]);
Vh [uu, vv], [w, s];

real sqrt2 = sqrt(2.);
macro epsilon(u1,u2) [dx(u1), dy(u2), (dy(u1)+dx(u2))/sqrt2] // EOM
macro div(u,v) (dx(u)+dy(v)) // EOM

real mu = E / (2*(1+sigma));
real lambda = E*sigma / ((1+sigma)*(1-2*sigma));

solve Elasticity([uu,vv],[w,s], solver=sparsesolver)
    = int2d(th)(
          lambda * div(w,s) * div(uu,vv)
        + 2.*mu * (epsilon(w,s)' * epsilon(uu,vv))
    )
    - int2d(th)(gravity*s)
    + on(1, uu=0, vv=0);

fespace Wh(th, P1);
Wh sigmaxx, sigmayy, sigmaxy, sigmavm;
sigmaxx = lambda*(dx(uu)+dy(vv)) + 2*mu*dx(uu);
sigmayy = lambda*(dx(uu)+dy(vv)) + 2*mu*dy(vv);
sigmaxy = mu*(dy(uu)+dx(vv));
sigmavm = sqrt(sigmaxx^2 + sigmayy^2 - sigmaxx*sigmayy + 3*sigmaxy^2);

cout << "Max uy = " << vv[].linfty << endl;
cout << "Max ux = " << uu[].linfty << endl;
cout << "sigmaxx max = " << sigmaxx[].max << endl;
cout << "sigmavm max = " << sigmavm[].max << endl;


{{
    ofstream fout("{OUTPUT_FILE}");
    fout << "x y ux uy sigmaxx sigmayy sigmaxy sigmavm" << endl;
    for(int i=0; i<th.nv; i++){{
        real xi = th(i).x;
        real yi = th(i).y;
        fout << xi << " "
             << yi << " "
             << uu(xi, yi) << " "
             << vv(xi, yi) << " "
             << sigmaxx(xi, yi) << " "
             << sigmayy(xi, yi) << " "
             << sigmaxy(xi, yi) << " "
             << sigmavm(xi, yi) << endl;
    }}
}}
""".format(E=E, NU=NU, GRAVITY=GRAVITY,
           N_LONG=N_LONG, N_HAUT=N_HAUT,
           OUTPUT_FILE=OUTPUT_FILE)

# ========== EXÉCUTION ==========
print("Lancement FreeFEM...")
with tempfile.NamedTemporaryFile(mode='w', suffix='.edp', delete=False) as f:
    f.write(FREEFEM_CODE)
    temp_file = f.name

try:
    result = subprocess.run(['FreeFem++', temp_file], 
                            capture_output=True, 
                            text=True,
                            encoding='utf-8')
    print(result.stdout)
    if result.stderr:
        print("Erreurs:")
        print(result.stderr)
finally:
    os.unlink(temp_file)

# ========== LECTURE DES RÉSULTATS ==========
if os.path.exists(OUTPUT_FILE):
    
    df = pd.read_csv(OUTPUT_FILE, sep='\s+')
    print("\n=== RÉSULTATS ===")
    print(f"Max uy = {df['uy'].max()}")
    print(f"Max ux = {df['ux'].max()}")
    print(f"Max sigmaxx = {df['sigmaxx'].max()}")
    print(f"Max sigmavm = {df['sigmavm'].max()}")
    
 
else:
    print(f"Fichier {OUTPUT_FILE} non trouvé")


 