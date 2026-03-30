import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import tempfile
import os
import sys

# Parametres en commun Freefem et SOFA 
L        = 1.0
E        = 1e6
A        = 0.001
E_eff    = E * A       
F        = 1.0         
F_sofa   = F * L       
N        = 8           # nb éléments, nombre de noeuds N+1

WORK_DIR    = r"C:/Users/fatim/Desktop/SOFA_tuto"
OUTPUT_FF   = WORK_DIR + "/freefem_deplacement.txt"
OUTPUT_MSH  = WORK_DIR + "/maillage-ref.msh"
OUTPUT_SOFA = WORK_DIR + "/sofa_deplacement.txt"
RUNSOFA_EXE = r"C:/Project/sofa-build-v25.12/bin/Release/runSofa.exe"


#  FREEFEM

FREEFEM_CODE = (
    "real L = " + str(L) + ";\n"
    "real E = " + str(E) + ";\n"
    "real A = " + str(A) + ";\n"
    "int  n = " + str(N) + ";\n"
    "\n"
    "meshL Th = segment(n, [x*L]);\n"
    "\n"
    "// Export Gmsh v2\n"
    'ofstream msh("' + OUTPUT_MSH.replace("\\", "/") + '");\n'
    'msh << "$MeshFormat"   << endl;\n'
    'msh << "2.2 0 8"       << endl;\n'
    'msh << "$EndMeshFormat" << endl;\n'
    'msh << "$Nodes"  << endl;\n'
    "msh << Th.nv     << endl;\n"
    "for (int i = 0; i < Th.nv; i++) {\n"
    "    msh << i+1 << \" \" << Th(i).x << \" 0.0 0.0\" << endl;\n"
    "}\n"
    'msh << "$EndNodes" << endl;\n'
    'msh << "$Elements" << endl;\n'
    "msh << Th.nt + 2   << endl;\n"
    'msh << 1 << " 15 2 1 1 " << 1      << endl;\n'
    'msh << 2 << " 15 2 2 2 " << Th.nv  << endl;\n'
    "for (int i = 0; i < Th.nt; i++) {\n"
    '    msh << i+3 << " 1 2 0 1 " << Th[i][0]+1 << " " << Th[i][1]+1 << endl;\n'
    "}\n"
    'msh << "$EndElements" << endl;\n'
    'cout << "Maillage exporte" << endl;\n'
    "\n"
    "// Résolution EF\n"
    "fespace Vh(Th, P1);\n"
    "Vh u, v;\n"
    "problem Traction(u, v) =\n"
    "    int1d(Th)( E * A * dx(u) * dx(v) )\n"
    "    - int1d(Th)( 1.0 * v )\n"
    "    + on(1, u=0);\n"
    "Traction;\n"
    "\n"
    'ofstream file("' + OUTPUT_FF.replace("\\", "/") + '");\n'
    'file << "x u" << endl;\n'
    "for (int i = 0; i < Th.nv; i++) {\n"
    "    real xi = Th(i).x;\n"
    "    real ui = u(xi, 0);\n"
    "    file << xi << \" \" << ui << endl;\n"
    "}\n"
    'cout << "FreeFEM OK : " << Th.nv << " noeuds." << endl;\n'
)

print("=" * 60)
print("ÉTAPE 1 — FreeFEM")
print("=" * 60)
with tempfile.NamedTemporaryFile(mode='w', suffix='.edp',
                                  delete=False, encoding='utf-8') as f:
    f.write(FREEFEM_CODE)
    edp_file = f.name
try:
    res = subprocess.run(['FreeFem++', edp_file],
                         capture_output=True, text=True, encoding='utf-8')
    print(res.stdout)
    if res.returncode != 0:
        print("ERREUR FreeFEM :\n", res.stderr)
        sys.exit(1)
finally:
    os.unlink(edp_file)

df_ff = pd.read_csv(OUTPUT_FF, sep=r'\s+').sort_values('x').reset_index(drop=True)
x_ff  = df_ff['x'].values
u_ff  = df_ff['u'].values
print(f"  → {len(x_ff)} noeuds FreeFEM chargés")


print("\n" + "=" * 60)
print("ÉTAPE 2 — SOFA")
print("=" * 60)


sofa_lines = [
    "import Sofa",
    "import Sofa.Core",
    "import numpy as np",
    "",
    "OUTPUT_SOFA = r'" + OUTPUT_SOFA + "'",
    "OUTPUT_MSH  = r'" + OUTPUT_MSH  + "'",
    "F_SOFA = " + str(F_sofa),
    "E_EFF  = " + str(E_eff),
    "L      = " + str(L),
    "",
    "class ExportController(Sofa.Core.Controller):",
    "    def __init__(self, dofs_node, *args, **kwargs):",
    "        super().__init__(*args, **kwargs)",
    "        self.dofs_node = dofs_node",
    "        self.exported  = False",
    "",
    "    def onSimulationInitDoneEvent(self, event):",
    "        pos = self.dofs_node.position.array()",
    "        self.x_initial = pos[:, 0].copy()",
    "        print('[SOFA] Init OK,', len(self.x_initial), 'noeuds')",
    "",
    "    def onAnimateEndEvent(self, event):",
    "        if self.exported:",
    "            return",
    "        pos   = self.dofs_node.position.array()",
    "        u_x   = pos[:, 0] - self.x_initial",
    "        order = np.argsort(self.x_initial)",
    "        x0    = self.x_initial[order]",
    "        ux    = u_x[order]",
    "        with open(OUTPUT_SOFA, 'w') as f:",
    "            f.write('x_initial  u_x\\n')",
    "            for xi, ui in zip(x0, ux):",
    "                f.write(f'{xi:.8f}  {ui:.8f}\\n')",
    "        print('[SOFA] Export OK ->', OUTPUT_SOFA)",
    "        print('[SOFA] u_x(L) =', round(ux[-1], 6))",
    "        self.exported = True",
    "",
    "def createScene(rootNode):",
    "    rootNode.gravity = [0, 0, 0]",
    "    rootNode.dt      = 1.0",
    "    rootNode.addObject('DefaultAnimationLoop')",
    "    rootNode.addObject('VisualStyle', displayFlags='showBehaviorModels')",
    "",
    "    plugins = rootNode.addChild('plugins')",
    "    for p in ['Elasticity',",
    "              'Sofa.Component.Constraint.Projective',",
    "              'Sofa.Component.Engine.Select',",
    "              'Sofa.Component.IO.Mesh',",
    "              'Sofa.Component.LinearSolver.Direct',",
    "              'Sofa.Component.MechanicalLoad',",
    "              'Sofa.Component.ODESolver.Backward',",
    "              'Sofa.Component.StateContainer',",
    "              'Sofa.Component.Topology.Container.Dynamic',",
    "              'Sofa.Component.Visual',",
    "              'Sofa.GL.Component.Rendering3D']:",
    "        plugins.addObject('RequiredPlugin', name=p)",
    "",
    "    bar = rootNode.addChild('bar')",
    "    bar.addObject('NewtonRaphsonSolver', name='newton',",
    "                  maxNbIterationsNewton=1, maxNbIterationsLineSearch=1,",
    "                  lineSearchCoefficient=1,",
    "                  relativeSuccessiveStoppingThreshold=0,",
    "                  absoluteResidualStoppingThreshold=1e-7,",
    "                  absoluteEstimateDifferenceThreshold=1e-12,",
    "                  relativeInitialStoppingThreshold=1e-12,",
    "                  relativeEstimateDifferenceThreshold=0)",
    "    bar.addObject('SparseLDLSolver', name='ldl',",
    "                  template='CompressedRowSparseMatrixd')",
    "    bar.addObject('StaticSolver', newtonSolver='@newton', linearSolver='@ldl')",
    "",
    "    bar.addObject('MeshGmshLoader', name='loader', filename=OUTPUT_MSH)",
    "    dofs = bar.addObject('MechanicalObject', template='Vec3d',",
    "                         name='dofs', src='@loader',",
    "                         showObject=True, showObjectScale=0.02)",
    "",
    "    edges = bar.addChild('edges')",
    "    edges.addObject('EdgeSetTopologyContainer', name='topo', src='@../loader')",
    "    edges.addObject('EdgeSetTopologyModifier')",
    "    edges.addObject('LinearSmallStrainFEMForceField', name='FEM',",
    "                    youngModulus=E_EFF, poissonRatio=0.0, topology='@topo')",
    "",
    "    bar.addObject('BoxROI', name='fix_roi',",
    "                  box='-0.01 -1 -1  0.01 1 1', drawBoxes=False)",
    "    bar.addObject('FixedProjectiveConstraint', indices='@fix_roi.indices')",
    "",
    "    bar.addObject('BoxROI', name='load_roi',",
    "                  box=str(L-0.01)+' -1 -1  '+str(L+0.01)+' 1 1',",
    "                  drawBoxes=False)",
    "    bar.addObject('ConstantForceField',",
    "                  indices='@load_roi.indices',",
    "                  forces=str(F_SOFA)+' 0 0', showArrowSize=1e-4)",
    "",
    "    rootNode.addObject(ExportController(dofs_node=dofs, name='exportCtrl'))",
    "    return rootNode",
]

sofa_scene_file = WORK_DIR + "/_scene_temp.py"
with open(sofa_scene_file, 'w', encoding='utf-8') as f:
    f.write("\n".join(sofa_lines))

print(f"  Scène écrite → {sofa_scene_file}")


res_sofa = subprocess.run(
    [RUNSOFA_EXE, '-l', 'SofaPython3', '--nb-iterations', '1', '-g', 'imgui', sofa_scene_file], #-g batch sans interface graphique en sofa 
    capture_output=True, text=True,
    cwd=WORK_DIR,
    env=os.environ.copy()
)

print(res_sofa.stdout[-4000:])
if res_sofa.returncode != 0:
    print("ERREUR SOFA :\n", res_sofa.stderr[-2000:])
    sys.exit(1)


if not os.path.exists(OUTPUT_SOFA):
    print(f"ERREUR : {OUTPUT_SOFA} non trouvé — le contrôleur SOFA n'a pas exporté.")
    sys.exit(1)

os.remove(sofa_scene_file)

df_sofa = pd.read_csv(OUTPUT_SOFA, sep=r'\s+').sort_values('x_initial').reset_index(drop=True)
x_sofa  = df_sofa['x_initial'].values
u_sofa  = df_sofa['u_x'].values
print(f"  → {len(x_sofa)} noeuds SOFA chargés")


#======= Figures de comparaison 

u_ff_interp = np.interp(x_sofa, x_ff, u_ff)
diff        = u_sofa - u_ff_interp
norme_L2    = np.linalg.norm(diff)
norme_rel   = norme_L2 / np.linalg.norm(u_ff_interp)

print(f"  ||u_sofa - u_ff||_2         = {norme_L2:.4e}")
print(f"  ||u_sofa - u_ff||_2 relatif = {norme_rel:.4e}")
print(f"  max |u_sofa - u_ff|         = {np.max(np.abs(diff)):.4e}")


x_exact = np.linspace(0, L, 500)
u_exact = x_exact * (L - x_exact / 2.0) / (E * A)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(x_exact, u_exact, '-',   color='gray',    lw=1.5, label='Exacte')
ax.plot(x_ff,    u_ff,    'o-',  color='#1a6faf',  lw=2,  ms=4, label=f'FreeFEM (N={N})')
ax.plot(x_sofa,  u_sofa,  's--', color='#e05c2a',  lw=2,  ms=4, label='SOFA (même maillage)')
ax.set_xlabel("x (m)"); ax.set_ylabel("u(x) (m)")
ax.set_title("Déplacement axial — Barre 1D")
ax.legend(); ax.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(x_sofa, np.abs(diff), 'o-', color='#c0392b', lw=1.5, ms=4)
ax2.set_xlabel("x (m)"); ax2.set_ylabel("|u_sofa − u_ff|  (m)")
ax2.set_title(f"Erreur ponctuelle\n||·||₂ = {norme_L2:.2e}  (rel. {norme_rel:.2e})")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(WORK_DIR + "/comparaison_sofa_freefem.png", dpi=150)
plt.show()
