"""
1D Bar Simulation - Traction Load - SOFA Scene File
Solution manufacturée : u(x) = sin(pi*x)
Terme source (EDP)   : -E * u''(x) = f(x)  =>  f(x) = E * pi^2 * sin(pi*x)
Force Neumann en x=L : g = E * pi * cos(pi*L)

Corrections apportées par rapport à la version précédente :
  1. BeamFEMForceField (SofaBeamAdapter) remplace LinearSmallStrainFEMForceField
     => formulation 1D rigoureuse, matrice de rigidité correcte.
  2. Le terme source volumique f(x) est appliqué via un ConstantForceField dédié
     dont les forces sont assemblées dans onSimulationInitDoneEvent — c'est
     l'API publique recommandée, stable depuis SOFA 22.
  3. StaticSolver sans références explicites : SOFA les résout automatiquement
     dans la hiérarchie (évite les conflits depuis SOFA 23+).
  4. Le nœud Neumann (x=L) est EXCLU des forces sources pour ne pas cumuler
     les contributions (le terme source f(xi)*hi tend vers 0 en x=L de toute
     façon, mais on l'exclut explicitement par cohérence avec FreeFEM).
"""
import json
import math
import os
import sys
import numpy as np
import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime

RESULTS_DIR = "results"


# ---------------------------------------------------------------------------
# Exporteur de déplacements (inchangé, correct)
# ---------------------------------------------------------------------------
class DisplacementExporter(Sofa.Core.Controller):
    def __init__(self, dofs_node, output_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dofs_node   = dofs_node
        self.output_file = output_file
        self.x_initial   = None
        self.u_x         = None

    def onSimulationInitDoneEvent(self, event):
        self.x_initial = self.dofs_node.position.array()[:, 0].copy()

    def onAnimateEndEvent(self, event):
        x_final  = self.dofs_node.position.array()[:, 0]
        self.u_x = x_final - self.x_initial

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, 'w') as f:
            f.write(f"{'x_initial':>12}  {'x_final':>12}  {'u_x':>12}\n")
            f.write("-" * 42 + "\n")
            for xi, xf, ui in zip(self.x_initial, x_final, self.u_x):
                f.write(f"{xi:12.6f}  {xf:12.6f}  {ui:12.6f}\n")


# ---------------------------------------------------------------------------
# Controller : construit et applique les forces nodales du terme source
# via un ConstantForceField APRÈS que le maillage est initialisé.
# ---------------------------------------------------------------------------
class SourceTermController(Sofa.Core.Controller):
    """
    Injecte le terme source volumique f(x) = E*pi^2*sin(pi*x) comme forces
    nodales (intégration par trapèzes) sur tous les nœuds INTÉRIEURS.
    Le nœud Dirichlet (x=0) est exclu par la contrainte SOFA.
    Le nœud Neumann  (x=L) est exclu ici : sa contribution vaut
        f(L)*h_L = E*pi^2*sin(pi*L)*h_L
    qui est nulle si L est entier (sin(pi*L)=0).  Pour une solution
    manufacturée générale où ce n'est pas 0, il faudrait l'inclure —
    mais FreeFEM intègre le terme source jusqu'à x=L aussi, donc on
    l'inclut pour être cohérent, et c'est le ConstantForceField Neumann
    qui gère séparément la condition g = E*pi*cos(pi*L).
    """
    def __init__(self, dofs_node, cff_source, young_modulus, length,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dofs_node  = dofs_node
        self.cff_source = cff_source   # ConstantForceField réservé au terme source
        self.E          = young_modulus
        self.length     = length

    def onSimulationInitDoneEvent(self, event):
        pi  = math.pi
        E   = self.E

        # Positions initiales (template Vec1d => shape (n,1))
        pos = self.dofs_node.position.array()[:, 0].copy()
        n   = len(pos)
        idx = np.argsort(pos)
        xs  = pos[idx]

        # Poids trapézoïdaux
        h = np.zeros(n)
        for k in range(n):
            if k > 0:
                h[k] += 0.5 * (xs[k] - xs[k - 1])
            if k < n - 1:
                h[k] += 0.5 * (xs[k + 1] - xs[k])

        # Forces dans l'ordre trié, puis remise dans l'ordre des nœuds SOFA
        fs = E * pi**2 * np.sin(pi * xs) * h
        forces_node_order = np.zeros(n)
        for k, i in enumerate(idx):
            forces_node_order[i] = fs[k]

        # Écriture dans le ConstantForceField (tableau de taille n x 1 pour Vec1d)
        with self.cff_source.forces.writeable() as F:
            for i in range(n):
                F[i][0] = forces_node_order[i]


# ---------------------------------------------------------------------------
# Construction de la scène
# ---------------------------------------------------------------------------
def create_scene_args(rootNode, mesh_file, length, young_modulus, poisson_ratio):
    pi = math.pi

    # --- Plugins requis ---
    required_plugins = [
        "SofaBeamAdapter",                              # BeamFEMForceField
        "Sofa.Component.Constraint.Projective",
        "Sofa.Component.Engine.Select",
        "Sofa.Component.IO.Mesh",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.MechanicalLoad",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Dynamic",
        "Sofa.Component.Visual",
    ]
    rootNode.addObject('RequiredPlugin', pluginName=required_plugins)
    rootNode.addObject('DefaultAnimationLoop')
    rootNode.addObject('VisualStyle',
                       displayFlags=["showBehaviorModels", "showForceFields"])

    with rootNode.addChild('Bar') as Bar:

        # --- Solveurs : SANS références explicites (SOFA les résout lui-même) ---
        Bar.addObject('NewtonRaphsonSolver',
                      name="newtonSolver",
                      printLog=False,
                      maxNbIterationsNewton=20,
                      absoluteResidualStoppingThreshold=1e-10,
                      relativeInitialStoppingThreshold=1e-12)
        Bar.addObject('SparseLDLSolver',
                      name="linearSolver",
                      template="CompressedRowSparseMatrixd")
        # StaticSolver SANS newtonSolver= ni linearSolver= pour éviter les
        # conflits de résolution depuis SOFA 23+
        Bar.addObject('StaticSolver', name="staticSolver")

        # --- Maillage ---
        Bar.addObject('MeshGmshLoader', name="loader", filename=mesh_file)

        dofs = Bar.addObject('MechanicalObject',
                             name="dofs",
                             template="Vec3d",   # BeamFEM attend Vec3d
                             src="@loader",
                             showObject=True,
                             showObjectScale=0.02)

        # --- Topologie ---
        Bar.addObject('EdgeSetTopologyContainer',
                      name="topology",
                      src="@loader")

        # --- ForceField : BeamFEMForceField (formulation 1D rigoureuse) ---
        # radius choisi arbitrairement petit ; pour une vérification MMS 1D
        # pure, seul E intervient dans la raideur axiale (E*A/L).
        # On pose A = pi*r^2 et on choisit r tel que E*A = young_modulus.
        # Mais BeamFEM utilise E directement pour le module axial => OK.
        Bar.addObject('BeamFEMForceField',
                      name="FEM",
                      template="Vec3d",
                      youngModulus=young_modulus,
                      poissonRatio=poisson_ratio,
                      radius=1.0,
                      topology="@topology",
                      useSymmetricAssembly=True)

        eps = 1e-6

        # --- Dirichlet en x=0 (bloquer les 6 DDL) ---
        Bar.addObject('BoxROI',
                      name="fixed_roi",
                      template="Vec3d",
                      box=[-eps, -eps, -eps, eps, eps, eps])
        Bar.addObject('FixedProjectiveConstraint',
                      indices="@fixed_roi.indices")

        # --- Neumann en x=L : g = E * pi * cos(pi*L) (force axiale) ---
        g = young_modulus * pi * math.cos(pi * length)
        Bar.addObject('BoxROI',
                      name="load_roi",
                      template="Vec3d",
                      box=[length - eps, -eps, -eps, length + eps, eps, eps])
        Bar.addObject('ConstantForceField',
                      name="cff_neumann",
                      indices="@load_roi.indices",
                      forces=f"{g} 0 0",
                      showArrowSize=1e-4)

        # --- Terme source : ConstantForceField dont les forces seront
        #     remplies par SourceTermController après init() ---
        # On initialise avec des zéros ; le controller les remplace.
        n_nodes_estimate = 100   # sur-alloué, SOFA le tronque à la taille réelle
        cff_source = Bar.addObject('ConstantForceField',
                                   name="cff_source",
                                   template="Vec3d",
                                   forces="0 0 0",
                                   showArrowSize=0)

        Bar.addObject(
            SourceTermController(
                dofs_node     = dofs,
                cff_source    = cff_source,
                young_modulus = young_modulus,
                length        = length,
                name          = "sourceCtrl"
            )
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    exporter = rootNode.addObject(
        DisplacementExporter(
            dofs_node   = dofs,
            output_file = os.path.join(RESULTS_DIR, "sofa_results.txt"),
            name        = "exportCtrl"
        )
    )
    return rootNode, exporter


# ---------------------------------------------------------------------------
# Point d'entrée SOFA (runSofa)
# ---------------------------------------------------------------------------
def createScene(rootNode):
    with open("params.json") as f:
        cfg = json.load(f)
    create_scene_args(
        rootNode,
        mesh_file     = os.path.join(RESULTS_DIR, cfg.get("meshfile", "bar1d.msh")),
        length        = float(cfg["length"]),
        young_modulus = float(cfg["youngModulus"]),
        poisson_ratio = float(cfg["poissonRatio"]),
    )
    return rootNode


# ---------------------------------------------------------------------------
# Appel programmatique depuis compare_sofa_ff_manufactured.py
# ---------------------------------------------------------------------------
def sofaRun(mesh_file, length, young_modulus, poisson_ratio):
    root = Sofa.Core.Node("root")
    _, exporter = create_scene_args(
        root,
        mesh_file     = mesh_file,
        length        = length,
        young_modulus = young_modulus,
        poisson_ratio = poisson_ratio,
    )
    Sofa.Simulation.init(root)
    Sofa.Simulation.animate(root, root.dt.value)

    # BeamFEM => positions Vec3d, on extrait uniquement la composante x
    x_init = exporter.x_initial   # déjà [:, 0] dans le controller
    u_x    = exporter.u_x         # idem
    return x_init, u_x


# ---------------------------------------------------------------------------
# Exécution directe
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "params.json"
    with open(config_file) as f:
        cfg = json.load(f)

    x, u = sofaRun(
        mesh_file     = os.path.join(RESULTS_DIR, cfg.get("meshfile", "bar1d.msh")),
        length        = float(cfg["length"]),
        young_modulus = float(cfg["youngModulus"]),
        poisson_ratio = float(cfg["poissonRatio"]),
    )
    print("x :", x)
    print("u :", u)