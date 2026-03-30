"""
1D Bar Simulation - Traction Load
"""
import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime
import numpy as np

# Exporter les resultats 
class DisplacementExporter(Sofa.Core.Controller):
    """Récupère les déplacements après convergence et les écrit dans un .txt"""

    def __init__(self, dofs_node, output_file="sofa_displacement.txt", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dofs_node   = dofs_node
        self.output_file = output_file
        self.exported    = False  

    def onAnimateEndEvent(self, event):
        """Appelé automatiquement par SOFA à la fin de chaque pas de temps"""
        if self.exported:
            return

        
        pos = self.dofs_node.position.array()   
        
        x_deform = pos[:, 0]
        u_x      = x_deform - self.x_initial  

        
        order = np.argsort(self.x_initial)
        x0_sorted = self.x_initial[order]
        ux_sorted = u_x[order]

        with open(self.output_file, 'w') as f:
            f.write("x_initial  u_x\n")
            for xi, ui in zip(x0_sorted, ux_sorted):
                f.write(f"{xi:.6f}  {ui:.6f}\n")

        print(f"[DisplacementExporter] Déplacements exportés → {self.output_file}")
        print(f"  u_x(L) = {ux_sorted[-1]:.6f}  (attendu : 1.0)")
        self.exported = True

    def onSimulationInitDoneEvent(self, event):
        """Stocke les positions initiales au démarrage"""
        pos = self.dofs_node.position.array()
        self.x_initial = pos[:, 0].copy()
        print(f"[DisplacementExporter] {len(self.x_initial)} noeuds initialisés.")


# creation de la scene 
def createScene(rootNode):
    L     = 1.0
    E_eff = 1000.0
    F     = 1000.0
    N     = 9

    rootNode.addObject('DefaultAnimationLoop')
    rootNode.addObject('VisualStyle',
                       displayFlags="showBehaviorModels showForceFields showWireframe")

    plugins = rootNode.addChild('plugins')
    for plugin in [
        "Elasticity",
        "Sofa.Component.Constraint.Projective",
        "Sofa.Component.Engine.Select",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.MechanicalLoad",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Dynamic",
        "Sofa.Component.Topology.Container.Grid",
        "Sofa.Component.Visual",
        "Sofa.GL.Component.Rendering3D",
    ]:
        plugins.addObject('RequiredPlugin', name=plugin)

    bar = rootNode.addChild('bar')

    bar.addObject('NewtonRaphsonSolver',
                  name="newtonSolver",
                  printLog=True,
                  warnWhenLineSearchFails=True,
                  maxNbIterationsNewton=1,
                  maxNbIterationsLineSearch=1,
                  lineSearchCoefficient=1,
                  relativeSuccessiveStoppingThreshold=0,
                  absoluteResidualStoppingThreshold=1e-7,
                  absoluteEstimateDifferenceThreshold=1e-12,
                  relativeInitialStoppingThreshold=1e-12,
                  relativeEstimateDifferenceThreshold=0)

    bar.addObject('SparseLDLSolver',
                  name="linearSolver",
                  template="CompressedRowSparseMatrixd")

    bar.addObject('StaticSolver',
                  name="staticSolver",
                  newtonSolver="@newtonSolver",
                  linearSolver="@linearSolver")

    bar.addObject('RegularGridTopology',
                  name="grid",
                  min="0 0 0",
                  max=f"{L} 0 0",
                  n=f"{N} 1 1")

    dofs = bar.addObject('MechanicalObject',
                         template="Vec3d",
                         name="dofs",
                         showObject=True,
                         showObjectScale=0.02)

    edges = bar.addChild('edges')
    edges.addObject('EdgeSetTopologyContainer',
                    name="topology",
                    position="@../dofs.position",
                    edges="@../grid.edges")
    edges.addObject('LinearSmallStrainFEMForceField',
                    name="FEM",
                    youngModulus=E_eff,
                    poissonRatio=0.0,
                    topology="@topology")

    bar.addObject('BoxROI',
                  name="fixed_roi",
                  box="-0.01 -1 -1  0.01 1 1",
                  drawBoxes=False)
    bar.addObject('FixedProjectiveConstraint',
                  indices="@fixed_roi.indices")

    bar.addObject('BoxROI',
                  name="load_roi",
                  box=f"{L-0.01} -1 -1  {L+0.01} 1 1",
                  drawBoxes=False)
    bar.addObject('ConstantForceField',
                  indices="@load_roi.indices",
                  forces=f"{F} 0 0",
                  showArrowSize=1e-4)

    
    rootNode.addObject(
        DisplacementExporter(
            dofs_node   = dofs,
            output_file = "sofa_displacement.txt",
            name        = "exportCtrl"
        )
    )

    return rootNode
