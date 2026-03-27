"""
1D Bar Simulation - Traction Load - SOFA Scene File
"""
import json
import os
import sys
import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime

RESULTS_DIR = "results"


class DisplacementExporter(Sofa.Core.Controller):
    """Stores displacements in memory and writes them to file."""

    def __init__(self, dofs_node, output_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dofs_node   = dofs_node
        self.output_file = output_file
        self.x_initial   = None
        self.u_x         = None

    def onSimulationInitDoneEvent(self, event):
        self.x_initial = self.dofs_node.position.array().flatten().copy()

    def onAnimateEndEvent(self, event):
        x_final  = self.dofs_node.position.array().flatten()
        self.u_x = x_final - self.x_initial

        with open(self.output_file, 'w') as f:
            f.write("x_initial  x_final  u_x\n")
            for xi, xf, ui in zip(self.x_initial, x_final, self.u_x):
                f.write(f"{xi:.6f}  {xf:.6f}  {ui:.6f}\n")


def create_scene_args(rootNode, length, force, young_modulus, poisson_ratio, nx):
    requiredPlugins = [
        "Elasticity",
        "Sofa.Component.Constraint.Projective",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.MechanicalLoad",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Dynamic",
        "Sofa.Component.Visual",
        "Sofa.GL.Component.Rendering3D",
    ]

    rootNode.addObject('RequiredPlugin', pluginName=requiredPlugins)

    rootNode.addObject('DefaultAnimationLoop')
    rootNode.addObject('VisualStyle', displayFlags=["showBehaviorModels", "showForceFields"])

    with rootNode.addChild('Bar') as Bar:
        Bar.addObject('NewtonRaphsonSolver'
                    , name="newtonSolver"
                    , printLog=True
                    , warnWhenLineSearchFails=True
                    , maxNbIterationsNewton=1
                    , maxNbIterationsLineSearch=1
                    , lineSearchCoefficient=1
                    , relativeSuccessiveStoppingThreshold=0
                    , absoluteResidualStoppingThreshold=1e-7
                    , absoluteEstimateDifferenceThreshold=1e-12
                    , relativeInitialStoppingThreshold=1e-12
                    , relativeEstimateDifferenceThreshold=0
                    )

        Bar.addObject('SparseLDLSolver'
                    , name="linearSolver"
                    , template="CompressedRowSparseMatrixd")
        Bar.addObject('StaticSolver'
                    , name="staticSolver"
                    , newtonSolver="@newtonSolver"
                    , linearSolver="@linearSolver")

        # nx nodes at uniform spacing; node 0 fixed, node nx-1 loaded
        positions = [[i * length / (nx - 1)] for i in range(nx)]
        edges     = [[i, i + 1]              for i in range(nx - 1)]

        dofs = Bar.addObject('MechanicalObject'
                            , name="dofs"
                            , template="Vec1d"
                            , position=positions
                            , showObject=True
                            , showObjectScale=0.02)

        with Bar.addChild('edges') as Edges:
            Edges.addObject('EdgeSetTopologyContainer'
                            , name="topology"
                            , position="@../dofs.position"
                            , edges=edges)
            Edges.addObject('LinearSmallStrainFEMForceField'
                            , name="FEM"
                            , template="Vec1d"
                            , youngModulus=young_modulus
                            , poissonRatio=poisson_ratio
                            , topology="@topology")

        Bar.addObject('FixedProjectiveConstraint'
                      , indices="0")
        Bar.addObject('ConstantForceField'
                      , indices=f"{nx - 1}"
                      , forces=f"{force}"
                      , showArrowSize=1e-4)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    exporter = rootNode.addObject(
        DisplacementExporter(
            dofs_node   = dofs,
            output_file = os.path.join(RESULTS_DIR, "sofa_results.txt"),
            name        = "exportCtrl"
        )
    )

    return rootNode, exporter


def createScene(rootNode):
    with open("params.json") as f:
        cfg = json.load(f)
    create_scene_args(rootNode
                    , length=float(cfg["length"])
                    , force=float(cfg["force"])
                    , young_modulus=float(cfg["youngModulus"])
                    , poisson_ratio=float(cfg["poissonRatio"])
                    , nx=int(cfg["nx"]))
    return rootNode


def sofaRun(length, force, young_modulus, poisson_ratio, nx):
    root = Sofa.Core.Node("root")
    _, exporter = create_scene_args(root
                                  , length=length
                                  , force=force
                                  , young_modulus=young_modulus
                                  , poisson_ratio=poisson_ratio
                                  , nx=nx)
    Sofa.Simulation.init(root)
    Sofa.Simulation.animate(root, root.dt.value)
    return exporter.x_initial, exporter.u_x


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "params.json"
    with open(config_file) as f:
        cfg = json.load(f)

    sofaRun(length=float(cfg["length"])
            , force=float(cfg["force"])
            , young_modulus=float(cfg["youngModulus"])
            , poisson_ratio=float(cfg["poissonRatio"])
            , nx=int(cfg["nx"]))
