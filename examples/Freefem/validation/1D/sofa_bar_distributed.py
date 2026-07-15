"""
1D Bar Simulation - Distributed Load - SOFA Scene File

Physical case: bar fixed at x=0 (Dirichlet), free at x=L, subject to a
uniform distributed load q per unit length (e.g. self-weight).

Consistent nodal forces for a constant q on a uniform mesh of spacing h:
    F_0     = q*h/2   (absorbed by the Dirichlet reaction, value irrelevant)
    F_i     = q*h     for interior nodes
    F_(N-1) = q*h/2   (free end)
"""
import json
import os
import sys
import Sofa
import Sofa.Core
import Sofa.Simulation

RESULTS_DIR = "results"


class DisplacementExporter(Sofa.Core.Controller):
    
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
            f.write(f"{'x_initial':>12}  {'x_final':>12}  {'u_x':>12}\n")
            f.write("-" * 42 + "\n")
            for xi, xf, ui in zip(self.x_initial, x_final, self.u_x):
                f.write(f"{xi:12.6f}  {xf:12.6f}  {ui:12.6f}\n")


def _consistent_nodal_forces(q, h, nx):
    """Consistent (Galerkin) nodal forces for a constant distributed load q."""
    forces = [q * h] * nx
    forces[0]     = q * h / 2.0   
    forces[-1]    = q * h / 2.0   
    return [[f] for f in forces]


def create_scene_args(rootNode, length, q, young_modulus, poisson_ratio, nx):
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

    h = length / (nx - 1)

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

        positions = [[i * h] for i in range(nx)]
        edges     = [[i, i + 1] for i in range(nx - 1)]

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

        Bar.addObject('FixedProjectiveConstraint', indices="0")

        Bar.addObject('ConstantForceField'
                      , name="DistributedLoad"
                      , indices=list(range(nx))
                      , forces=_consistent_nodal_forces(q, h, nx))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    exporter = rootNode.addObject(
        DisplacementExporter(
            dofs_node   = dofs,
            output_file = os.path.join(RESULTS_DIR, "sofa_distributed_results.txt"),
            name        = "exportCtrl"
        )
    )

    return rootNode, exporter


def _default_params_path():
    """params_distributed.json next to this script, regardless of CWD."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "params_distributed.json")


def createScene(rootNode):
    with open(_default_params_path()) as f:
        cfg = json.load(f)
    create_scene_args(rootNode
                    , length=float(cfg["length"])
                    , q=float(cfg["q"])
                    , young_modulus=float(cfg["youngModulus"])
                    , poisson_ratio=float(cfg["poissonRatio"])
                    , nx=int(cfg["nx"]))
    return rootNode


def sofaRun(length, q, young_modulus, poisson_ratio, nx):
    root = Sofa.Core.Node("root")
    _, exporter = create_scene_args(root
                                  , length=length
                                  , q=q
                                  , young_modulus=young_modulus
                                  , poisson_ratio=poisson_ratio
                                  , nx=nx)
    Sofa.Simulation.init(root)
    Sofa.Simulation.animate(root, root.dt.value)
    return exporter.x_initial, exporter.u_x


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else _default_params_path()
    with open(config_file) as f:
        cfg = json.load(f)

    sofaRun(length=float(cfg["length"])
            , q=float(cfg["q"])
            , young_modulus=float(cfg["youngModulus"])
            , poisson_ratio=float(cfg["poissonRatio"])
            , nx=int(cfg["nx"]))