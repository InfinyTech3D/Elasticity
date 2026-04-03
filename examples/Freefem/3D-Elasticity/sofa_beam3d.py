"""
3D Beam Simulation  - SOFA Scene
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
    """Stores 3D nodal displacements and writes them to file."""

    def __init__(self, dofs_node, output_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dofs_node   = dofs_node
        self.output_file = output_file
        self.pos_initial = None
        self.u           = None

    def onSimulationInitDoneEvent(self, event):
        self.pos_initial = self.dofs_node.position.array()[:, :3].copy()

    def onAnimateEndEvent(self, event):
        pos_final = self.dofs_node.position.array()[:, :3]
        self.u    = pos_final - self.pos_initial

        with open(self.output_file, 'w') as f:
            f.write(f"{'x':>12}  {'y':>12} {'z':>12}   {'ux':>12}  {'uy':>12} {'uz':>12} \n")
            f.write("-" * 54 + "\n")
            for (x, y,z), (ux, uy, uz) in zip(self.pos_initial, self.u):
                f.write(f"{x:12.6f}  {y:12.6f} {z:12.6f} {ux:12.6f}  {uy:12.6f} {uz:12.6f} \n")


# FIXED: added `width` parameter (was used in BoxROI but never defined)
# FIXED: removed `formulation` parameter (no longer needed in 3D-only code)
def create_scene_args(rootNode
                    , mesh_file
                    , height
                    , width        # ADDED: needed for BoxROI y-extent
                    , young_modulus
                    , poisson_ratio
                    , rho
                    , gravity):
    dim_template = "Vec3d"

    requiredPlugins = [
        "Elasticity",
        "Sofa.Component.Constraint.Projective",
        "Sofa.Component.Engine.Select",
        "Sofa.Component.IO.Mesh",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.Mass",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Dynamic",
        "Sofa.Component.Visual",
        "Sofa.GL.Component.Rendering3D",
    ]

    rootNode.addObject('RequiredPlugin', pluginName=requiredPlugins)
    rootNode.gravity = [0, 0, -gravity]
    rootNode.dt      = 1.0

    rootNode.addObject('DefaultAnimationLoop')
    rootNode.addObject('VisualStyle', displayFlags=["showBehaviorModels", "showForceFields", "showWireframe"])

    with rootNode.addChild('Beam') as Beam:
        Beam.addObject('NewtonRaphsonSolver'
                     , name="newtonSolver"
                     , printLog=False
                     , warnWhenLineSearchFails=True
                     , maxNbIterationsNewton=1
                     , maxNbIterationsLineSearch=1
                     , lineSearchCoefficient=1
                     , relativeSuccessiveStoppingThreshold=0
                     , absoluteResidualStoppingThreshold=1e-7
                     , absoluteEstimateDifferenceThreshold=1e-12
                     , relativeInitialStoppingThreshold=1e-12
                     , relativeEstimateDifferenceThreshold=0)
        Beam.addObject('SparseLDLSolver'
                     , name="linearSolver"
                     , template="CompressedRowSparseMatrixd")
        Beam.addObject('StaticSolver'
                     , name="staticSolver"
                     , newtonSolver="@newtonSolver"
                     , linearSolver="@linearSolver")

        Beam.addObject('MeshGmshLoader'
                     , name="loader"
                     , filename=mesh_file)
        dofs = Beam.addObject('MechanicalObject'
                            , name="dofs"
                            , template=dim_template
                            , src="@loader")

        with Beam.addChild('tetrahedra') as Tetrahedra:
            Tetrahedra.addObject('TetrahedronSetTopologyContainer'
                              , name="topology"
                              , src="@../loader")
            Tetrahedra.addObject('TetrahedronSetTopologyModifier')
            Tetrahedra.addObject('TetrahedronSetGeometryAlgorithms'
                              , template=dim_template
                              , drawTriangles=True)
            Tetrahedra.addObject('MeshMatrixMass'
                              , name="mass"
                              , template=f"{dim_template},{dim_template}"
                              , massDensity=rho
                              , topology="@topology")
            Tetrahedra.addObject('LinearSmallStrainFEMForceField'
                              , name="FEM"
                              , template=dim_template
                              , youngModulus=young_modulus
                              , poissonRatio=poisson_ratio
                              , topology="@topology")

        # Fix the left wall (x=0, label 4 in gmsh matching FreeFEM's on(4,...))
        eps = 1e-5
        Beam.addObject('BoxROI'
                     , name="fixed_roi"
                     , template=dim_template
                     , box=[-eps, -eps, -eps, eps, width + eps, height + eps]  # FIXED: width now defined
                     , drawBoxes=True)
        Beam.addObject('FixedProjectiveConstraint'
                     , template=dim_template
                     , indices="@fixed_roi.indices")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    exporter = rootNode.addObject(
        DisplacementExporter(
            dofs_node   = dofs,
            output_file = os.path.join(RESULTS_DIR, "sofa_3d_results.txt"),  # FIXED: removed formulation (no longer a parameter)
            name        = "exportCtrl"
        )
    )

    return rootNode, exporter


def createScene(rootNode):
    with open("params.json") as f:
        cfg = json.load(f)
    create_scene_args(rootNode
                    , mesh_file    = os.path.join(RESULTS_DIR, cfg.get("meshfile", "beam3d.msh"))
                    , height       = float(cfg["height"])
                    , width        = float(cfg["width"])   # ADDED: read width from params.json
                    , young_modulus= float(cfg["youngModulus"])
                    , poisson_ratio= float(cfg["poissonRatio"])
                    , rho          = float(cfg["rho"])
                    , gravity      = float(cfg["gravity"])
                    # FIXED: removed formulation= (no longer a parameter)
    )
    return rootNode


# FIXED: removed `formulation` parameter, added `width`
def sofaRun(height, width, young_modulus, poisson_ratio, rho, gravity, mesh_file):
    root = Sofa.Core.Node("root")
    _, exporter = create_scene_args(root
                                  , mesh_file    = mesh_file
                                  , height       = height
                                  , width        = width   # ADDED
                                  , young_modulus= young_modulus
                                  , poisson_ratio= poisson_ratio
                                  , rho          = rho
                                  , gravity      = gravity
                                  # FIXED: removed formulation=
                                  )
    Sofa.Simulation.init(root)
    Sofa.Simulation.animate(root, root.dt.value)
    return exporter.pos_initial, exporter.u


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "params.json"
    with open(config_file) as f:
        cfg = json.load(f)

    sofaRun(height        = float(cfg["height"])
          , width         = float(cfg["width"])        # ADDED
          , young_modulus = float(cfg["youngModulus"])
          , poisson_ratio = float(cfg["poissonRatio"])
          , rho           = float(cfg["rho"])
          , gravity       = float(cfg["gravity"])
          , mesh_file     = os.path.join(RESULTS_DIR, cfg.get("meshfile", "beam3d.msh"))
          # FIXED: removed formulation=
          )