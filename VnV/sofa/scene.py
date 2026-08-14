"""Generic SOFA scene assembly."""

import SofaRuntime

from .solvers import add_solvers
from .prefabs.beam import ElasticBeam


PLUGINS = [
    "Sofa.Component.SolidMechanics.FEM.Elastic",
    "Sofa.Component.Constraint.Projective",
    "Sofa.Component.Engine.Select",
    "Sofa.Component.Mapping.Linear",
    "Sofa.Component.MechanicalLoad",
    "Sofa.Component.LinearSolver.Direct",
    "Sofa.Component.LinearSolver.Iterative",
    "Sofa.Component.LinearSolver.Preconditioner",
    "Sofa.Component.LinearSystem",
    "Sofa.Component.ODESolver.Backward",
    "Sofa.Component.StateContainer",
    "Sofa.Component.Topology.Container.Grid",
    "Sofa.Component.Topology.Container.Dynamic",
    "Sofa.Component.Topology.Mapping",
]


def load_plugins():
    """Load the scene's plugins up front; a caller that prints tables calls this before printing.

    PluginManager logs one info line per plugin the first time it is loaded, and the first scene
    build would otherwise flush all of them into the middle of the first table. Loading an already
    loaded plugin returns early and logs nothing, so the RequiredPlugin in `build` stays silent
    afterwards and the scene remains self-contained for runSofa.
    """
    for plugin in PLUGINS:
        SofaRuntime.importPlugin(plugin)


class Scene:
    """Assembles a SOFA scene: the elastic beam, then boundary conditions, then solvers."""

    def __init__(self, geometry, material, force_field, element, resolution, solvers):
        self.geometry = geometry
        self.material = material
        self.force_field = force_field
        self.element = element
        self.resolution = resolution
        self.solvers = solvers

    def apply_bcs(self, beam):
        """Boundary conditions — filled by the verification or validation suite."""

    def build(self, root):
        root.addObject('RequiredPlugin', pluginName=PLUGINS)
        root.addObject('DefaultAnimationLoop')
        resolution = [self.resolution[i] if i < len(self.resolution) else 1 for i in range(3)]
        params = dict(name='beam',
                      extents=self.geometry.extents,
                      resolution=resolution,
                      spatialDimensions=self.geometry.spatial_dimensions,
                      element=self.element,
                      youngModulus=self.material['youngModulus'],
                      poissonRatio=self.material['poissonRatio'],
                      forceFieldName=self.force_field)
        beam = root.addChild(ElasticBeam(**params))
        self.apply_bcs(beam)
        add_solvers(beam.beam, self.solvers)
        return beam
