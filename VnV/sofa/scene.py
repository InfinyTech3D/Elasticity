"""Generic SOFA scene assembly."""

from .solvers import add_solvers
from .prefabs.beam import ElasticBeam


PLUGINS = [
    "Elasticity",
    "Sofa.Component.LinearSolver.Direct",
    "Sofa.Component.LinearSolver.Iterative",
    "Sofa.Component.LinearSolver.Preconditioner",
    "Sofa.Component.LinearSystem",
    "Sofa.Component.ODESolver.Backward",
    "Sofa.Component.StateContainer",
    "Sofa.Component.Topology.Container.Grid",
    "Sofa.Component.Topology.Container.Dynamic",
    "Sofa.Component.Visual",
]


class Scene:
    """Assembles a SOFA scene: the elastic beam, then boundary conditions, then solvers."""

    def __init__(self, geometry, material, force_field, resolution, solvers):
        self.geometry = geometry
        self.material = material
        self.force_field = force_field
        self.resolution = resolution
        self.solvers = solvers

    def apply_bcs(self, beam):
        """Boundary conditions — filled by the verification or validation suite."""

    def build(self, root):
        root.addObject('RequiredPlugin', pluginName=PLUGINS)
        root.addObject('DefaultAnimationLoop')
        beam = root.addChild(ElasticBeam(name='beam', geometry=self.geometry,
                                         material=self.material, force_field=self.force_field,
                                         resolution=self.resolution))
        self.apply_bcs(beam)
        add_solvers(beam.beam, self.solvers)
        return beam
