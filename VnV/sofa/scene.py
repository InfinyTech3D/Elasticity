"""Generic SOFA scene assembly."""

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
    "Sofa.Component.Visual",
]


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
        dim = self.geometry.dim
        resolution = [self.resolution[i] if i < len(self.resolution) else 1 for i in range(3)]
        params = dict(name='beam',
                      extents=self.geometry.extents,
                      resolution=resolution,
                      dim=dim,
                      element=self.element,
                      youngModulus=self.material['youngModulus'],
                      forceFieldName=self.force_field)
        if 'poissonRatio' in self.material:
            params['poissonRatio'] = self.material['poissonRatio']
        beam = root.addChild(ElasticBeam(**params))
        self.apply_bcs(beam)
        add_solvers(beam.beam, self.solvers)
        return beam
