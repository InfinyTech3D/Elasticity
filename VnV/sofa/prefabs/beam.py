"""SOFA prefab: a dimension/element-agnostic elastic beam."""

from .base import ScenePrefab
from ..conventions import VEC_BY_SPATIAL_DIM, CONTAINER, MAPPING, ELEMENT_CPP


def validate_parameters(config):
    """Required ElasticBeam parameters."""
    required = ['extents', 'resolution', 'spatialDimensions', 'element']
    missing = [p for p in required if p not in config]
    if missing:
        raise ValueError(f"ElasticBeam: missing required parameters {missing}")


class ElasticBeam(ScenePrefab):
    """SOFA elastic beam: RegularGridTopology + dofs, then the components the deck states."""

    prefabParameters = [
        {'name': 'extents',        'type': 'Vec3d',  'help': 'box max corner [Lx, Ly, Lz]'},
        {'name': 'resolution',     'type': 'Vec3d',  'help': 'nodes per axis [nx, ny, nz]'},
        {'name': 'spatialDimensions', 'type': 'int', 'help': 'dimension of the embedding space'},
        {'name': 'element',        'type': 'string', 'help': 'element kind (edge/tri/quad/tet/hexa)'},
    ]

    def __init__(self, *args, **kwargs):
        validate_parameters(kwargs)
        super().__init__(*args, **kwargs)

    def init(self):
        VecType = VEC_BY_SPATIAL_DIM[self.spatialDimensions.value]
        element = self.element.value
        container, connectivity = CONTAINER[element]
        mapping = MAPPING[element]
        res = self.resolution.value

        # Grid Topology Node
        with self.addChild('Grid') as grid_node:
            grid_node.addObject('RegularGridTopology', name='grid',
                                nx=int(res[0]), ny=int(res[1]), nz=int(res[2]),
                                min=[0.0, 0.0, 0.0], max=list(self.extents.value))

        # Node containing Beam components
        with self.addChild('Beam') as beam:
            self.beam = beam

            # Topology
            if mapping is None:
                beam.addObject(container, name='topology', position='@../Grid/grid.position',
                               **{connectivity: f'@../Grid/grid.{connectivity}'})
            else:
                beam.addObject(container, name='topology', position='@../Grid/grid.position')
                beam.addObject(mapping, input='@../Grid/grid', output='@topology')
                beam.addObject(container.replace('Container', 'Modifier'))
            # DOFs
            beam.addObject('MechanicalObject', name='dofs', template=VecType)
            # The component under test and the solve, both stated by the deck (see ScenePrefab).
            self.add_force_field(beam, self.spec['forceField'], self.spec['material'],
                                 template=f"{VecType},{ELEMENT_CPP[element]}")
            self.add_solvers(beam, self.spec['solvers'])
