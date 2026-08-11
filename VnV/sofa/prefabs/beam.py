"""SOFA prefab: a dimension/element-agnostic elastic beam."""

import Sofa
import Sofa.Core

from ..conventions import VEC_BY_DIM, CONTAINER, MAPPING, ELEMENT_CPP


def validate_parameters(config):
    """Required ElasticBeam parameters."""
    required = ['extents', 'resolution', 'dim', 'element', 'youngModulus', 'poissonRatio', 'forceFieldName']
    missing = [p for p in required if p not in config]
    if missing:
        raise ValueError(f"ElasticBeam: missing required parameters {missing}")


class ElasticBeam(Sofa.Prefab):
    """SOFA elastic beam: RegularGridTopology + dofs + FEM force field."""

    prefabParameters = [
        {'name': 'extents',        'type': 'Vec3d',  'help': 'box max corner [Lx, Ly, Lz]'},
        {'name': 'resolution',     'type': 'Vec3d',  'help': 'nodes per axis [nx, ny, nz]'},
        {'name': 'dim',            'type': 'int',    'help': 'spatial dimension'},
        {'name': 'element',        'type': 'string', 'help': 'element kind (edge/tri/quad/tet/hexa)'},
        {'name': 'youngModulus',   'type': 'double', 'help': "Young's modulus"},
        {'name': 'poissonRatio',   'type': 'double', 'help': "Poisson's ratio"},
        {'name': 'forceFieldName', 'type': 'string', 'help': 'FEM force field component name'},
    ]

    def __init__(self, *args, **kwargs):
        validate_parameters(kwargs)
        Sofa.Prefab.__init__(self, *args, **kwargs)

    def init(self):
        dim = self.dim.value
        VecType = VEC_BY_DIM[dim]
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
            # FEM
            paramsFEM = dict(youngModulus=self.youngModulus.value,
                             poissonRatio=self.poissonRatio.value)
            beam.addObject(self.forceFieldName.value, name='FEM',
                           template=f"{VecType},{ELEMENT_CPP[element]}",
                           topology='@topology', **paramsFEM)
