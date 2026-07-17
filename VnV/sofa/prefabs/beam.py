"""SOFA prefab: a dimension/element-agnostic elastic beam built from a geometry."""

import Sofa
import Sofa.Core

from ..conventions import VEC_BY_DIM, CONTAINER


class ElasticBeam(Sofa.Prefab):
    """SOFA realization of a geometry: topology + dofs + FEM force field."""

    def __init__(self, *args, geometry, material, force_field, resolution, **kwargs):
        self.geo = geometry
        self.material = material
        self.force_field = force_field
        self.resolution = resolution
        Sofa.Prefab.__init__(self, *args, **kwargs)

    def init(self):
        g = self.geo
        VecType = VEC_BY_DIM[g.dim]
        container, connectivity = CONTAINER[g.element]
        n, L = self.resolution, g.length

        # Beam spans [0, L] per active dimension; inactive axes collapse to one layer.
        if g.dim == 1:
            grid = dict(nx=n, ny=1, nz=1, min=[0.0, 0.0, 0.0], max=[L, 0.0, 0.0])
        elif g.dim == 2:
            grid = dict(nx=n, ny=n, nz=1, min=[0.0, 0.0, 0.0], max=[L, L, 0.0])
        else:
            grid = dict(nx=n, ny=n, nz=n, min=[0.0, 0.0, 0.0], max=[L, L, L])

        # Grid Topology Node
        with self.addChild('Grid') as grid_node:
            grid_node.addObject('RegularGridTopology', name='grid', **grid)

        # Node containing Beam components
        with self.addChild('Beam') as beam:
            self.beam = beam

            # Topology
            beam.addObject(container, name='topology', position='@../Grid/grid.position',
                           **{connectivity: f'@../Grid/grid.{connectivity}'})
            # DOFs
            beam.addObject('MechanicalObject', name='dofs', template=VecType)
            # FEM
            beam.addObject(self.force_field, name='FEM', template=VecType,
                           topology='@topology', **self.material)
