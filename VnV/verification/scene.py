"""Verification suite: an MMS scene whose BCs come from a manufactured solution."""

import numpy as np

from ..sofa.scene import Scene
from ..sofa.conventions import CONTAINER, VEC_BY_DIM
from ..sofa.controllers import NodalForceAssembler, RegionClamp, region_facets
from .fem import ELEMENT_RULES, FACET_RULES, source_integration, source_integration_boundary


class MMSScene(Scene):
    """Scene whose boundary conditions are those of a manufactured solution."""

    def __init__(self, geometry, material, force_field, element, resolution, solvers, mms):
        super().__init__(geometry, material, force_field, element, resolution, solvers)
        self.mms = mms

    def apply_bcs(self, beam):
        node = beam.beam
        g, mms, material, element = self.geometry, self.mms, self.material, self.element
        connectivity = CONTAINER[element][1]
        element_rule = ELEMENT_RULES[element]()
        facet_rule = FACET_RULES[element]()

        # Prescribed displacement u_ex per region+direction mask: one partial clamp per mask, filled post-init.
        by_mask = {}
        for region, mask in mms.prescribe_displacement_on.items():
            by_mask.setdefault(tuple(mask), []).append(region)
        groups = []
        for mask, regions in by_mask.items():
            constraint = node.addObject('PartialFixedProjectiveConstraint',
                                        name='clamp' + ''.join(str(m) for m in mask),
                                        template=VEC_BY_DIM[g.dim], fixedDirections=list(mask))
            groups.append((constraint, regions, mask))
        node.addObject(RegionClamp(geometry=g, dofs=node.dofs, groups=groups,
                                   displacement=mms.u, name='clampCtrl'))

        # Body force from the source + boundary traction sigma.n: assembled post-init.
        n = int(np.prod(self.resolution))
        load = node.addObject('ConstantForceField', name='load', template=VEC_BY_DIM[g.dim],
                              indices=list(range(n)), forces=[[0.0] * g.dim] * n)

        def compute(nodes, topology):
            conn = getattr(topology, connectivity).array()

            def body_force(*coords):
                return mms.source(np.asarray(coords), material)

            F = source_integration(body_force, nodes, conn, element_rule)
            for r in mms.traction_on:
                normal = np.asarray(g.normals[r])

                def traction(*coords):
                    return mms.stress(np.asarray(coords), material) @ normal

                facets = region_facets(g, r, nodes, topology, element)
                F += source_integration_boundary(traction, nodes, facets, facet_rule)
            return F

        node.addObject(NodalForceAssembler(dofs=node.dofs, topology=node.topology,
                                           force_field=load, compute_forces=compute,
                                           name='loadCtrl'))
