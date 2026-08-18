"""Verification suite: an MMS scene whose BCs come from a manufactured solution."""

import numpy as np

from ..sofa.scene import Scene
from ..sofa.conventions import (BOUNDARY_KIND, CONTAINER, ELEMENT_CPP, FACET_FIELD,
                                VEC_BY_SPATIAL_DIM)
from ..sofa.controllers import SourceTermFiller, RegionClamp, RegionPointLoad, region_box


class MMSScene(Scene):
    """Scene whose boundary conditions are those of a manufactured solution."""

    def __init__(self, deck, resolution):
        """The deck stops here. `Scene` takes its arguments one by one because it is the generic
        layer the validation suite will reuse, and a validation deck is a different shape; this
        subclass is verification's own, so it may know what a Deck is and unpack one.
        """
        super().__init__(deck.geometry, deck.material, deck.force_field, deck.element,
                         resolution, deck.solvers)
        self.mms = deck.solution
        self.source_quadrature_degree = deck.source_quadrature_degree

    def apply_bcs(self, beam):
        node = beam.beam
        g, mms, element = self.geometry, self.mms, self.element
        VEC = VEC_BY_SPATIAL_DIM[g.spatial_dimensions]

        # Prescribed displacement u_ex per region+direction mask: one partial clamp per mask, filled post-init.
        by_mask = {}
        for region, mask in mms.prescribe_displacement_on.items():
            # A solution states its mask in its own dimension; off-manifold components are not the
            # field's business and are handled once by the out-of-plane clamp below.
            padded = tuple(mask) + (0,) * (g.spatial_dimensions - len(mask))
            by_mask.setdefault(padded, []).append(region)
        groups = []
        for mask, regions in by_mask.items():
            constraint = node.addObject('PartialFixedProjectiveConstraint',
                                        name='clamp' + ''.join(str(m) for m in mask),
                                        template=VEC, fixedDirections=list(mask))
            groups.append((constraint, regions, mask))
        node.addObject(RegionClamp(geometry=g, dofs=node.dofs, groups=groups,
                                   displacement=mms.u, name='clampCtrl'))

        # An embedded mesh has an out-of-plane null mode: that block is decoupled from the in-plane one
        # and carries no source, so it has a rigid translation for a null mode. Fixing it everywhere is
        # what keeps the stiffness matrix regular -- and u = 0 there, so the constraint costs no physics.
        if g.dim < g.spatial_dimensions:
            node.addObject('PartialFixedProjectiveConstraint', name='outOfPlane', template=VEC,
                           fixAll=True,
                           fixedDirections=[0] * g.dim + [1] * (g.spatial_dimensions - g.dim))

        # Body force from the source: integrated by SOFA's FEMSourceTerm component (SOFA quadrature).
        bf = node.addObject('FEMSourceTerm', name='bodyForce',
                            template=f"{VEC},{ELEMENT_CPP[element]}",
                            quadratureDegree=self.source_quadrature_degree)
        node.addObject(SourceTermFiller(dofs=node.dofs, field=bf, sample=mms.source,
                                        name='bodyForceCtrl'))

        boundary = BOUNDARY_KIND.get(element)
        spacings = [e / (res - 1) for e, res in zip(g.extents, self.resolution) if res > 1]
        eps = 0.25 * min(spacings)
        for region in mms.traction_on:
            normal = np.asarray(g.normals[region])

            def traction(point, n=normal):
                return mms.stress(np.asarray(point)) @ n

            if boundary is None:
                load = node.addObject('ConstantForceField', name=f'load_{region}', template=VEC,
                                      indices=[0], forces=[[0.0] * g.spatial_dimensions])
                node.addObject(RegionPointLoad(geometry=g, region=region, dofs=node.dofs,
                                               force_field=load, traction=traction,
                                               name=f'load_{region}Ctrl'))
                continue

            facets = FACET_FIELD[element]
            compute = {f'compute{kind.capitalize()}': kind == facets
                       for kind in ('edges', 'triangles', 'quads', 'tetrahedra', 'hexahedra')}
            child = node.addChild(f'neumann_{region}')
            child.addObject('BoxROI', name='roi', template=VEC, strict=True,
                            box=[region_box(g.extents, normal, eps)],
                            position='@../dofs.rest_position',
                            **{facets: f'@../topology.{facets}'}, **compute)
            child.addObject(CONTAINER[boundary][0], name='surface',
                            position='@../dofs.rest_position',
                            **{CONTAINER[boundary][1]: f'@roi.{facets}InROI'})
            child.addObject('MechanicalObject', name='surfaceDofs', template=VEC)
            child.addObject('IdentityMapping', template=f'{VEC},{VEC}', applyRestPosition=True,
                            input='@../dofs', output='@surfaceDofs')
            load = child.addObject('FEMSourceTerm', name='traction', topology='@surface',
                                   template=f'{VEC},{ELEMENT_CPP[boundary]}',
                                   quadratureDegree=self.source_quadrature_degree)
            child.addObject(SourceTermFiller(dofs=child.surfaceDofs, field=load,
                                             sample=traction, name='tractionCtrl'))
