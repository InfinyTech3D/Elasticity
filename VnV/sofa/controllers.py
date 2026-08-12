"""Post-init SOFA controllers for the VnV suite."""

import numpy as np

import Sofa
import Sofa.Core


def region_indices(geometry, region, nodes):
    """Node indices whose coordinate satisfies the geometry's named-region predicate."""
    predicate = geometry.regions[region]
    return [i for i, p in enumerate(nodes) if predicate(p)]


def region_box(extents, normal, eps):
    """Vec6 BoxROI box enclosing the boundary face whose outward normal is `normal`."""
    lo = [-eps, -eps, -eps]
    hi = [e + eps for e in extents]
    axis = int(np.argmax(np.abs(normal)))
    if normal[axis] > 0.0:
        lo[axis] = extents[axis] - eps
    else:
        hi[axis] = eps
    return lo + hi


class RegionPointLoad(Sofa.Core.Controller):
    """Fills a ConstantForceField with the point forces of a region (1D: a facet is a vertex)."""

    def __init__(self, geometry, region, dofs, force_field, traction, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry = geometry
        self.region = region
        self.dofs = dofs
        self.force_field = force_field
        self.traction = traction

    def onSimulationInitDoneEvent(self, event):
        rest = self.dofs.rest_position.array()
        indices = region_indices(self.geometry, self.region, rest)
        self.force_field.indices.value = indices
        self.force_field.forces.value = [self.traction(rest[i]) for i in indices]


class NodalFieldFiller(Sofa.Core.Controller):
    """Fills a component's nodal Data field after init by sampling a function at the rest positions."""

    def __init__(self, dofs, field, sample, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dofs = dofs
        self.field = field
        self.sample = sample

    def onSimulationInitDoneEvent(self, event):
        rest = self.dofs.rest_position.array()
        values = np.array([self.sample(p) for p in rest])
        with self.field.nodalSourceDensity.writeableArray() as arr:
            arr[:] = values


class RegionClamp(Sofa.Core.Controller):
    """Fix per-region direction masks after init; move the fixed components to rest + u first (rest kept)."""

    def __init__(self, geometry, dofs, groups, displacement=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry = geometry
        self.dofs = dofs
        self.groups = groups          # list of (constraint, regions, mask)
        self.displacement = displacement

    def onSimulationInitDoneEvent(self, event):
        rest = self.dofs.rest_position.array()
        with self.dofs.position.writeableArray() as pos:
            for constraint, regions, mask in self.groups:
                idx = sorted({i for r in regions for i in region_indices(self.geometry, r, rest)})
                if self.displacement is not None:
                    # Prescribed displacement: move fixed comps to rest + u; rest_position is untouched.
                    for i in idx:
                        u_i = self.displacement(rest[i])
                        for d, fixed in enumerate(mask):
                            if fixed:
                                pos[i, d] = rest[i, d] + u_i[d]
                constraint.indices.value = idx
