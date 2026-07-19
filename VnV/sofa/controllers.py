"""Post-init SOFA controllers for the VnV suite."""

import Sofa
import Sofa.Core

from .conventions import FACET_FIELD


def region_indices(geometry, region, nodes):
    """Node indices whose coordinate satisfies the geometry's named-region predicate."""
    predicate = geometry.regions[region]
    return [i for i, p in enumerate(nodes) if predicate(p)]


def region_facets(geometry, region, nodes, topology, element):
    """SOFA boundary facets whose every node satisfies the region predicate (flat region loci)."""
    predicate = geometry.regions[region]
    facet_field = FACET_FIELD[element]
    if facet_field is None:                       # 1D: a facet is a single vertex
        candidates = [(i,) for i in range(len(nodes))]
    else:
        candidates = getattr(topology, facet_field).array()
    return [tuple(int(i) for i in f) for f in candidates
            if all(predicate(nodes[i]) for i in f)]


class NodalForceAssembler(Sofa.Core.Controller):
    """Fills a placeholder ConstantForceField after init."""

    def __init__(self, dofs, topology, force_field, compute_forces, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dofs = dofs
        self.topology = topology
        self.force_field = force_field
        self.compute_forces = compute_forces

    def onSimulationInitDoneEvent(self, event):
        # Use the rest position; displacement BC may move-&-clamp boundary to prescribed location
        nodes = self.dofs.rest_position.array().copy()
        F = self.compute_forces(nodes, self.topology)
        with self.force_field.forces.writeableArray() as forces:
            forces[:] = F


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
