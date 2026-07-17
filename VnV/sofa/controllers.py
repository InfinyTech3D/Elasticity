"""Post-init SOFA controllers for the VnV suite."""

import Sofa
import Sofa.Core


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
