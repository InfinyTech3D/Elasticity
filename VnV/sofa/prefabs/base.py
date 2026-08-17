"""Prefabs a study configures: they take the deck blocks they need and build themselves."""

import Sofa
import Sofa.Core


def params(config, *exclude):
    """Component parameters: every key except 'type' and any extra excluded ones."""
    skip = ('type',) + exclude
    return {k: v for k, v in config.items() if k not in skip}


class ScenePrefab(Sofa.Prefab):
    """A prefab that receives the deck blocks describing what to build, and knows how to build them.

    `spec` carries whatever a subclass needs that is not a Data -- the deck's material, force field
    and solver blocks, which are dicts of arbitrary keys and so cannot be prefabParameters. It has to
    arrive before Sofa.Prefab.__init__ because that call ends by invoking init(), which is what reads
    it, and it has to arrive through object.__setattr__ because Sofa's own __setattr__ routes through
    a C++ object that does not exist yet. This is the only place in the suite that does either, which
    is the point of putting it on a base class: a subclass writes `self.spec` and nothing else.

    The adders take the node to build on rather than reading it off the prefab, so the only contract
    between this class and a subclass is the `spec` this constructor sets.
    """

    def __init__(self, *args, spec=None, **kwargs):
        # NOT `self.spec = spec`: that goes through Sofa's __setattr__, reaches an unconstructed C++
        # object, and segfaults with no traceback. And not after the call below either -- init() has
        # already run by then.
        object.__setattr__(self, 'spec', spec or {})
        Sofa.Prefab.__init__(self, *args, **kwargs)

    def add_force_field(self, node, config, material, template):
        """The component under test, with everything the deck states about it.

        The same contract the solvers get: keys other than 'type' reach the component as Data, so a
        deck can set a rotation method or a compute strategy without a Python change. The
        constitutive parameters come from `material`, which the manufactured source reads too, so the
        law being verified and the law being solved cannot drift apart.
        """
        return node.addObject(config['type'], name='FEM', template=template, topology='@topology',
                              **params(material), **params(config))

    def add_solvers(self, node, config):
        """Newton (if present) + linear solver (+ preconditioner) + integration scheme."""
        newton = config.get('newton')
        if newton is not None:
            node.addObject(newton['type'], name='newton', **params(newton))
        self._add_linear_solver(node, config['linearSolver'])
        scheme = params(config['integration'])
        if newton is not None:
            scheme['newtonSolver'] = '@newton'
        node.addObject(config['integration']['type'], name='ode', **scheme)

    def _add_linear_solver(self, node, config):
        precond = config.get('preconditioner')
        if precond is None:
            return node.addObject(config['type'], name='linearSolver', **params(config))

        # Linear Solver
        system = config['system']
        node.addObject(system['type'], name='solverSystem',
                       preconditionerSystem='@precondSystem', **params(system))
        node.addObject(config['type'], name='linearSolver',
                       **params(config, 'system', 'preconditioner'))

        # Preconditioner
        precond_system = precond['system']
        node.addObject(precond_system['type'], name='precondSystem', **params(precond_system))
        return node.addObject(precond['type'], name='precond', **params(precond, 'system'))
