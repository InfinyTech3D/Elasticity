"""Solver components: ODE integration scheme, Newton, linear solver (+ preconditioner)."""


def _params(config, *exclude):
    """Component parameters: every key except 'type' and any extra excluded ones."""
    skip = ('type',) + exclude
    return {k: v for k, v in config.items() if k not in skip}


def add_newton(node, config):
    return node.addObject(config['type'], name='newton', **_params(config))


def add_integration_scheme(node, config, has_newton):
    params = _params(config)
    if has_newton:
        params['newtonSolver'] = '@newton'
    return node.addObject(config['type'], name='ode', **params)


def add_linear_solver(node, config):
    precond = config.get('preconditioner')
    if precond is None:
        return node.addObject(config['type'], name='linearSolver', **_params(config))

    # Linear Solver
    system = config['system']
    node.addObject(system['type'], name='solverSystem',
                   preconditionerSystem='@precondSystem', **_params(system))
    node.addObject(config['type'], name='linearSolver', **_params(config, 'system', 'preconditioner'))

    # Preconditioner
    precond_system = precond['system']
    node.addObject(precond_system['type'], name='precondSystem', **_params(precond_system))
    return node.addObject(precond['type'], name='precond', **_params(precond, 'system'))


def add_solvers(node, config):
    """Add newton (if present) + linear solver (+ preconditioner) + ODE scheme onto node."""
    has_newton = 'newton' in config
    if has_newton:
        add_newton(node, config['newton'])
    add_linear_solver(node, config['linearSolver'])
    add_integration_scheme(node, config['integration'], has_newton)
