"""
sofa_bar_quad.py - Barre 1D MMS solution QUADRATIQUE
u_exact(x) = x*(L-x)/L^2
f(x)       = 2E/L^2   const 
sigma(L)   = -E/L     Neumann
u(0)       = 0        Dirichlet
"""
import Sofa
import Sofa.Core
import Sofa.Simulation
import numpy as np


def createScene(rootNode, length=1.0, young_modulus=1000.0,
                poisson_ratio=0.3, nx=10):

    rootNode.addObject("RequiredPlugin", pluginName=[
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.Mass",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.SolidMechanics.Spring",
        "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Grid",
        "Sofa.Component.Constraint.Projective",
        "Sofa.Component.MechanicalLoad",
        "Sofa.Component.AnimationLoop",
    ])

    rootNode.gravity = [0, 0, 0]

    rootNode.dt = 1e6
    rootNode.addObject("DefaultAnimationLoop")

    rootNode.addObject("EulerImplicitSolver",
                       name="odeSolver",
                       rayleighStiffness=0.0,
                       rayleighMass=0.0)
    rootNode.addObject("SparseLDLSolver", name="linearSolver",
                       template="CompressedRowSparseMatrixd")

    rootNode.addObject("RegularGridTopology", name="grid",
                       min=[0, 0, 0], max=[length, 0, 0],
                       nx=nx, ny=1, nz=1)
    rootNode.addObject("MechanicalObject", name="dofs", template="Vec3d")


    rootNode.addObject("UniformMass", name="mass", totalMass=1e-10)

    rootNode.addObject("FixedProjectiveConstraint", name="dirichletBC",
                       indices="0")

    # Rigidite : k = EA/h, A=1
    h           = length / (nx - 1)
    k_spring    = young_modulus / h
    springs     = [[i, i+1, k_spring, 0.0, h] for i in range(nx - 1)]
    springs_str = " ".join(f"{s[0]} {s[1]} {s[2]} {s[3]} {s[4]}"
                           for s in springs)
    rootNode.addObject("StiffSpringForceField", name="springs",
                       spring=springs_str)

    L       = length
    f_const = 2.0 * young_modulus / (L * L)

    nodal_forces = np.zeros(nx)
    for i in range(1, nx - 1):
        nodal_forces[i] = f_const * h
    nodal_forces[nx - 1] = f_const * h / 2.0

    indices_body = list(range(1, nx))
    forces_body  = " ".join(f"{nodal_forces[i]} 0 0" for i in indices_body)
    rootNode.addObject("ConstantForceField", name="bodyForce",
                       indices=" ".join(map(str, indices_body)),
                       forces=forces_body)

    #  NEUMANN BC  : sigma(L) = -E/L
    
    rootNode.addObject("ConstantForceField", name="neumannBC",
                       indices=str(nx - 1),
                       forces=f"{-young_modulus / L} 0 0")

    return rootNode


def sofaRun(length=1.0, young_modulus=1000.0,
            poisson_ratio=0.3, nx=10):

    root = Sofa.Core.Node("root")
    createScene(root, length, young_modulus, poisson_ratio, nx)
    Sofa.Simulation.init(root)


    Sofa.Simulation.animate(root, root.dt.value)

    dofs           = root.dofs
    positions      = dofs.position.array()
    rest_positions = dofs.rest_position.array()

    x_coords = rest_positions[:, 0]
    u_x      = positions[:, 0] - rest_positions[:, 0]

    Sofa.Simulation.unload(root)
    return x_coords, u_x