"""
sofa_bar_mms.py u(x) cubic; 
  
"""
import Sofa
import Sofa.Core
import Sofa.Simulation
import numpy as np


def createScene(rootNode, length=1.0, young_modulus=1000.0, poisson_ratio=0.3, nx=10):

    rootNode.addObject("RequiredPlugin", pluginName=[
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.Mass",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.SolidMechanics.Spring",
        "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Grid",
        "Sofa.Component.Constraint.Projective",
        "Sofa.Component.MechanicalLoad",
        "Sofa.Component.AnimationLoop"
    ])

    rootNode.gravity = [0, 0, 0]
    rootNode.dt = 0.01
    rootNode.addObject("DefaultAnimationLoop")
    rootNode.addObject("EulerImplicitSolver", name="odeSolver")
    rootNode.addObject("SparseLDLSolver", name="linearSolver",
                       template="CompressedRowSparseMatrixd")

    rootNode.addObject("RegularGridTopology", name="grid",
                       min=[0, 0, 0], max=[length, 0, 0],
                       nx=nx, ny=1, nz=1)

    rootNode.addObject("MechanicalObject", name="dofs", template="Vec3d")
    rootNode.addObject("UniformMass", name="mass", totalMass=1.0)
    rootNode.addObject("FixedProjectiveConstraint", name="dirichletBC", indices="0")

    # Rigidité 
    h = length / (nx - 1)
    A = 1.0
    k_spring = young_modulus * A / h
    springs = [[i, i+1, k_spring, 0.0, h] for i in range(nx - 1)]
    springs_str = " ".join([f"{s[0]} {s[1]} {s[2]} {s[3]} {s[4]}" for s in springs])
    rootNode.addObject("StiffSpringForceField", name="springs", spring=springs_str)

    
    x_coords = np.linspace(0, length, nx)
    nodal_forces = np.zeros(nx)

    for i in range(1, nx):         
        x = x_coords[i]
        f_val = young_modulus * (6*x - 2*length) / (length**2)
        if i == nx - 1:
            nodal_forces[i] = f_val * h / 2   
        else:
            nodal_forces[i] = f_val * h        

    nodal_forces[nx - 1] += -young_modulus / length


    indices_body = list(range(1, nx))
    force_str = " ".join([f"{nodal_forces[i]} 0 0" for i in indices_body])
    rootNode.addObject("ConstantForceField", name="allForces",
                       indices=" ".join(map(str, indices_body)),
                       forces=force_str)

    return rootNode


def sofaRun(length=1.0, young_modulus=1000.0, poisson_ratio=0.3, nx=10):
    root = Sofa.Core.Node("root")
    createScene(root, length, young_modulus, poisson_ratio, nx)

    Sofa.Simulation.init(root)

    for _ in range(100):
        Sofa.Simulation.animate(root, root.dt.value)

    dofs = root.dofs
    positions      = dofs.position.array()
    rest_positions = dofs.rest_position.array()

    x_coords = rest_positions[:, 0]
    u_x      = positions[:, 0] - rest_positions[:, 0]

    Sofa.Simulation.unload(root)

    return x_coords, u_x