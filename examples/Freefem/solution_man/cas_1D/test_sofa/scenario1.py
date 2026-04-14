"""
Scenario 1: Traction load only at x=L
Bar fixed at x=0, traction force F applied at x=L.

The analytical solution is u(x) = F*x / (E*A) — a LINEAR function of x.

"""

import numpy as np
import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime
import matplotlib.pyplot as plt  


def build_scene(root, length, force, young_modulus, poisson_ratio, nx):
    
    root.addObject('RequiredPlugin', pluginName=[
        "Elasticity",
        "Sofa.Component.Constraint.Projective",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.MechanicalLoad",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Dynamic",
    ])
    root.addObject('DefaultAnimationLoop')

    positions = [[i * length / (nx - 1)] for i in range(nx)]
    edges     = [[i, i + 1]              for i in range(nx - 1)]

    Bar = root.addChild('Bar')
    Bar.addObject('NewtonRaphsonSolver',
                  name="newtonSolver",
                  maxNbIterationsNewton=10,
                  absoluteResidualStoppingThreshold=1e-10,
                  printLog=False)
    Bar.addObject('SparseLDLSolver',
                  name="linearSolver",
                  template="CompressedRowSparseMatrixd")
    Bar.addObject('StaticSolver',
                  name="staticSolver",
                  newtonSolver="@newtonSolver",
                  linearSolver="@linearSolver")

    dofs_ref = Bar.addObject('MechanicalObject',
                             name="dofs",
                             template="Vec1d",
                             position=positions)

    Bar.addObject('EdgeSetTopologyContainer',
                  name="topology",
                  edges=edges)
    Bar.addObject('LinearSmallStrainFEMForceField',
                  name="FEM",
                  template="Vec1d",
                  youngModulus=young_modulus,
                  poissonRatio=poisson_ratio,
                  topology="@topology")

    Bar.addObject('FixedProjectiveConstraint', indices="0")
    Bar.addObject('ConstantForceField',
                  indices=f"{nx - 1}",
                  forces=f"{force}")

    return dofs_ref


def run_simulation(length, force, young_modulus, poisson_ratio, nx):
    """Run one SOFA simulation, return (x_initial, u_computed)."""
    root     = Sofa.Core.Node("root")
    dofs_ref = build_scene(root, length, force, young_modulus, poisson_ratio, nx)
    Sofa.Simulation.init(root)
    x_initial  = dofs_ref.position.array().copy().flatten()
    Sofa.Simulation.animate(root, root.dt.value)
    u_computed = dofs_ref.position.array().flatten() - x_initial
    Sofa.Simulation.unload(root)
    return x_initial, u_computed


# ====== SOFA vs u(x) = F*x / (E*A) ===========================


def sol_exact(length, force, E, nu, nx=2):

    x, u_sofa = run_simulation(length, force, E, nu, nx)

    # Analytical solution: E*A
    u_exact = force * x / (E * 1.0)

    with open('resultats_simulation.txt', 'w') as f:
        f.write("Position (x) | Déplacement SOFA | Déplacement Exact | Erreur Absolue\n")
        for xi, us, un in zip(x, u_sofa, u_exact):
            erreur = abs(us - un)
            f.write(f"{xi:10.4f}  {us:14.6e}  {un:14.6e}  {erreur:14.6e}\n")
            print(f"{xi:10.4f}  {us:14.6e}  {un:14.6e}  {erreur:14.6e}")

        u_tip_sofa  = u_sofa[-1]
        u_tip_naive = u_exact[-1]
        

        f.write(f"  u(L) SOFA    = {u_tip_sofa:.6e} m\n")
        f.write(f"  u(L) Exact   = {u_tip_naive:.6e} m\n")

    print(f"\n  u(L) SOFA    = {u_tip_sofa:.6e} m")
    print(f"  u(L) Exact   = {u_tip_naive:.6e} m")
    
    # Tracer le déplacement 
    plt.figure(figsize=(10, 6))
    plt.plot(x, u_sofa, 'bo-', label='SOFA Simulation', markersize=8, linewidth=2)
    plt.plot(x, u_exact, 'r--', label='Solution Exacte', linewidth=2)
    plt.xlabel('Position x ', fontsize=12)
    plt.ylabel('Déplacement u(x) ', fontsize=12)
    plt.title(f'Comparaison des déplacements - Barre en traction\nF={force} N, E={E} Pa, L={length} m', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('deplacement_barre.png', dpi=300, bbox_inches='tight')

    

    return x, u_sofa, u_exact

 

def main():
    length = 10.0
    force  = 100.0
    E      = 1e6
    nu     = 0.0
    nx     = 2     
    
    x, u_sofa, u_naive = sol_exact(length, force, E, nu, nx)
    
    


def createScene(rootNode):
    build_scene(rootNode,
                length=10.0, force=100.0,
                young_modulus=1e6, poisson_ratio=0.3,
                nx=2)
    return rootNode


if __name__ == "__main__":
    main()