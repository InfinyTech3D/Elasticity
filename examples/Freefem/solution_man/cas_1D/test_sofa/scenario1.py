"""
Scenario 1: Traction load only at x=L
Bar fixed at x=0, traction force F applied at x=L.

The analytical solution is u(x) = F*x / (E*A) — a LINEAR function of x.


Problem with Vec1d: 
  LinearSmallStrainFEMForceField uses the 3D oedometric modulus
  E_oed = lambda + 2*mu  instead of E 
  which introduces a Poisson-ratio dependency even in 1D.

This script demonstrates:
  - Part I : naive comparison SOFA vs u_exact using E directly == error 
  - Part II : corrected comparison SOFA vs u_exact using E_oed  == exact result
"""

import numpy as np
import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime


def compute_lame_coefficients(E, nu):
    """
    3D Lamé coefficients and oedometric modulus.
    SOFA's LinearSmallStrainFEMForceField (Vec1d) uses E_oed = lambda + 2*mu
    """
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu    = E / (2 * (1 + nu))
    E_oed = lmbda + 2 * mu
    return lmbda, mu, E_oed


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



# ================Part I  : SOFA vs u(x) = F*x / (E*A) ===========================


def part_A_naive(length, force, E, nu, nx=2):

    x, u_sofa = run_simulation(length, force, E, nu, nx)

    # Analytical solution: E*A
    u_naive = force * x / (E * 1.0)

    for xi, us, un in zip(x, u_sofa, u_naive):
        print(f"{xi:10.4f}  {us:14.6e}  {un:14.6e}  {abs(us-un):14.6e}")

    u_tip_sofa  = u_sofa[-1]
    u_tip_naive = u_naive[-1]
    err_pct     = abs(u_tip_sofa - u_tip_naive) / u_tip_naive * 100

    print(f"\n  u(L) SOFA    = {u_tip_sofa:.6e} m")
    print(f"  u(L) naive   = {u_tip_naive:.6e} m")
    print(f"  Relative error = {err_pct:.2f}%")



# ============== Part II: using  u(x) = F*x / (E_oed*A) ============================

def part_B_corrected(length, force, E, nu):

    lmbda, mu, E_oed = compute_lame_coefficients(E, nu)

    print(f"  lambda     = {lmbda:.6e} Pa")
    print(f"  mu         = {mu:.6e} Pa")
    print(f"  E_oed      = {E_oed:.6e} Pa  = lambda + 2*mu")

    print(f"\n{'nx':>6}  {'u_SOFA(L)':>14}  {'u_exact(L)':>14}  {'Error':>14}  {'Error%':>8}")

    all_pass = True
    for nx in [2, 4, 8, 16, 32]:
        x, u_sofa  = run_simulation(length, force, E, nu, nx)
        u_exact     = force * x / (E_oed * 1.0)
        u_tip_sofa  = u_sofa[-1]
        u_tip_exact = u_exact[-1]
        err         = abs(u_tip_sofa - u_tip_exact)
        err_pct     = err / u_tip_exact * 100
        status      = "✓" if err < 1e-10 else "✗"
        if err >= 1e-10:
            all_pass = False
        print(f"{nx:6d}  {u_tip_sofa:14.6e}  {u_tip_exact:14.6e}  {err:14.6e}  {err_pct:7.2f}%  {status}")

    print()
    if all_pass:
        print(" With E_oed, SOFA matches the exact linear")
        print("    The 1-element result is EXACT as expected for P1 ")
    else:
        print("   Some cases did not converge ")

    
    print(f"\n  {'nu':>6}  {'E_oed':>14}  {'u_SOFA(L)':>14}  {'u_exact(L)':>14}  {'Error%':>8}")
    for nu_test in [0.0, 0.1, 0.2, 0.3, 0.4, 0.49]:
        _, _, E_oed_t = compute_lame_coefficients(E, nu_test)
        x, u_s = run_simulation(length, force, E, nu_test, nx=2)
        u_e    = force * x[-1] / (E_oed_t * 1.0)
        ep     = abs(u_s[-1] - u_e) / u_e * 100 if u_e != 0 else 0.0
        print(f"  {nu_test:6.2f}  {E_oed_t:14.4e}  {u_s[-1]:14.6e}  {u_e:14.6e}  {ep:7.2f}%")

    print(f"\n  When nu=0: E_oed = E and both solutions coincide.")
    print(f"  When nu>0: E_oed > E, SOFA is stiffer than naive expectation.")


def main():
    length = 10.0
    force  = 100.0
    E      = 1e6
    nu     = 0.3
    nx     = 2     

    part_A_naive(length, force, E, nu, nx)
    part_B_corrected(length, force, E, nu)


def createScene(rootNode):
    """Entry point for runSofa GUI."""
    build_scene(rootNode,
                length=10.0, force=100.0,
                young_modulus=1e6, poisson_ratio=0.3,
                nx=8)
    return rootNode


if __name__ == "__main__":
    main()