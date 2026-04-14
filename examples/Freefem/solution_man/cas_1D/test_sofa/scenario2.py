"""
Scenario 2: Constant distributed load across the bar
Bar fixed at x=0, constant distributed load q applied along the bar.

The analytical solution is u(x) = (q/(E*A)) * (L*x - x²/2) — a QUADRATIC function of x.

"""

import numpy as np
import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime
import matplotlib.pyplot as plt
import os
plt.switch_backend('Agg')


def exact_solution_quadratic(x, length, q, E, area=1.0):
    """
    u(x) = q/(E*A) * (L*x - x²/2)
    """
    return (q / (E * area)) * (length * x - x**2 / 2.0)


def build_scene_distributed(root, length, q, young_modulus, nx):
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

    # Maillage uniforme
    h = length / (nx - 1)
    positions = [[i * h] for i in range(nx)]
    edges = [[i, i + 1] for i in range(nx - 1)]

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
                  poissonRatio=0.3,  
                  topology="@topology")

    Bar.addObject('FixedProjectiveConstraint', indices="0")

    for i in range(1, nx):
        if i == 0:
            continue
        elif i == nx - 1:
            nodal_force = q * h / 2.0  
        else:
            nodal_force = q * h      
        
        Bar.addObject('ConstantForceField',
                      indices=str(i),
                      forces=str(nodal_force))

    return dofs_ref


def run_simulation(length, q, young_modulus, nx):
    root = Sofa.Core.Node("root")
    dofs_ref = build_scene_distributed(root, length, q, young_modulus, nx)
    Sofa.Simulation.init(root)
    x_initial = dofs_ref.position.array().copy().flatten()
    Sofa.Simulation.animate(root, root.dt.value)
    u_computed = dofs_ref.position.array().flatten() - x_initial
    Sofa.Simulation.unload(root)
    return x_initial, u_computed


def compute_l2_error_nodal(x_nodes, u_computed, length, q, E):
    idx = np.argsort(x_nodes)
    x_s = x_nodes[idx]
    u_s = u_computed[idx]
    u_ex = exact_solution_quadratic(x_s, length, q, E)
    err = u_s - u_ex
    return np.sqrt(np.trapezoid(err**2, x_s))


def compute_l2_error_midpoints(x_nodes, u_computed, length, q, E):
    idx = np.argsort(x_nodes)
    x_s = x_nodes[idx]
    u_s = u_computed[idx]
    l2_error = 0.0
    
    for i in range(len(x_s) - 1):
        x_mid = (x_s[i] + x_s[i+1]) / 2.0
        u_interp = (u_s[i] + u_s[i+1]) / 2.0  
        u_ex = exact_solution_quadratic(np.array([x_mid]), length, q, E)[0]
        dx = x_s[i+1] - x_s[i]
        l2_error += (u_interp - u_ex)**2 * dx
    
    return np.sqrt(l2_error)


def simulation_distributed(length, q, E, nx):
   
    
    x, u_sofa = run_simulation(length, q, E, nx)
    u_exact = exact_solution_quadratic(x, length, q, E)
    
    with open('resultats_scenario2.txt', 'w') as f:
        f.write(f"Longueur L = {length} \n")
        f.write(f"Charge répartie q = {q} \n")
        f.write(f"Module d'Young E = {E} \n")
        f.write(f"Nombre de nœuds nx = {nx}\n")
        f.write(f"Nombre d'éléments = {nx-1}\n")
        f.write(f"{'Position x (m)':>15} | {'u_SOFA (m)':>15} | {'u_exact (m)':>15} | {'Erreur (m)':>15}\n")

        
        print(f"{'x (m)':>10} | {'u_SOFA (m)':>15} | {'u_exact (m)':>15} | {'Erreur (m)':>15}")
        
        for xi, us, ue in zip(x, u_sofa, u_exact):
            erreur = abs(us - ue)
            f.write(f"{xi:15.4f} | {us:15.6e} | {ue:15.6e} | {erreur:15.6e}\n")
            print(f"{xi:10.4f} | {us:15.6e} | {ue:15.6e} | {erreur:15.6e}")
    
    # Erreurs L2
    err_nodal = compute_l2_error_nodal(x, u_sofa, length, q, E)
    err_mid = compute_l2_error_midpoints(x, u_sofa, length, q, E)
    
    with open('resultats_scenario2.txt', 'a') as f:

        f.write(f"Erreur L2 (aux nœuds)     = {err_nodal:.6e} (superconvergence)\n")
        f.write(f"Erreur L2 (aux milieux)   = {err_mid:.6e} (vraie erreur d'interpolation)\n")
    
    print(f"Erreur L2 (aux nœuds)     = {err_nodal:.6e} (superconvergence)")
    print(f"Erreur L2 (aux milieux)   = {err_mid:.6e} (vraie erreur d'interpolation)")
    
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, u_sofa, 'bo-', label='Simulation SOFA', markersize=6, linewidth=2)
    
    
    x_fine = np.linspace(0, length, 200)
    u_fine = exact_solution_quadratic(x_fine, length, q, E)
    plt.plot(x_fine, u_fine, 'r--', label='Solution exacte (quadratique)', linewidth=2)
    
    plt.xlabel('Position x ', fontsize=12)
    plt.ylabel('Déplacement u(x) ', fontsize=12)
    plt.title(f'Charge répartie constante - Barre en traction\nq={q} N/m, E={E} Pa, L={length} m, nx={nx}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('deplacement_scenario2.png', dpi=300, bbox_inches='tight')
    
    return x, u_sofa, u_exact


def convergence_study_distributed(length, q, E, nx_values):
    
    print(f"{'nx':>6} | {'h (m)':>10} | {'Erreur L2 (nœuds)':>20} | {'Erreur L2 (milieux)':>22} | {'Taux O(h)':>10}")

    
    l2_nodal = []
    l2_midpoints = []
    element_sizes = []
    
    with open('convergence_scenario2.txt', 'w') as f:
        f.write(f"{'nx':>6} | {'h (m)':>10} | {'Erreur L2 (nœuds)':>20} | {'Erreur L2 (milieux)':>22} | {'Taux O(h)':>10}\n")
    
        for i, nx in enumerate(nx_values):
            h = length / (nx - 1)
            x, u_sofa = run_simulation(length, q, E, nx)
            
            err_nodal = compute_l2_error_nodal(x, u_sofa, length, q, E)
            err_mid = compute_l2_error_midpoints(x, u_sofa, length, q, E)
            
            l2_nodal.append(err_nodal)
            l2_midpoints.append(err_mid)
            element_sizes.append(h)
            
            rate_str = ""
            if i > 0:
                rate = np.log(err_mid / l2_midpoints[i-1]) / np.log(h / element_sizes[i-1])
                rate_str = f"{rate:.2f}"
            
            print(f"{nx:6d} | {h:10.4f} | {err_nodal:20.6e} | {err_mid:22.6e} | {rate_str:>10}")
            f.write(f"{nx:6d} | {h:10.4f} | {err_nodal:20.6e} | {err_mid:22.6e} | {rate_str:>10}\n")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    
    axes[0].loglog(element_sizes, l2_nodal, 'rs--', linewidth=2, markersize=8,
                   label='Erreur L2 aux nœuds (superconvergence)')
    axes[0].loglog(element_sizes, l2_midpoints, 'bo-', linewidth=2, markersize=8,
                   label='Erreur L2 aux milieux (vraie erreur)')
    
    h_ref = np.array([element_sizes[0], element_sizes[-1]])
    err_ref = l2_midpoints[0] * (h_ref / element_sizes[0])**2
    axes[0].loglog(h_ref, err_ref, 'k--', linewidth=1.5, label='Référence O(h²)')
    
    axes[0].set_xlabel('Taille d\'élément h ', fontsize=12)
    axes[0].set_ylabel('Erreur L2', fontsize=12)
    axes[0].set_title('Convergence de l\'erreur L2', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, which='both')
    
    nx_demo = 4
    h_demo = length / (nx_demo - 1)
    x_demo, u_demo = run_simulation(length, q, E, nx_demo)
    
    x_elem = np.linspace(0, h_demo, 200)
    u_ex_elem = exact_solution_quadratic(x_elem, length, q, E)
    u_left = u_demo[0]
    u_right = u_demo[1]
    u_p1 = u_left + (u_right - u_left) * x_elem / h_demo
    
    axes[1].plot(x_elem, u_ex_elem * 1e3, 'k-', linewidth=2, label='Solution exacte (quadratique)')
    axes[1].plot(x_elem, u_p1 * 1e3, 'b--', linewidth=2, label='Interpolation P1')
    axes[1].plot([0, h_demo], [u_left*1e3, u_right*1e3], 'go', markersize=8, label='Nœuds FEM (exacts)')
    
    x_mid = h_demo / 2
    u_p1_mid = (u_left + u_right) / 2
    u_ex_mid = exact_solution_quadratic(np.array([x_mid]), length, q, E)[0]
    
    axes[1].annotate('', xy=(x_mid, u_ex_mid*1e3), xytext=(x_mid, u_p1_mid*1e3),
                     arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    axes[1].text(x_mid + h_demo*0.05, (u_p1_mid + u_ex_mid)/2 * 1e3,
                 f'Erreur max ≈ h²/8·u\'\'\n= {abs(u_ex_mid-u_p1_mid)*1e3:.4e} m',
                 color='red', fontsize=8)
    axes[1].axvline(x_mid, color='red', linestyle=':', linewidth=1, alpha=0.5)
    
    axes[1].set_xlabel('x (m)', fontsize=12)
    axes[1].set_ylabel('u(x) (×10⁻³ m)', fontsize=12)
    axes[1].set_title('Erreur d\'interpolation P1 au milieu de l\'élément', fontsize=14)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergence_scenario2.png', dpi=300, bbox_inches='tight')
    
    return element_sizes, l2_nodal, l2_midpoints


def main():

    length = 10.0
    q = 100.0  
    E = 1e6    
    nx = 5     
    
    x, u_sofa, u_exact = simulation_distributed(length, q, E, nx)
    
    nx_values = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 32, 64]
    convergence_study_distributed(length, q, E, nx_values)
    



def createScene(rootNode):
    """Entry point for runSofa GUI."""
    build_scene_distributed(rootNode,
                            length=10.0, q=100.0,
                            young_modulus=1e6,
                            nx=8)
    return rootNode


if __name__ == "__main__":
    main()







"""""
For a 1D bar with constant distributed load, the exact displacement is quadratic, 
while the FEM uses linear (P1) interpolation inside each element. 
At the nodes, both solutions coincide exactly due to superconvergence. 
However, between the nodes, the linear approximation deviates from the true quadratic solution.


The nodal displacements are exact up to machine precision for any mesh—this is nodal superconvergence 
in 1D P1 on a uniform mesh, a theoretically known result. 
To observe the expected convergence, we evaluate the L2 norm at the midpoints of the elements,
 where the P1 interpolation deviates from the quadratic solution. 
 This yields a convergence rate of 2.00 for all refinement levels, 
 consistent with the theoretical O(h^2) rate for P1 with an H^2 solution."

"""