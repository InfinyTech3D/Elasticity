"""
Sinusoidal :  MMS u(x) = sin(pi * x)
Equilibre : E*u'' + f = 0  =>  f(x) = E * pi^2 * sin(pi * x)
BC: u(0) = 0 (Dirichlet),  E*u'(L) = E*pi*cos(pi*L) (Neumann)

This code computes the deformation of a bar subjected to a force that varies as sin(pi*x).

Note: The nodal force values passed to the ConstantForceField are computed using a 2-point Gauss quadrature because f(x) is not constant within the element. 

The H1 error must be evaluated using 2 Gauss points to correctly recover the theoretical O(h^1) convergence rate for the H1 semi-norm with P1 elements. 
The midpoint of a P1 element is a superconvergence point for the gradient, so evaluating only there artificially gives O(h^2) for H1.
"""

import numpy as np
import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def u_mms(x, L):
    return np.sin(np.pi * x)

def du_mms(x):
    return np.pi * np.cos(np.pi * x)


def build_scene(root, length, young_modulus, poisson_ratio, nx):
    root.addObject('RequiredPlugin', pluginName=[
        "Elasticity",
        "Sofa.Component.Constraint.Projective",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.MechanicalLoad",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Dynamic"
    ])
    root.addObject('DefaultAnimationLoop')

    h = length / (nx - 1)
    positions = [[i * h] for i in range(nx)]
    edges     = [[i, i + 1] for i in range(nx - 1)]

    Bar = root.addChild('Bar')
    Bar.addObject('NewtonRaphsonSolver',
                  name="newtonSolver",
                  maxNbIterationsNewton=1,
                  absoluteResidualStoppingThreshold=1e-12,
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
    Bar.addObject('EdgeSetTopologyContainer', name="topology", edges=edges)
    Bar.addObject('LinearSmallStrainFEMForceField',
                  name="FEM",
                  template="Vec1d",
                  youngModulus=young_modulus,
                  poissonRatio=poisson_ratio,
                  topology="@topology")
    Bar.addObject('FixedProjectiveConstraint', indices="0")

    #  Gauss Quadrature 
    C          = young_modulus * (np.pi ** 2)   # factor: E * pi^2
    forces     = np.zeros(nx)
    gauss_pts  = np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)])
    gauss_w    = np.array([1.0, 1.0])

    for elem in range(nx - 1):
        x0 = elem * h
        x1 = (elem + 1) * h
        for gp, w in zip(gauss_pts, gauss_w):
            xi    = 0.5 * (x0 + x1) + 0.5 * h * gp
            f_val = C * np.sin(np.pi * xi)
            phi0  = (x1 - xi) / h       
            phi1  = (xi - x0) / h          
            forces[elem]     += w * f_val * phi0 * (h / 2.0)
            forces[elem + 1] += w * f_val * phi1 * (h / 2.0)

    all_indices = " ".join(str(i) for i in range(nx))
    all_forces  = " ".join(f"{fi:.15e}" for fi in forces)
    Bar.addObject('ConstantForceField', indices=all_indices, forces=all_forces)

    # ── Neumann BC 
    F_tip = young_modulus * np.pi * np.cos(np.pi * length)
    Bar.addObject('ConstantForceField',
                  name="NeumannTip",
                  indices=f"{nx - 1}",
                  forces=f"{F_tip:.15e}")

    return dofs_ref



def run_simulation(length, young_modulus, poisson_ratio, nx):
    root     = Sofa.Core.Node("root")
    dofs_ref = build_scene(root, length, young_modulus, poisson_ratio, nx)
    Sofa.Simulation.init(root)
    x_initial  = dofs_ref.position.array().copy().flatten()
    Sofa.Simulation.animate(root, root.dt.value)
    u_computed = dofs_ref.position.array().flatten() - x_initial
    Sofa.Simulation.unload(root)
    return x_initial, u_computed



# ERROR NORMS


def compute_l2_nodal(x_nodes, u_computed, length):
    
    return np.sqrt(np.trapezoid((u_computed - u_mms(x_nodes, length))**2, x_nodes))


def compute_l2_midpoints(x_nodes, u_computed, length):
    
    idx = np.argsort(x_nodes)
    x_s, u_s = x_nodes[idx], u_computed[idx]
    l2_sq = 0.0
    for i in range(len(x_s) - 1):
        dx     = x_s[i + 1] - x_s[i]
        x_mid  = (x_s[i] + x_s[i + 1]) / 2.0
        u_mid  = (u_s[i] + u_s[i + 1]) / 2.0
        l2_sq += (u_mid - u_mms(x_mid, length))**2 * dx
    return np.sqrt(l2_sq)


def compute_h1_error(x_nodes, u_computed, length):

    idx = np.argsort(x_nodes)
    x_s, u_s = x_nodes[idx], u_computed[idx]

    
    gauss_pts = np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)])
    gauss_w   = np.array([1.0, 1.0])

    l2_sq = 0.0
    h1_sq = 0.0

    for i in range(len(x_s) - 1):
        dx    = x_s[i + 1] - x_s[i]
        x_mid = (x_s[i] + x_s[i + 1]) / 2.0

        
        u_mid  = (u_s[i] + u_s[i + 1]) / 2.0
        l2_sq += (u_mid - u_mms(x_mid, length))**2 * dx

        
        du_h = (u_s[i + 1] - u_s[i]) / dx

    
        for gp, w in zip(gauss_pts, gauss_w):
            x_gp   = x_mid + 0.5 * dx * gp      
            du_ex  = du_mms(x_gp)                
            h1_sq += w * (du_h - du_ex)**2 * (dx / 2.0)

    h1_semi = np.sqrt(h1_sq)
    h1_full = np.sqrt(l2_sq + h1_sq)
    return h1_semi, h1_full




def sol_mms(length, young_modulus, poisson_ratio, nx=10):
    x, u_sofa = run_simulation(length, young_modulus, poisson_ratio, nx)
    u_exact   = u_mms(x, length)

    filepath = os.path.join(RESULTS_DIR, 'resultats_sin_mms.txt')
    with open(filepath, 'w') as f:
        f.write(f"L={length} m, E={young_modulus:.1e} Pa, nx={nx}\n")
        f.write(f"{'x':>8} | {'SOFA':>14} | {'MMS':>14} | {'Error':>14}\n")
        f.write("-" * 55 + "\n")
        for xi, us, ue in zip(x, u_sofa, u_exact):
            f.write(f"{xi:8.4f} | {us:14.6e} | {ue:14.6e} | {abs(us - ue):14.6e}\n")

        en        = compute_l2_nodal(x, u_sofa, length)
        em        = compute_l2_midpoints(x, u_sofa, length)
        hs, hf    = compute_h1_error(x, u_sofa, length)
       # ratio_h1_l2 = hs / em if em > 1e-20 else float('nan')

        f.write(f"\nL2 Nodes  : {en:.6e}\n")
        f.write(f"L2 Mid    : {em:.6e}\n")
        f.write(f"H1 Semi   : {hs:.6e}\n")
        f.write(f"H1 Full   : {hf:.6e}\n")
        

    print(f"  L2 Nodes = {en:.6e}")
    print(f"  L2 Mid   = {em:.6e}")
    print(f"  H1 Semi  = {hs:.6e}")
   # print(f"  H1/L2    = {ratio_h1_l2:.4f}  (theoretical ~~ = {np.pi:.4f})")

    
    x_fine = np.linspace(0, length, 300)
    plt.figure(figsize=(10, 5))
    plt.plot(x_fine, u_mms(x_fine, length), 'k-',
             label=r'Exact $u(x)=\sin(\pi x)$', linewidth=2, alpha=0.8)
    plt.plot(x, u_sofa, 'ro-',
             label='SOFA (FEM P1)', markersize=8, linewidth=2)
    plt.xlabel('Position x ')
    plt.ylabel('Displacement u(x)')
    plt.title(f'MMS Verification: Sinusoidal Load (L={length}m, nx={nx})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'sin_mms_result.png'), dpi=300)
    plt.close()
    



def convergence_study(length, young_modulus, poisson_ratio, nx_values):
    hs_list, l2_list, h1_list = [], [], []
    filepath = os.path.join(RESULTS_DIR, 'convergence_sin_mms.txt')

    

    with open(filepath, 'w') as f:
        f.write("\n ------- \n")

        for k, nx in enumerate(nx_values):
            h       = length / (nx - 1)
            x_nodes = np.linspace(0, length, nx)
            _, u_sofa = run_simulation(length, young_modulus, poisson_ratio, nx)

            em  = compute_l2_midpoints(x_nodes, u_sofa, length)
            hs1, _ = compute_h1_error(x_nodes, u_sofa, length)

            rate_l2 = rate_h1 = ""
            if k > 0:
                log_h = np.log(h / hs_list[-1])
                if em > 1e-15 and l2_list[-1] > 1e-15:
                    rate_l2 = f"{np.log(em / l2_list[-1]) / log_h:.2f}"
                if hs1 > 1e-15 and h1_list[-1] > 1e-15:
                    rate_h1 = f"{np.log(hs1 / h1_list[-1]) / log_h:.2f}"

            row = (f"{nx:4d} | {h:7.4f} | {em:12.6e} | {hs1:12.6e} "
                   f"| {rate_l2:>8} | {rate_h1:>8}")
            
            f.write(row + "\n")

            hs_list.append(h)
            l2_list.append(em)
            h1_list.append(hs1)

    
    h_arr  = np.array(hs_list)
    l2_arr = np.array(l2_list)
    h1_arr = np.array(h1_list)
    h_ref  = np.array([h_arr[0], h_arr[-1]])

    plt.figure(figsize=(8, 5))
    plt.loglog(h_arr, l2_arr, 'bo-',  label='L2 Error (Midpoints)', markersize=8)
    plt.loglog(h_arr, h1_arr, 'rs--', label='H1 Semi-Norm Error',    markersize=8)

    plt.loglog(h_ref, l2_arr[0] * (h_ref / h_arr[0])**2,
               'b:', linewidth=2, label='O(h²) reference')
    plt.loglog(h_ref, h1_arr[0] * (h_ref / h_arr[0])**1,
               'r:', linewidth=2, label='O(h¹) reference')

    plt.xlabel('Element size h')
    plt.ylabel('Error Norm')
    plt.title('Convergence Analysis: Sinusoidal MMS\n'
              'Expected L2 O(h2),  H1 semi  O(h1)', fontsize=13)
    plt.legend()
    plt.grid(True, alpha=0.4, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'convergence_sin_mms.png'), dpi=300)
    plt.close()
    

def main():
    length = 1.0
    E, nu  = 1e6, 0.0
    nx = 100
    print(f"Running validation  L={length}, nx={nx} ...")
    sol_mms(length, E, nu, nx)

    
    nx_values = [10, 20, 40, 80, 160]
    convergence_study(length, E, nu, nx_values)


if __name__ == "__main__":
    main()