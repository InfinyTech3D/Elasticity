"""
Sinusoidal :  MMS u(x) = sin(pi * x)
Equilibre : E*u'' + f = 0  =>  f(x) = E * pi^2 * sin(pi * x)
BC: u(0) = 0 (Dirichlet),  E*u'(L) = E*pi*cos(pi*L) (Neumann)
"""



"""
remarks
This code computes the deformation of a bar subjected to a force that varies 
as sin(pi*x). To be precise:

 2 Gauss points per element are used to compute the force (Gauss quadrature)
 At least 8 nodes along the entire bar are needed to properly capture the 
  5 oscillations of the sinusoidal force
 The solver is allowed 15 iterations to find the solution
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


def build_scene(root, length, young_modulus, poisson_ratio, nx):
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

    h         = length / (nx - 1)
    positions = [[i * h] for i in range(nx)]
    edges     = [[i, i + 1] for i in range(nx - 1)]

    Bar = root.addChild('Bar')
    Bar.addObject('NewtonRaphsonSolver',
                  name="newtonSolver",
                  maxNbIterationsNewton=15,
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

    
    C = young_modulus * (np.pi ** 2)
    forces = np.zeros(nx)
    gauss_pts = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    gauss_w   = np.array([1.0, 1.0])

    for elem in range(nx - 1):
        x0 = elem * h
        x1 = (elem + 1) * h
        for gp, w in zip(gauss_pts, gauss_w):
            xi      = 0.5 * (x0 + x1) + 0.5 * h * gp
            f_val   = C * np.sin(np.pi * xi)
            phi0    = (x1 - xi) / h
            phi1    = (xi - x0) / h
            forces[elem]     += w * f_val * phi0 * (h / 2.0)
            forces[elem + 1] += w * f_val * phi1 * (h / 2.0)

    all_indices = " ".join(str(i) for i in range(nx))
    all_forces  = " ".join(f"{fi:.15e}" for fi in forces)
    Bar.addObject('ConstantForceField', indices=all_indices, forces=all_forces)

    
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


def compute_l2_nodal(x_nodes, u_computed, length):
    u_ex = u_mms(x_nodes, length)
    err  = u_computed - u_ex
    return np.sqrt(np.trapezoid(err**2, x_nodes))

def compute_l2_midpoints(x_nodes, u_computed, length):
    idx  = np.argsort(x_nodes)
    x_s  = x_nodes[idx]
    u_s  = u_computed[idx]
    l2   = 0.0
    for i in range(len(x_s) - 1):
        x_mid    = (x_s[i] + x_s[i+1]) / 2.0
        u_interp = (u_s[i] + u_s[i+1]) / 2.0
        u_ex_mid = u_mms(x_mid, length)
        dx       = x_s[i+1] - x_s[i]
        l2      += (u_interp - u_ex_mid)**2 * dx
    return np.sqrt(l2)


def sol_mms(length, young_modulus, poisson_ratio, nx=10):
    x, u_sofa = run_simulation(length, young_modulus, poisson_ratio, nx)
    u_exact   = u_mms(x, length)

    #print(f"\n{'Position (x)':>12} | {'u SOFA':>14} | {'u MMS':>14} | {'Error abs.':>14}")
    

    filepath = os.path.join(RESULTS_DIR, 'resultats_sin_mms.txt')
    with open(filepath, 'w') as f:
        f.write(f"L={length} m,  E={young_modulus:.1e} Pa,  nx={nx}\n\n")
        f.write(f"{'Position (x)':>12} | {'u SOFA':>14} | {'u MMS':>14} | {'Error abs.':>14}\n")
        
        for xi, us, ue in zip(x, u_sofa, u_exact):
            err  = abs(us - ue)
            line = f"{xi:12.4f}  {us:14.6e}  {ue:14.6e}  {err:14.6e}"
            #print(line)
            f.write(line + "\n")

        err_nodal = compute_l2_nodal(x, u_sofa, length)
        err_mid   = compute_l2_midpoints(x, u_sofa, length)
        summary = (
            f"\nError L2 nodes    = {err_nodal:.6e}\n"
            f"Error L2 mid    = {err_mid:.6e}  <- real error P1\n"
            f" h       = {length/(nx-1):.4f} m\n"
        )
        #print(summary)
        f.write(summary)

    plt.figure(figsize=(10, 6))
    plt.plot(x, u_sofa,  'bo-', label='SOFA',               markersize=8, linewidth=2)
    plt.plot(x, u_exact, 'r--', label=r'MMS $\sin(\pi x)$', linewidth=2)
    plt.xlabel('Position x ', fontsize=12)
    plt.ylabel('Displacement u(x)', fontsize=12)
    plt.title(f' Sinusoidal MMS\nE={young_modulus:.1e} , L={length} , nx={nx}', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'sin_mms.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return x, u_sofa, u_exact

def convergence_study(length, young_modulus, poisson_ratio, nx_values):
    #print(f"\n{'nx':>6} | {'h':>10} | {'L2 noeuds':>16} | {'L2 milieux':>16} | {'Taux':>6}")


    hs, l2_nod, l2_mid = [], [], []
    filepath = os.path.join(RESULTS_DIR, 'convergence_sin_mms.txt')

    with open(filepath, 'w') as f:
        f.write(f"{'nx':>6} | {'h':>10} | {'L2 noeuds':>16} | {'L2 milieux':>16} | {'Taux':>6}\n")


        for k, nx in enumerate(nx_values):
            h          = length / (nx - 1)
            x, u_sofa  = run_simulation(length, young_modulus, poisson_ratio, nx)
            en         = compute_l2_nodal(x, u_sofa, length)
            em         = compute_l2_midpoints(x, u_sofa, length)

            hs.append(h); l2_nod.append(en); l2_mid.append(em)

            taux = ""
            if k > 0 and em > 1e-15 and l2_mid[k-1] > 1e-15:
                taux = f"{np.log(em / l2_mid[k-1]) / np.log(h / hs[k-1]):.2f}"
            elif k > 0:
                taux = "N/A"

            line = f"{nx:6d} | {h:10.4f} | {en:16.6e} | {em:16.6e} | {taux:>6}"
            #print(line)
            f.write(line + "\n")

    plt.figure(figsize=(8, 5))
    plt.loglog(hs, l2_nod, 'rs--', label='L2 nodes ', linewidth=2, markersize=8)
    plt.loglog(hs, l2_mid, 'bo-',  label='L2 mid real error', linewidth=2, markersize=8)
    h_ref  = np.array([hs[0], hs[-1]])
    e_ref  = l2_mid[0] * (h_ref / hs[0])**2
    plt.loglog(h_ref, e_ref, 'k--', label=' O(h2)', linewidth=1.5)
    plt.xlabel('h', fontsize=12)
    plt.ylabel('Error L2', fontsize=12)
    plt.title('Convergence MMS sinusoide', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'convergence_sin_mms.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return hs, l2_nod, l2_mid

def main():
    length = 10.0
    E      = 1e6
    nu     = 0.0
    nx     = 10

    sol_mms(length, E, nu, nx)

    
    nx_values = [8, 12, 16, 24, 32, 48, 64, 96]
    convergence_study(length, E, nu, nx_values)

def createScene(rootNode):
    build_scene(rootNode, length=10.0, young_modulus=1e6, poisson_ratio=0.3, nx=10)
    return rootNode

if __name__ == "__main__":
    main()