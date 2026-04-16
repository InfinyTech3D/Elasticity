"""
Quadratic MMS :  u_mms(x) = x*(L-x)/L^2
Bar fixed at x=0, constant distributed load q applied along the bar.

eq d'equilibre  E*u'' + f = 0 :
    f_body = 2*E / L^2

BC: 
    u(0) = 0            (Dirichlet)          
    u'(L) = -1/L        (Neumann)

"""

import numpy as np
import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime
import matplotlib.pyplot as plt
import os



results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

def u_mms(x, L):
    return x * (L - x) / L**2

def build_scene(root, length, young_modulus, nx):

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

    Bar.addObject('EdgeSetTopologyContainer',  name="topology", edges=edges)
    Bar.addObject('LinearSmallStrainFEMForceField',
                  name="FEM",
                  template="Vec1d",
                  youngModulus=young_modulus,
                  poissonRatio=0.0,
                  topology="@topology")

    Bar.addObject('FixedProjectiveConstraint', indices="0")

    f_body              = 2.0 * young_modulus / (length ** 2)
    nodal_forces        = [f_body * h] * nx
    nodal_forces[0]     = f_body * h / 2.0   
    nodal_forces[nx-1]  = f_body * h / 2.0   

    all_indices = " ".join(str(i)  for i in range(nx))
    all_forces  = " ".join(f"{fi}" for fi in nodal_forces)
    Bar.addObject('ConstantForceField', indices=all_indices, forces=all_forces)

    Bar.addObject('ConstantForceField',
                  name="NeumannTip",
                  indices=f"{nx - 1}",
                  forces=f"{-young_modulus / length}")

    return dofs_ref

def run_simulation(length, young_modulus, nx):
    root     = Sofa.Core.Node("root")
    dofs_ref = build_scene(root, length, young_modulus, nx)
    Sofa.Simulation.init(root)
    x_initial  = dofs_ref.position.array().copy().flatten()
    Sofa.Simulation.animate(root, root.dt.value)
    u_computed = dofs_ref.position.array().flatten() - x_initial
    Sofa.Simulation.unload(root)
    return x_initial, u_computed

# L^2 errors ====> 2 approaches 
# approach I: nodal error 
def compute_l2_nodal(x_nodes, u_computed, length):
    u_ex = u_mms(x_nodes, length)
    err  = u_computed - u_ex
    return np.sqrt(np.trapezoid(err**2, x_nodes))

# approach II : midpoint error 
def compute_l2_midpoints(x_nodes, u_computed, length):
    idx = np.argsort(x_nodes)
    x_s = x_nodes[idx];  u_s = u_computed[idx]
    l2  = 0.0
    for i in range(len(x_s) - 1):
        x_mid    = (x_s[i] + x_s[i+1]) / 2.0
        u_interp = (u_s[i] + u_s[i+1]) / 2.0
        u_ex_mid = u_mms(x_mid, length)
        l2      += (u_interp - u_ex_mid)**2 * (x_s[i+1] - x_s[i])
    return np.sqrt(l2)

def simulation_mms(length, young_modulus, nx):

    x, u_sofa = run_simulation(length, young_modulus, nx)
    u_exact   = u_mms(x, length)

    #print(f"\n{'x ':>10} | {'u SOFA':>15} | {'u MMS':>15} | {'Error':>15}")

    
    results_file = os.path.join(results_dir, 'resultats_quadratic_mms.txt')
    with open(results_file, 'w') as f:
        f.write(f"L={length} m,  E={young_modulus:.1e} Pa,  nx={nx}\n\n")
        for xi, us, ue in zip(x, u_sofa, u_exact):
            err  = abs(us - ue)
            line = f"{xi:10.4f} | {us:15.6e} | {ue:15.6e} | {err:15.6e}"
            #print(line)
            f.write(line + "\n")

        en = compute_l2_nodal(x, u_sofa, length)
        em = compute_l2_midpoints(x, u_sofa, length)
        summary = (
            f"\n Nodal error L^2   = {en:.6e}   superconvergence ~~ 0\n"
            f"Midpoint error L2   = {em:.6e}  real P1 error \n"
        )
        #print(summary)
        f.write(summary)

    x_fine  = np.linspace(0, length, 300)
    u_fine  = u_mms(x_fine, length)
    plt.figure(figsize=(10, 6))
    plt.plot(x,      u_sofa, 'bo-', label='SOFA',              markersize=7, linewidth=2)
    plt.plot(x_fine, u_fine, 'r--', label=r'MMS $x(L-x)/L^2$', linewidth=2)
    plt.xlabel('Position x ', fontsize=12)
    plt.ylabel('Displacement u(x)', fontsize=12)
    plt.title(f' MMS quadratic constant distributed load q applied along the bar.\nE={young_modulus:.1e} Pa, L={length} m, nx={nx}', fontsize=13)
    plt.legend(fontsize=11);  plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Sauvegarde dans le dossier results
    plot_file = os.path.join(results_dir, 'scenario2_mms.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    return x, u_sofa, u_exact

# Convergence 

def convergence_study(length, young_modulus, nx_values):

    

    hs, l2_nod, l2_mid = [], [], []


    conv_file = os.path.join(results_dir, 'convergence_quadratic_mms.txt')
    with open(conv_file, 'w') as f:
        f.write(f"{'nx':>6} | {'h':>10} | {'L2 nodes':>16} | {'L2 mid':>16} | {'Taux':>6}\n")

        for k, nx in enumerate(nx_values):
            h         = length / (nx - 1)
            x, u_sofa = run_simulation(length, young_modulus, nx)
            en        = compute_l2_nodal(x, u_sofa, length)
            em        = compute_l2_midpoints(x, u_sofa, length)

            hs.append(h);  l2_nod.append(en);  l2_mid.append(em)

            taux = ""
            if k > 0:
                taux = f"{np.log(em / l2_mid[k-1]) / np.log(h / hs[k-1]):.2f}"

            line = f"{nx:6d} | {h:10.4f} | {en:16.6e} | {em:16.6e} | {taux:>6}"
            #print(line)
            f.write(line + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].loglog(hs, l2_nod, 'rs--', label='Superconvergence', linewidth=2, markersize=8)
    axes[0].loglog(hs, l2_mid, 'bo-',  label=' Real error ',    linewidth=2, markersize=8)
    h_ref = np.array([hs[0], hs[-1]])
    axes[0].loglog(h_ref, l2_mid[0] * (h_ref / hs[0])**2, 'k--', label='O(h^2)', linewidth=1.5)
    axes[0].set_xlabel(" h ", fontsize=12)
    axes[0].set_ylabel('Error L2', fontsize=12)
    axes[0].set_title(' Convergence MMS quadratic', fontsize=13)
    axes[0].legend(fontsize=10);  axes[0].grid(True, alpha=0.3, which='both')

    nx_demo   = 4
    h_demo    = length / (nx_demo - 1)
    x_d, u_d  = run_simulation(length, young_modulus, nx_demo)
    x_e       = np.linspace(0, h_demo, 200)
    u_ex_e    = u_mms(x_e, length)
    u_p1      = u_d[0] + (u_d[1] - u_d[0]) * x_e / h_demo

    axes[1].plot(x_e, u_ex_e, 'k-',  linewidth=2, label=r'MMS $x(L-x)/L^2$')
    axes[1].plot(x_e, u_p1,   'b--', linewidth=2, label='Interpolation P1')
    axes[1].plot([0, h_demo], [u_d[0], u_d[1]], 'go', markersize=8, label='Nodes FEM ')
    x_mid     = h_demo / 2
    u_p1_mid  = (u_d[0] + u_d[1]) / 2
    u_ex_mid  = u_mms(x_mid, length)
    axes[1].annotate('', xy=(x_mid, u_ex_mid), xytext=(x_mid, u_p1_mid),
                     arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    axes[1].text(x_mid + h_demo * 0.05, (u_p1_mid + u_ex_mid) / 2,
                 f'Error = {abs(u_ex_mid - u_p1_mid):.4e}', color='red', fontsize=9)
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel('u(x)', fontsize=12)
    axes[1].set_title('Error P1 at mid-element', fontsize=13)
    axes[1].legend(fontsize=9);  axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    
    
    conv_plot_file = os.path.join(results_dir, 'convergence_quadratic_mms.png')
    plt.savefig(conv_plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    return hs, l2_nod, l2_mid

def main():
    length = 10.0
    E      = 1e6
    nx     = 10

    simulation_mms(length, E, nx)

    nx_values = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 32, 64]
    convergence_study(length, E, nx_values)
    

def createScene(rootNode):
    build_scene(rootNode, length=10.0, young_modulus=1e6, nx=10)
    return rootNode

if __name__ == "__main__":
    main()