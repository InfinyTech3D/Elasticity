"""
Cubic MMS :  u_mms(x) = x^2*(L-x)/L^2
Bar fixed at x=0, distributed body force derived from MMS.

Equilibrium eq  E*u'' + f = 0 :
    u'(x)  = (2x*L - 3x^2) / L^2
    u''(x) = (2L - 6x)     / L^2
    => f(x) = -E * u''(x) = E*(6x - 2L) / L^2

BC:
    u(0)  = 0          (Dirichlet)
    u'(L) = -1         (Neumann)   =====> F_N = E * u'(L) = -E
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
    return x**2 * (L - x) / L**2

def f_body(x, E, L):
    return E * (6.0 * x - 2.0 * L) / L**2


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
                  maxNbIterationsNewton=1,
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

    Bar.addObject('EdgeSetTopologyContainer', name="topology", edges=edges)
    Bar.addObject('LinearSmallStrainFEMForceField',
                  name="FEM",
                  template="Vec1d",
                  youngModulus=young_modulus,
                  poissonRatio=0.0,
                  topology="@topology")

    Bar.addObject('FixedProjectiveConstraint', indices="0")



    # CORRIGÉ (forces consistantes exactes aux bords)
    nodal_forces = [f_body(i * h, young_modulus, length) * h for i in range(nx)]
# Correction superconvergence 
# Bord gauche : h/6 * [2*f(x_0) + f(x_1)]
    nodal_forces[0]      = (h / 6.0) * (2*f_body(0,young_modulus, length) + f_body(h,young_modulus, length))
# Bord droit  : h/6 * [f(x_{N-2}) + 2*f(x_{N-1})]
    nodal_forces[nx - 1] = (h / 6.0) * (f_body((nx-2)*h, young_modulus, length) + 2*f_body((nx-1)*h, young_modulus, length))  

    all_indices = " ".join(str(i)   for i in range(nx))
    all_forces  = " ".join(f"{fi}"  for fi in nodal_forces)
    Bar.addObject('ConstantForceField',
                  name="BodyForce",
                  indices=all_indices,
                  forces=all_forces)

    # Condition de Neumann au tip : F_N = E * u'(L) = -E
    Bar.addObject('ConstantForceField',
                  name="NeumannTip",
                  indices=f"{nx - 1}",
                  forces=f"{-young_modulus}")

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


#  Erreurs L2

def compute_l2_nodal(x_nodes, u_computed, length):
    """Nodes errors superconvergence P1 """
    u_ex = u_mms(x_nodes, length)
    err  = u_computed - u_ex
    return np.sqrt(np.trapezoid(err**2, x_nodes))

def compute_l2_midpoints(x_nodes, u_computed, length):
    """ error midpoint  """
    idx = np.argsort(x_nodes)
    x_s = x_nodes[idx]
    u_s = u_computed[idx]
    l2  = 0.0
    for i in range(len(x_s) - 1):
        x_mid    = (x_s[i] + x_s[i + 1]) / 2.0
        u_interp = (u_s[i] + u_s[i + 1]) / 2.0
        u_ex_mid = u_mms(x_mid, length)
        l2      += (u_interp - u_ex_mid)**2 * (x_s[i + 1] - x_s[i])
    return np.sqrt(l2)


def simulation_mms(length, young_modulus, nx):

    x, u_sofa = run_simulation(length, young_modulus, nx)
    u_exact   = u_mms(x, length)

    results_file = os.path.join(results_dir, 'cubic_results_mms.txt')
    with open(results_file, 'w') as f:
        f.write(f"L={length} m,  E={young_modulus:.1e} Pa,  nx={nx}\n")
        f.write(f"u_mms(x) = x^2*(L-x)/L^2\n\n")
        f.write(f"{'x':>10} | {'u SOFA':>15} | {'u MMS':>15} | {'Error':>15}\n")
        for xi, us, ue in zip(x, u_sofa, u_exact):
            err  = abs(us - ue)
            line = f"{xi:10.4f} | {us:15.6e} | {ue:15.6e} | {err:15.6e}"
            f.write(line + "\n")

        en = compute_l2_nodal(x, u_sofa, length)
        em = compute_l2_midpoints(x, u_sofa, length)
        summary = (
            f"\n Nodal error L2 = {en:.6e}  (superconvergence)\n"
            f" Midpoint error L2 = {em:.6e}  (real error P1)\n"
        )
        f.write(summary)

    x_fine = np.linspace(0, length, 300)
    u_fine = u_mms(x_fine, length)
    plt.figure(figsize=(10, 6))
    plt.plot(x,      u_sofa, 'bo-', label='SOFA',
             markersize=7, linewidth=2)
    plt.plot(x_fine, u_fine, 'r--',
             label=r'MMS $x^2(L-x)/L^2$', linewidth=2)
    plt.xlabel('Position x', fontsize=12)
    plt.ylabel('Déplacement u(x)', fontsize=12)
    plt.title(
        f'MMS cubique: distributed body force \n'
        f'E={young_modulus:.1e} Pa,  L={length} m,  nx={nx}',
        fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'SOFA vs mms .png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    return x, u_sofa, u_exact



def convergence_study(length, young_modulus, nx_values):

    hs, l2_nod, l2_mid = [], [], []

    conv_file = os.path.join(results_dir, 'convergence_cubic_mms.txt')
    with open(conv_file, 'w') as f:
        f.write(f"{'nx':>6} | {'h':>10} | {'L2 nodes':>16} | {'L2 mid':>16} | {'Taux':>6}\n")

        for k, nx in enumerate(nx_values):
            h         = length / (nx - 1)
            x, u_sofa = run_simulation(length, young_modulus, nx)
            en        = compute_l2_nodal(x, u_sofa, length)
            em        = compute_l2_midpoints(x, u_sofa, length)

            hs.append(h)
            l2_nod.append(en)
            l2_mid.append(em)

            taux = ""
            if k > 0:
                taux = f"{np.log(em / l2_mid[k-1]) / np.log(h / hs[k-1]):.2f}"

            line = f"{nx:6d} | {h:10.4f} | {en:16.6e} | {em:16.6e} | {taux:>6}"
            f.write(line + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].loglog(hs, l2_nod, 'rs--', label=' Nodes (superconvergence)',
                   linewidth=2, markersize=8)
    axes[0].loglog(hs, l2_mid, 'bo-',  label='Midpoints (real error) ',
                   linewidth=2, markersize=8)
    h_ref = np.array([hs[0], hs[-1]])
    axes[0].loglog(h_ref, l2_mid[0] * (h_ref / hs[0])**2, 'k--',
                   label='O(h2)', linewidth=1.5)
    axes[0].set_xlabel('h', fontsize=12)
    axes[0].set_ylabel('Error L2', fontsize=12)
    axes[0].set_title('Convergence MMS cubic', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, which='both')


    nx_demo  = 4
    h_demo   = length / (nx - 1)
    x_d, u_d = run_simulation(length, young_modulus, nx_demo)
    x_e      = np.linspace(0, h_demo, 200)
    u_ex_e   = u_mms(x_e, length)
    u_p1     = u_d[0] + (u_d[1] - u_d[0]) * x_e / h_demo

    axes[1].plot(x_e, u_ex_e, 'k-',  linewidth=2,
                 label=r'MMS $x^2(L-x)/L^2$')
    axes[1].plot(x_e, u_p1,   'b--', linewidth=2,
                 label='Interpolation P1')
    axes[1].plot([0, h_demo], [u_d[0], u_d[1]], 'go',
                 markersize=8, label='Nœuds FEM')
    x_mid    = h_demo / 2.0
    u_p1_mid = (u_d[0] + u_d[1]) / 2.0
    u_ex_mid = u_mms(x_mid, length)
    axes[1].annotate('', xy=(x_mid, u_ex_mid), xytext=(x_mid, u_p1_mid),
                     arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    axes[1].text(x_mid + h_demo * 0.05,
                 (u_p1_mid + u_ex_mid) / 2,
                 f'Erreur = {abs(u_ex_mid - u_p1_mid):.4e}',
                 color='red', fontsize=9)
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel('u(x)', fontsize=12)
    axes[1].set_title('Error P1 midpoint', fontsize=13)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'convergence_cubic_mms.png'),
                dpi=300, bbox_inches='tight')
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