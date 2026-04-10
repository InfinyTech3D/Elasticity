"""
Scenario 2: Constant distributed load across the bar
Double L2 error analysis:
  - Approach I : L2 evaluated AT nodes : superconvergence (error = machine precision)
  - Approach II : L2 evaluated at element MIDPOINTS : true O(h^2) convergence
Both are shown side by side to justify why midpoint evaluation is the correct choice.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime

RESULTS_DIR = "results_scenario2"


def compute_lame_coefficients(E, nu):
    lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu      = E / (2 * (1 + nu))
    E_oed   = lambda_ + 2 * mu
    return lambda_, mu, E_oed


def exact_solution_quadratic(x, length, q, E_oed, area=1.0):
    """u(x) = q / (E_oed * A) * (L*x - x²/2)"""
    return (q / (E_oed * area)) * (length * x - x**2 / 2.0)


def build_scene_distributed(root, length, q, young_modulus, poisson_ratio, nx):

    root.addObject('RequiredPlugin', pluginName=[
        "Elasticity",
        "Sofa.Component.Constraint.Projective",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.MechanicalLoad",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Dynamic",
        "Sofa.Component.Visual",
        "Sofa.GL.Component.Rendering3D",
    ])
    root.addObject('DefaultAnimationLoop')
    root.addObject('VisualStyle', displayFlags=["showBehaviorModels", "showForceFields"])

    h         = length / (nx - 1)
    positions = [[i * h] for i in range(nx)]
    edges     = [[i, i + 1] for i in range(nx - 1)]

    Bar = root.addChild('Bar')
    Bar.addObject('NewtonRaphsonSolver',
                  name="newtonSolver",
                  printLog=False,
                  maxNbIterationsNewton=10,
                  absoluteResidualStoppingThreshold=1e-10)
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
                             position=positions,
                             showObject=True,
                             showObjectScale=0.02)

    Bar.addObject('EdgeSetTopologyContainer', name="topology", edges=edges)
    Bar.addObject('LinearSmallStrainFEMForceField',
                  name="FEM",
                  template="Vec1d",
                  youngModulus=young_modulus,
                  poissonRatio=poisson_ratio,
                  topology="@topology")

    Bar.addObject('FixedProjectiveConstraint', indices="0")

    for i in range(1, nx):
        nodal_force = q * h / 2.0 if i == nx - 1 else q * h
        Bar.addObject('ConstantForceField',
                      indices=str(i),
                      forces=str(nodal_force),
                      showArrowSize=1e-4)

    return dofs_ref


def run_simulation(length, q, young_modulus, poisson_ratio, nx):
    root     = Sofa.Core.Node("root")
    dofs_ref = build_scene_distributed(root, length, q, young_modulus, poisson_ratio, nx)
    Sofa.Simulation.init(root)
    x_initial  = dofs_ref.position.array().copy().flatten()  
    Sofa.Simulation.animate(root, root.dt.value)
    x_final    = dofs_ref.position.array().flatten()
    u_computed = x_final - x_initial
    Sofa.Simulation.unload(root)
    return x_initial, u_computed


#  ================  Approach I : L2 at NODES ================
def compute_l2_error_nodal(x_nodes, u_computed, length, q, E_oed):
    """
    L2 error evaluated directly at the FEM nodes;
    In 1D P1 on a uniform mesh with consistent nodal lumping, the FEM
    solution is EXACT at nodes (superconvergence). The error is therefore
    at machine precision and does NOT reflect the approximation quality
    of the P1 interpolation between nodes.
    """
    idx  = np.argsort(x_nodes)
    x_s  = x_nodes[idx]
    u_s  = u_computed[idx]
    u_ex = exact_solution_quadratic(x_s, length, q, E_oed)
    err  = u_s - u_ex
    return np.sqrt(np.trapezoid(err**2, x_s))



# ===================  Approach II : L2 at element MIDPOINTS ==========================
def compute_l2_error_midpoints(x_nodes, u_computed, length, q, E_oed):
    """
    L2 error evaluated at element midpoints.

    The P1 interpolation is linear inside each element; the exact solution
    is quadratic;

    """
    idx      = np.argsort(x_nodes)
    x_s      = x_nodes[idx]
    u_s      = u_computed[idx]
    l2_error = 0.0
    for i in range(len(x_s) - 1):
        x_mid    = (x_s[i] + x_s[i+1]) / 2.0
        u_interp = (u_s[i] + u_s[i+1]) / 2.0   # linear inter
        u_ex     = exact_solution_quadratic(np.array([x_mid]), length, q, E_oed)[0]
        dx       = x_s[i+1] - x_s[i]
        l2_error += (u_interp - u_ex)**2 * dx
    return np.sqrt(l2_error)


# ================= Convergence study ===================
def run_convergence_study():

    length        = 10.0
    q             = 100.0
    young_modulus = 1e6
    poisson_ratio = 0.3
    mesh_sizes    = [2, 4, 8, 16, 32, 64]

    _, _, E_oed = compute_lame_coefficients(young_modulus, poisson_ratio)
    

    l2_nodal      = []
    l2_midpoints  = []
    element_sizes = []

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    for nx in mesh_sizes:
        h = length / (nx - 1)
        x_initial, u_computed = run_simulation(length, q, young_modulus, poisson_ratio, nx)

        err_nod = compute_l2_error_nodal(    x_initial, u_computed, length, q, E_oed)
        err_mid = compute_l2_error_midpoints(x_initial, u_computed, length, q, E_oed)

        l2_nodal.append(err_nod)
        l2_midpoints.append(err_mid)
        element_sizes.append(h)

        print(f"{nx:>6}  {h:>10.4f}  {err_nod:>16.6e}  {err_mid:>18.6e}")

        if nx in [2, 4, 8, 16, 32]:
            axes[0].plot(x_initial, u_computed, 'o-',
                         label=f'{nx-1} elements', markersize=4)

    # ── Plot 1 : displacement ========================
    x_fine = np.linspace(0, length, 300)
    u_fine = exact_solution_quadratic(x_fine, length, q, E_oed)
    axes[0].plot(x_fine, u_fine, 'k-', linewidth=2, label='Exact')
    axes[0].set_xlabel('x (m)')
    axes[0].set_ylabel('u(x) (m)')
    axes[0].set_title('Displacement : constant distributed load')
    axes[0].legend(fontsize=8)
    axes[0].grid(True)

    #  =========== Plot 2 : convergence comparison ===================
    h_arr = np.array(element_sizes)
    h_ref = np.array([h_arr[0], h_arr[-1]])

    axes[1].loglog(h_arr, l2_nodal,    'rs--', linewidth=2, markersize=7,
                   label='Approach I : at nodes\n(superconvergence)')
    axes[1].loglog(h_arr, l2_midpoints,'bo-',  linewidth=2, markersize=7,
                   label='Approach II — at midpoints\n(true interpolation error)')
    err_ref2 = l2_midpoints[0] * (h_ref / h_arr[0])**2
    axes[1].loglog(h_ref, err_ref2, 'k--', linewidth=1.5, label='O(h^2) reference')
    axes[1].set_xlabel("Element size h ")
    axes[1].set_ylabel("L2 error")
    axes[1].set_title("Convergence — Approach I vs Approach II")
    axes[1].legend(fontsize=7.5)
    axes[1].grid(True, which='both')

    nx_demo = 4
    h_demo  = length / (nx_demo - 1)
    u_left  = exact_solution_quadratic(np.array([0.0]),    length, q, E_oed)[0]
    u_right = exact_solution_quadratic(np.array([h_demo]), length, q, E_oed)[0]
    x_elem  = np.linspace(0, h_demo, 200)
    u_p1    = u_left + (u_right - u_left) * x_elem / h_demo
    u_ex_el = exact_solution_quadratic(x_elem, length, q, E_oed)

    axes[2].plot(x_elem, u_ex_el * 1e3, 'k-',  linewidth=2, label='Exact (quadratic)')
    axes[2].plot(x_elem, u_p1    * 1e3, 'b--', linewidth=2, label='P1 interpolation')
    axes[2].plot([0.0, h_demo],
                 [u_left*1e3, u_right*1e3], 'go', markersize=8,
                 label='FEM nodes : exact == superconv')

    x_mid    = h_demo / 2
    u_p1_mid = (u_left + u_right) / 2
    u_ex_mid = exact_solution_quadratic(np.array([x_mid]), length, q, E_oed)[0]
    axes[2].annotate('', xy=(x_mid, u_ex_mid*1e3), xytext=(x_mid, u_p1_mid*1e3),
                     arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
    axes[2].text(x_mid + h_demo*0.04,
                 (u_p1_mid + u_ex_mid) / 2 * 1e3,
                 f'max error\n≈ h²/8·u\'\'\n= {abs(u_ex_mid-u_p1_mid)*1e3:.4f}×10⁻³ m',
                 color='red', fontsize=7.5)
    axes[2].axvline(x_mid, color='red', linestyle=':', linewidth=1, alpha=0.5)
    axes[2].set_xlabel('x (m)')
    axes[2].set_ylabel('u(x) (×10⁻³ m)')
    axes[2].set_title(' midpoints\nMax P1 interpolation error on element 1')
    axes[2].legend(fontsize=7.5)
    axes[2].grid(True)

    # ===============  Convergence rates ========================
    for i in range(len(l2_midpoints) - 1):
        rate = (np.log(l2_midpoints[i+1]) - np.log(l2_midpoints[i])) / \
               (np.log(element_sizes[i+1]) - np.log(element_sizes[i]))
        print(f"  h: {element_sizes[i]:.4f} -> {element_sizes[i+1]:.4f},  rate: {rate:.2f}")

    for i in range(len(l2_nodal) - 1):
        if l2_nodal[i] > 1e-30 and l2_nodal[i+1] > 1e-30:
            rate = (np.log(l2_nodal[i+1]) - np.log(l2_nodal[i])) / \
                   (np.log(element_sizes[i+1]) - np.log(element_sizes[i]))
            print(f"  h: {element_sizes[i]:.4f} -> {element_sizes[i+1]:.4f},  "
                  f"rate: {rate:.2f}  (values at machine precision)")
        else:
            print(f"  h: {element_sizes[i]:.4f} -> {element_sizes[i+1]:.4f},  "
                  f"rate: N/A  (error = 0, exact superconvergence)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "convergence_analysis.png"), dpi=150)
    print(f"\nFigure saved : {RESULTS_DIR}/convergence_analysis.png")

    with open(os.path.join(RESULTS_DIR, "convergence_results.txt"), 'w') as f:
        json.dump({
            "element_sizes": element_sizes,
            "l2_nodal":      l2_nodal,
            "l2_midpoints":  l2_midpoints,
        }, f, indent=2)

    return element_sizes, l2_nodal, l2_midpoints


def test_nodal_superconvergence():

    length        = 10.0
    q             = 100.0
    young_modulus = 1e6
    poisson_ratio = 0.3
    _, _, E_oed   = compute_lame_coefficients(young_modulus, poisson_ratio)


    for nx in [5, 33]:
        print(f"Mesh with {nx-1} elements:")
        x_initial, u_computed = run_simulation(length, q, young_modulus, poisson_ratio, nx)
        u_exact = exact_solution_quadratic(x_initial, length, q, E_oed)

        print(f"  {'x':>10}  {'u_SOFA':>12}  {'u_exact':>12}  {'Error':>12}")
        max_err = 0.0
        for xi, uc, ue in zip(x_initial, u_computed, u_exact):
            err     = abs(uc - ue)
            max_err = max(max_err, err)
            print(f"  {xi:10.4f}  {uc:12.6e}  {ue:12.6e}  {err:12.6e}")



def createScene(rootNode):

    build_scene_distributed(rootNode,
                            length=10.0, q=100.0,
                            young_modulus=1e6, poisson_ratio=0.3,
                            nx=8)
    return rootNode


if __name__ == "__main__":
    element_sizes, l2_nodal, l2_midpoints = run_convergence_study()
    test_nodal_superconvergence()





""""
remark : 
“The nodal displacements are exact up to machine precision for any mesh—this is nodal superconvergence in 1D P1 on a uniform mesh,
 a theoretically known result. To observe the expected convergence, I evaluate the L2 norm at the midpoints of the elements, 
 where the P1 interpolation deviates from the quadratic solution. This yields a convergence rate of 2.00 for all refinement levels, 
 consistent with the theoretical O(h^2) rate for P1 with an H^2 solution.”
"""