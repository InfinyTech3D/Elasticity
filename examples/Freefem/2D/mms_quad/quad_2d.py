import numpy as np
import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "results_quads"
os.makedirs(RESULTS_DIR, exist_ok=True)


#  MMS :  ux = x^2(L-x)/L^2     uy = x(L-x)y/L^2


def ux_mms(x, y, L):  return x**2 * (L - x) / L**2
def uy_mms(x, y, L):  return x * (L - x) * y / L**2

def dux_dx(x, y, L):  return (2*x*L - 3*x**2) / L**2
def dux_dy(x, y, L):  return np.zeros_like(np.asarray(x, float))
def duy_dx(x, y, L):  return (L - 2*x) * y / L**2
def duy_dy(x, y, L):  return x * (L - x) / L**2

def sigma_xx(x, y, E, L):  return E * dux_dx(x, y, L)
def sigma_yy(x, y, E, L):  return E * duy_dy(x, y, L)
def sigma_xy(x, y, E, L):  return E * 0.5 * duy_dx(x, y, L)

def fx_body(x, y, E, L):
    return -(E*(2*L - 6*x)/L**2 + E*(L - 2*x)/(2*L**2))

def fy_body(x, y, E, L):
    return E*y / L**2

def traction(x, y, nx_c, ny_c, E, L):
    sxx = sigma_xx(x, y, E, L)
    syy = sigma_yy(x, y, E, L)
    sxy = sigma_xy(x, y, E, L)
    return sxx*nx_c + sxy*ny_c,  sxy*nx_c + syy*ny_c



#  Maillage Q1


def get_nodes_2d(L, nx, ny):
    dx = L/(nx-1);  dy = L/(ny-1)
    return np.array([[i*dx, j*dy]
                     for j in range(ny) for i in range(nx)], dtype=float)

def get_quads(nx, ny):

    quads = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            k00 =  j    * nx + i
            k10 =  j    * nx + (i + 1)
            k01 = (j+1) * nx + i
            k11 = (j+1) * nx + (i + 1)
            quads.append([k00, k10, k11, k01])
    return np.array(quads)




# Points et poids de Gauss 
_GAUSS_PTS  = np.array([-1.0/np.sqrt(3), 1.0/np.sqrt(3)])
_GAUSS_WTS  = np.array([1.0, 1.0])

def _shape_q1(xi, eta):
    """Fonctions de forme Q1"""
    N = 0.25 * np.array([
        (1 - xi)*(1 - eta),   
        (1 + xi)*(1 - eta),   
        (1 + xi)*(1 + eta),   
        (1 - xi)*(1 + eta),   
    ])
    dN_dxi = 0.25 * np.array([
        -(1 - eta),  (1 - eta),  (1 + eta), -(1 + eta)
    ])
    dN_deta = 0.25 * np.array([
        -(1 - xi), -(1 + xi),  (1 + xi),  (1 - xi)
    ])
    return N, dN_dxi, dN_deta


def compute_nodal_forces(nodes_2d, L, E, nx, ny):

    quads = get_quads(nx, ny)
    F = np.zeros((len(nodes_2d), 2))

    # Forces volumiques 
    for quad in quads:
        xe = nodes_2d[quad, 0]   # coordonnées x des 4 noeuds
        ye = nodes_2d[quad, 1]   # coordonnées y des 4 noeuds
        for xi, wi in zip(_GAUSS_PTS, _GAUSS_WTS):
            for eta, wj in zip(_GAUSS_PTS, _GAUSS_WTS):
                N, dN_dxi, dN_deta = _shape_q1(xi, eta)

                # Jacobien 
                J = np.array([
                    [dN_dxi  @ xe, dN_dxi  @ ye],
                    [dN_deta @ xe, dN_deta @ ye]
                ])
                detJ = np.linalg.det(J)

                # coord physiques 
                xg = N @ xe
                yg = N @ ye


                fx = fx_body(xg, yg, E, L)
                fy = fy_body(xg, yg, E, L)

                w = wi * wj * detJ
                for a, node in enumerate(quad):
                    F[node, 0] += N[a] * fx * w
                    F[node, 1] += N[a] * fy * w

    #  Traction de Neumann sur y = 0 

    eps = 1e-10
    for i in range(nx - 1):
        k0 = i;       x0, y0 = nodes_2d[k0]
        k1 = i + 1;   x1, y1 = nodes_2d[k1]
        # Longueur de l'arete
        Le = np.sqrt((x1-x0)**2 + (y1-y0)**2)
        for xi, wi in zip(_GAUSS_PTS, _GAUSS_WTS):
            # Parametrage
            t = 0.5*(xi + 1)                    
            xg = (1-t)*x0 + t*x1
            yg = (1-t)*y0 + t*y1
            Tx, Ty = traction(xg, yg, 0., -1., E, L)
            # Fonctions de forme 
            N0 = 0.5*(1 - xi)
            N1 = 0.5*(1 + xi)
            w = wi * Le / 2.0
            F[k0, 0] += N0 * Tx * w;  F[k0, 1] += N0 * Ty * w
            F[k1, 0] += N1 * Tx * w;  F[k1, 1] += N1 * Ty * w

    # Neumann y=L 
    for i in range(nx - 1):
        k0 = (ny-1)*nx + i;      x0, y0 = nodes_2d[k0]
        k1 = (ny-1)*nx + i + 1;  x1, y1 = nodes_2d[k1]
        Le = np.sqrt((x1-x0)**2 + (y1-y0)**2)
        for xi, wi in zip(_GAUSS_PTS, _GAUSS_WTS):
            t = 0.5*(xi + 1)
            xg = (1-t)*x0 + t*x1
            yg = (1-t)*y0 + t*y1
            Tx, Ty = traction(xg, yg, 0., +1., E, L)
            N0 = 0.5*(1 - xi)
            N1 = 0.5*(1 + xi)
            w = wi * Le / 2.0
            F[k0, 0] += N0 * Tx * w;  F[k0, 1] += N0 * Ty * w
            F[k1, 0] += N1 * Tx * w;  F[k1, 1] += N1 * Ty * w

    return F


def createScene(rootNode, L=1.0, E=1e6, nx=10, ny=10, with_visual=True):
    rootNode.addObject('RequiredPlugin', name='Sofa.component.Visual')
    rootNode.addObject('RequiredPlugin', pluginName=[
        "Elasticity",
        "Sofa.Component.Constraint.Projective",
        "Sofa.Component.Engine.Select",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.MechanicalLoad",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Dynamic",
    ])
    rootNode.addObject('DefaultAnimationLoop')

    if with_visual:
        rootNode.addObject('VisualStyle',
                           displayFlags="showBehaviorModels showForceFields")

    nodes_2d = get_nodes_2d(L, nx, ny)
    quads_np = get_quads(nx, ny)
    Beam     = rootNode.addChild('Beam')

    Beam.addObject('StaticSolver',    name="staticSolver", printLog=False)
    Beam.addObject('SparseLDLSolver', name="linearSolver",
                   template="CompressedRowSparseMatrixd")

    dofs = Beam.addObject('MechanicalObject',
                          name="dofs", template="Vec2d",
                          position=nodes_2d.tolist(),
                          showObject=with_visual,
                          showObjectScale=0.005*L)

    Beam.addObject('QuadSetTopologyContainer',
                   name="topology", quads=quads_np.tolist())
    Beam.addObject('QuadSetTopologyModifier')
    Beam.addObject('LinearSmallStrainFEMForceField',
                   name="FEM", template="Vec2d",
                   youngModulus=E, poissonRatio=0.0,
                   topology="@topology")

    e = 1e-4 * L / max(nx-1, ny-1)

    Beam.addObject('BoxROI', name="box_left",  template="Vec2d",
                   box=[-e, -e, 0, e, L+e, 0], drawBoxes=with_visual)
    Beam.addObject('FixedProjectiveConstraint', name="fix_left",
                   template="Vec2d", indices="@box_left.indices")

    Beam.addObject('BoxROI', name="box_right", template="Vec2d",
                   box=[L-e, -e, 0, L+e, L+e, 0], drawBoxes=with_visual)
    Beam.addObject('FixedProjectiveConstraint', name="fix_right",
                   template="Vec2d", indices="@box_right.indices")

    F   = compute_nodal_forces(nodes_2d, L, E, nx, ny)
    idx = " ".join(str(k) for k in range(len(nodes_2d)))
    frc = " ".join(f"{F[k,0]} {F[k,1]}" for k in range(len(nodes_2d)))
    Beam.addObject('ConstantForceField',
                   name="MMS_forces", template="Vec2d",
                   indices=idx, forces=frc)

    return dofs, nodes_2d, quads_np


# Simu  

def run_simulation(L, E, nx, ny):
    root = Sofa.Core.Node("root")
    dofs, nodes_2d, quads_np = createScene(root, L=L, E=E,
                                            nx=nx, ny=ny, with_visual=False)
    Sofa.Simulation.init(root)
    pos0 = dofs.position.array().copy()
    Sofa.Simulation.animate(root, root.dt.value)
    pos1 = dofs.position.array().copy()
    Sofa.Simulation.unload(root)
    return nodes_2d, quads_np, pos1[:,0]-pos0[:,0], pos1[:,1]-pos0[:,1]



#  Erreur L2  

def compute_l2(nodes_2d, ux, uy, L, quads_np):
    err2 = 0.0
    for quad in quads_np:
        xe = nodes_2d[quad, 0]
        ye = nodes_2d[quad, 1]
        for xi, wi in zip(_GAUSS_PTS, _GAUSS_WTS):
            for eta, wj in zip(_GAUSS_PTS, _GAUSS_WTS):
                N, dN_dxi, dN_deta = _shape_q1(xi, eta)
                J = np.array([
                    [dN_dxi  @ xe, dN_dxi  @ ye],
                    [dN_deta @ xe, dN_deta @ ye]
                ])
                detJ = np.linalg.det(J)
                xg = N @ xe
                yg = N @ ye
                ux_h = N @ ux[quad]
                uy_h = N @ uy[quad]
                ex = ux_h - ux_mms(xg, yg, L)
                ey = uy_h - uy_mms(xg, yg, L)
                err2 += (ex**2 + ey**2) * wi * wj * detJ
    return np.sqrt(err2)



#  Erreur H1  

def compute_h1(nodes_2d, ux, uy, L, quads_np):
    err2 = 0.0
    for quad in quads_np:
        xe = nodes_2d[quad, 0]
        ye = nodes_2d[quad, 1]
        for xi, wi in zip(_GAUSS_PTS, _GAUSS_WTS):
            for eta, wj in zip(_GAUSS_PTS, _GAUSS_WTS):
                N, dN_dxi, dN_deta = _shape_q1(xi, eta)
                J = np.array([
                    [dN_dxi  @ xe, dN_dxi  @ ye],
                    [dN_deta @ xe, dN_deta @ ye]
                ])
                detJ = np.linalg.det(J)
                Jinv = np.linalg.inv(J)

                xg = N @ xe
                yg = N @ ye

            
                dN_dx = Jinv[0,0]*dN_dxi + Jinv[1,0]*dN_deta
                dN_dy = Jinv[0,1]*dN_dxi + Jinv[1,1]*dN_deta

                dux_dx_h = dN_dx @ ux[quad]
                dux_dy_h = dN_dy @ ux[quad]
                duy_dx_h = dN_dx @ uy[quad]
                duy_dy_h = dN_dy @ uy[quad]

                
                dux_dx_e = dux_dx(xg, yg, L)
                dux_dy_e = dux_dy(xg, yg, L)
                duy_dx_e = duy_dx(xg, yg, L)
                duy_dy_e = duy_dy(xg, yg, L)

                err2 += (
                    (dux_dx_h - dux_dx_e)**2 +
                    (dux_dy_h - dux_dy_e)**2 +
                    (duy_dx_h - duy_dx_e)**2 +
                    (duy_dy_h - duy_dy_e)**2
                ) * wi * wj * detJ

    return np.sqrt(err2)




def simulation_ponctuelle(L, E, nx, ny):
    nodes_2d, quads_np, ux, uy = run_simulation(L, E, nx, ny)
    l2 = compute_l2(nodes_2d, ux, uy, L, quads_np)
    h1 = compute_h1(nodes_2d, ux, uy, L, quads_np)

    ux_ref = ux_mms(nodes_2d[:,0], nodes_2d[:,1], L)
    uy_ref = uy_mms(nodes_2d[:,0], nodes_2d[:,1], L)

    print(f"\n[Vec2d Q1]  nx={nx} ny={ny}  L={L}")
    print(f"  L2           = {l2:.4e}")
    print(f"  H1 semi-norm = {h1:.4e}")
    print(f"  max|ux-ux_mms| = {np.max(np.abs(ux-ux_ref)):.4e}")
    print(f"  max|uy-uy_mms| = {np.max(np.abs(uy-uy_ref)):.4e}")

    mid_j = (ny-1)//2
    sl    = slice(mid_j*nx, mid_j*nx+nx)
    yc    = nodes_2d[mid_j*nx, 1]
    xf    = np.linspace(0, L, 300)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, u_sofa, u_fn, lbl, fmt in zip(
            axes,
            [ux[sl], uy[sl]],
            [lambda x: ux_mms(x, yc, L), lambda x: uy_mms(x, yc, L)],
            [r'$u_x$', r'$u_y$'], ['o-', 's-']):
        ax.plot(nodes_2d[sl, 0], u_sofa, fmt, color='tab:green',
                label='SOFA Q1', ms=5)
        ax.plot(xf, u_fn(xf), '--', color='tab:blue', label='MMS exact')
        ax.set_xlabel('x');  ax.set_ylabel(lbl)
        ax.legend();  ax.grid(True, alpha=0.3)
    plt.suptitle(f'MMS 2D — Vec2d Q1  nx={nx}  |L2={l2:.2e}  H1={h1:.2e}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'solution_nx{nx}.png'), dpi=150)
    plt.close()

    tris_plot = []
    for q in quads_np:
        tris_plot.append([q[0], q[1], q[2]])
        tris_plot.append([q[0], q[2], q[3]])
    tris_plot = np.array(tris_plot)

    x  = nodes_2d[:,0];  y = nodes_2d[:,1]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, data, title, cmap in [
        (axes[0,0], ux,                r'$u_x$ SOFA Q1',        'gray'),
        (axes[0,1], ux_ref,            r'$u_x$ MMS',            'gray'),
        (axes[0,2], np.abs(ux-ux_ref), r'$|u_x - u_x^{MMS}|$', 'gray_r'),
        (axes[1,0], uy,                r'$u_y$ SOFA Q1',        'gray'),
        (axes[1,1], uy_ref,            r'$u_y$ MMS',            'gray'),
        (axes[1,2], np.abs(uy-uy_ref), r'$|u_y - u_y^{MMS}|$', 'gray_r'),
    ]:
        tc = ax.tricontourf(x, y, tris_plot.tolist(), data, levels=20, cmap=cmap)
        ax.triplot(x, y, tris_plot.tolist(), 'k-', lw=0.3, alpha=0.4)
        plt.colorbar(tc, ax=ax, shrink=0.8)
        ax.set_title(title);  ax.set_aspect('equal')
        ax.set_xlabel('x');   ax.set_ylabel('y')
    plt.suptitle(f'Champs 2D — Vec2d Q1  nx={nx}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'champs2D_nx{nx}.png'), dpi=150)
    plt.close()
    



def convergence_study(L, E, nx_values):
    hs, errs_l2, errs_h1 = [], [], []
    hdr = (f"{'nx':>5} | {'h':>8} | {'L2':>14} | {'ord_L2':>7}"
           f" | {'H1':>14} | {'ord_H1':>7}")

    with open(os.path.join(RESULTS_DIR, 'convergence_Vec2d_Q1.txt'), 'w') as f:
        f.write("Convergence MMS 2D  Q1\n")
        f.write(hdr + "\n")

        for k, nx in enumerate(nx_values):
            ny = nx;  h = L/(nx-1)
            nodes_2d, quads_np, ux, uy = run_simulation(L, E, nx, ny)
            l2 = compute_l2(nodes_2d, ux, uy, L, quads_np)
            h1 = compute_h1(nodes_2d, ux, uy, L, quads_np)
            hs.append(h);  errs_l2.append(l2);  errs_h1.append(h1)

            ord_l2 = (f"{np.log(l2/errs_l2[k-1])/np.log(h/hs[k-1]):.2f}"
                      if k > 0 else "")
            ord_h1 = (f"{np.log(h1/errs_h1[k-1])/np.log(h/hs[k-1]):.2f}"
                      if k > 0 else "")
            line = (f"{nx:5d} | {h:8.4f} | {l2:14.6e} | {ord_l2:>7}"
                    f" | {h1:14.6e} | {ord_h1:>7}")
            print(line);  f.write(line + "\n")

    hs_arr    = np.array(hs)
    errs_l2_a = np.array(errs_l2)
    errs_h1_a = np.array(errs_h1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(hs_arr, errs_l2_a, 'o-',  color='tab:blue', lw=2, ms=8,
              label='L2')
    ax.loglog(hs_arr, errs_h1_a, 's--', color='tab:red', lw=2, ms=8,
              label='H1 semi-norme')

    h_ref = np.array([hs_arr[0], hs_arr[-1]])
    ax.loglog(h_ref, errs_l2_a[0]*(h_ref/hs_arr[0])**2,
              ':', color='tab:blue',  lw=1.2, label='O(h^2)')
    ax.loglog(h_ref, errs_h1_a[0]*(h_ref/hs_arr[0])**1,
              ':', color='tab:red', lw=1.2, label='O(h^1)')

    ax.set_xlabel('h');  ax.set_ylabel('Erreur')
    ax.set_title('Convergence MMS 2D — Vec2d Q1')
    ax.legend();  ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'convergence_Vec2d_Q1.png'), dpi=150)
    plt.close()



if __name__ == "__main__":
    L, E = 1.0, 1e6
    simulation_ponctuelle(L, E, nx=10, ny=10)
    convergence_study(L, E, nx_values=[3, 4, 5, 6, 8, 10, 12, 16, 100])