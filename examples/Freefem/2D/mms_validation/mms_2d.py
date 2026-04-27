import numpy as np
import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "results_coupled"
os.makedirs(RESULTS_DIR, exist_ok=True)



#  ux = x^2(L-x)/L^2       uy = x(L-x)y/L^2


def ux_mms(x, y, L):  return x**2 * (L - x) / L**2
def uy_mms(x, y, L):  return x * (L - x) * y / L**2

def dux_dx(x, y, L):  return (2*x*L - 3*x**2) / L**2
def dux_dy(x, y, L):  return np.zeros_like(np.asarray(x, float))
def duy_dx(x, y, L):  return (L - 2*x) * y / L**2
def duy_dy(x, y, L):  return x * (L - x) / L**2

def sigma_xx(x, y, E, L):  return E * dux_dx(x, y, L)
def sigma_yy(x, y, E, L):  return E * duy_dy(x, y, L)
def sigma_xy(x, y, E, L):  return E * 0.5 * (duy_dx(x, y, L))      

def fx_body(x, y, E, L):
    return -(E*(2*L - 6*x)/L**2 + E*(L - 2*x)/(2*L**2))

def fy_body(x, y, E, L):
    return (E*y/L**2)                                           

def traction(x, y, nx_c, ny_c, E, L):
    sxx = sigma_xx(x, y, E, L)
    syy = sigma_yy(x, y, E, L)
    sxy = sigma_xy(x, y, E, L)
    return sxx*nx_c + sxy*ny_c,  sxy*nx_c + syy*ny_c



def get_nodes_2d(L, nx, ny):
    dx = L/(nx-1);  dy = L/(ny-1)
    return np.array([[i*dx, j*dy]
                     for j in range(ny) for i in range(nx)], dtype=float)

def get_triangles(nx, ny):
    tris = []
    for j in range(ny-1):
        for i in range(nx-1):
            k00=j*nx+i; k10=j*nx+(i+1); k01=(j+1)*nx+i; k11=(j+1)*nx+(i+1)
            tris += [[k00,k10,k11], [k00,k11,k01]]
    return np.array(tris)




def compute_nodal_forces(nodes_2d, L, E, nx, ny):
    dx  = L/(nx-1);  dy = L/(ny-1)
    F   = np.zeros((len(nodes_2d), 2))
    eps = 1e-10
 

    for k, (xk, yk) in enumerate(nodes_2d):
        i = round(xk/dx);  j = round(yk/dy)
        wx = dx/2 if (i==0 or i==nx-1) else dx
        wy = dy/2 if (j==0 or j==ny-1) else dy
        F[k,0] += fx_body(xk, yk, E, L) * wx * wy
        F[k,1] += fy_body(xk, yk, E, L) * wx * wy
 
    # Neumann y=0  (vec normal = (0,-1)T)
    #    Exclus :  x=0 ou x=L  (Dirichlet)
    for i in range(nx):
        k = i;  xk, yk = nodes_2d[k]
        if xk < eps or xk > L-eps:   
            continue
        wx = dx/2 if (i==0 or i==nx-1) else dx
        Tx, Ty = traction(xk, yk, 0., -1., E, L)
        F[k,0] += Tx*wx;  F[k,1] += Ty*wx
 
    # Neumann y=L 
    #    noeuds avec x=0 ou x=L  (Dirichlet)
    for i in range(nx):
        k = (ny-1)*nx + i;  xk, yk = nodes_2d[k]
        if xk < eps or xk > L-eps:  
            continue
        wx = dx/2 if (i==0 or i==nx-1) else dx
        Tx, Ty = traction(xk, yk, 0., +1., E, L)
        F[k,0] += Tx*wx;  F[k,1] += Ty*wx
 

 
    return F
 


def createScene(rootNode, L=1.0, E=1e6, nx=10, ny=10, with_visual=True):
    rootNode.addObject('RequiredPlugin',pluginName='Sofa.Component.Visual')
    rootNode.addObject('RequiredPlugin',  pluginName=[                       

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
    tris_np  = get_triangles(nx, ny)
    Beam     = rootNode.addChild('Beam')

    
    Beam.addObject('StaticSolver',
                   name="staticSolver",
                   printLog=False)
    Beam.addObject('NewtonRaphsonSolver',
                  name="newtonSolver",
                  maxNbIterationsNewton=1,
                  absoluteResidualStoppingThreshold=1e-10,
                  printLog=False)
    Beam.addObject('SparseLDLSolver',
                   name="linearSolver",
                   template="CompressedRowSparseMatrixd")

    # DOFs
    dofs = Beam.addObject('MechanicalObject',
                          name="dofs",
                          template="Vec2d",
                          position=nodes_2d.tolist(),
                          showObject=with_visual,
                          showObjectScale=0.005*L)

    # topo
    Beam.addObject('TriangleSetTopologyContainer',
                   name="topology",
                   triangles=tris_np.tolist())
    Beam.addObject('TriangleSetTopologyModifier')

    # FEM
    Beam.addObject('LinearSmallStrainFEMForceField',
                   name="FEM",
                   template="Vec2d",
                   youngModulus=E,
                   poissonRatio=0.0,
                   topology="@topology")

    # Dirichlet x=0 & x=L
    e = 1e-4 * L / max(nx-1, ny-1)

    Beam.addObject('BoxROI', name="box_left",  template="Vec2d",
                   box=[-e,-e,0, e, L+e,0], drawBoxes=with_visual)
    Beam.addObject('FixedProjectiveConstraint', name="fix_left",
                   template="Vec2d", indices="@box_left.indices")

    Beam.addObject('BoxROI', name="box_right", template="Vec2d",
                   box=[L-e,-e,0, L+e, L+e,0], drawBoxes=with_visual)
    Beam.addObject('FixedProjectiveConstraint', name="fix_right",
                   template="Vec2d", indices="@box_right.indices")

    # forces 
    F   = compute_nodal_forces(nodes_2d, L, E, nx, ny)
    idx = " ".join(str(k) for k in range(len(nodes_2d)))
    frc = " ".join(f"{F[k,0]} {F[k,1]}" for k in range(len(nodes_2d)))
    Beam.addObject('ConstantForceField',
                   name="MMS_forces",
                   template="Vec2d",
                   indices=idx,
                   forces=frc)

    return dofs, nodes_2d, tris_np


# sim 

def run_simulation(L, E, nx, ny):
    root = Sofa.Core.Node("root")
    dofs, nodes_2d, tris_np = createScene(root, L=L, E=E,
                                           nx=nx, ny=ny, with_visual=False)
    Sofa.Simulation.init(root)
    pos0 = dofs.position.array().copy()
    Sofa.Simulation.animate(root, root.dt.value)
    pos1 = dofs.position.array().copy()
    Sofa.Simulation.unload(root)
    return nodes_2d, tris_np, pos1[:,0]-pos0[:,0], pos1[:,1]-pos0[:,1]




def compute_l2(nodes_2d, ux, uy, L, nx, ny, tris_np):
    dx = L/(nx-1);  dy = L/(ny-1);  area = dx*dy/2.0;  err2 = 0.0
    for i0,i1,i2 in tris_np:
        xc = (nodes_2d[i0,0]+nodes_2d[i1,0]+nodes_2d[i2,0])/3
        yc = (nodes_2d[i0,1]+nodes_2d[i1,1]+nodes_2d[i2,1])/3
        ux_c = (ux[i0]+ux[i1]+ux[i2])/3
        uy_c = (uy[i0]+uy[i1]+uy[i2])/3
        err2 += ((ux_c-ux_mms(xc,yc,L))**2 + (uy_c-uy_mms(xc,yc,L))**2) * area
    return np.sqrt(err2)




def simulation_ponctuelle(L, E, nx, ny):
    nodes_2d, tris_np, ux, uy = run_simulation(L, E, nx, ny)
    l2 = compute_l2(nodes_2d, ux, uy, L, nx, ny, tris_np)

    ux_ref = ux_mms(nodes_2d[:,0], nodes_2d[:,1], L)
    uy_ref = uy_mms(nodes_2d[:,0], nodes_2d[:,1], L)

    print(f"\n[Vec2d]  nx={nx} ny={ny}  L={L}")
    print(f"  L2  = {l2:.4e}")
    print(f"  max|ux-ux_mms| = {np.max(np.abs(ux-ux_ref)):.4e}")
    print(f"  max|uy-uy_mms| = {np.max(np.abs(uy-uy_ref)):.4e}")

     
    mid_j  = (ny-1)//2
    sl     = slice(mid_j*nx, mid_j*nx+nx)
    yc     = nodes_2d[mid_j*nx, 1]
    xf     = np.linspace(0, L, 300)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, u_sofa, u_fn, lbl, fmt in zip(
            axes,
            [ux[sl], uy[sl]],
            [lambda x: ux_mms(x,yc,L), lambda x: uy_mms(x,yc,L)],
            [r'$u_x$', r'$u_y$'], ['bo-','gs-']):
        ax.plot(nodes_2d[sl,0], u_sofa, fmt, label='SOFA', ms=5)
        ax.plot(xf, u_fn(xf), 'r--', label='MMS exact')
        ax.set_xlabel('x');  ax.set_ylabel(lbl)
        ax.legend();  ax.grid(True, alpha=0.3)
    plt.suptitle(f'MMS 2D — Vec2d  nx={nx}  L2={l2:.2e}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'solution_nx{nx}.png'), dpi=150)
    plt.close()

    # Champs 2D
    x = nodes_2d[:,0];  y = nodes_2d[:,1];  tl = tris_np.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, data, title, cmap in [
        (axes[0,0], ux,              r'$u_x$ SOFA',         'RdBu_r'),
        (axes[0,1], ux_ref,          r'$u_x$ MMS',          'RdBu_r'),
        (axes[0,2], np.abs(ux-ux_ref), r'$|u_x - u_x^{MMS}|$','hot_r'),
        (axes[1,0], uy,              r'$u_y$ SOFA',         'RdBu_r'),
        (axes[1,1], uy_ref,          r'$u_y$ MMS',          'RdBu_r'),
        (axes[1,2], np.abs(uy-uy_ref), r'$|u_y - u_y^{MMS}|$','hot_r'),
    ]:
        tc = ax.tricontourf(x, y, tl, data, levels=20, cmap=cmap)
        ax.triplot(x, y, tl, 'k-', lw=0.3, alpha=0.4)
        plt.colorbar(tc, ax=ax, shrink=0.8)
        ax.set_title(title);  ax.set_aspect('equal')
        ax.set_xlabel('x');  ax.set_ylabel('y')
    plt.suptitle(f'Champs 2D — Vec2d  nx={nx}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'champs2D_nx{nx}.png'), dpi=150)
    plt.close()
    print(f"  → figures dans '{RESULTS_DIR}/'")

 

def convergence_study(L, E, nx_values):
    hs, errs = [], []
    hdr = f"{'nx':>5} | {'h':>8} | {'L2':>14} | {'ordre':>7}"
    print(f"\n── Convergence Vec2d ──\n{hdr}")

    with open(os.path.join(RESULTS_DIR,'convergence_Vec2d.txt'),'w') as f:
        f.write("Convergence MMS 2D — Vec2d  (L2 centroides)\n")
        f.write(hdr + "\n")

        for k, nx in enumerate(nx_values):
            ny = nx;  h = L/(nx-1)
            nodes_2d, tris_np, ux, uy = run_simulation(L, E, nx, ny)
            l2 = compute_l2(nodes_2d, ux, uy, L, nx, ny, tris_np)
            hs.append(h);  errs.append(l2)

            ordre = (f"{np.log(l2/errs[k-1])/np.log(h/hs[k-1]):.2f}"
                     if k > 0 else "")
            line = f"{nx:5d} | {h:8.4f} | {l2:14.6e} | {ordre:>7}"
            print(line);  f.write(line + "\n")

    hs_arr = np.array(hs);  errs_arr = np.array(errs)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(hs_arr, errs_arr, 'bo-', lw=2, ms=8, label='L2 (centroides)')
    h_ref = np.array([hs_arr[0], hs_arr[-1]])
    ax.loglog(h_ref, errs_arr[0]*(h_ref/hs_arr[0])**2, 'k--', lw=1.2, label='O(h^2)')
    ax.set_xlabel('h');  ax.set_ylabel('Erreur L2')
    ax.set_title('Convergence MMS 2D — Vec2d')
    ax.legend();  ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,'convergence_Vec2d.png'), dpi=150)
    plt.close()
    


 

if __name__ == "__main__":
    L, E = 1.0, 1e6

    simulation_ponctuelle(L, E, nx=10, ny=10)
    convergence_study(L, E, nx_values=[3, 4, 5, 6, 8, 10, 12, 16, 100])
