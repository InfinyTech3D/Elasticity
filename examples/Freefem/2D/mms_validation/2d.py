import numpy as np
import os
import Sofa, Sofa.Core, Sofa.Simulation
import matplotlib.pyplot as plt

RESULTS_DIR = "results_mms_2d"
os.makedirs(RESULTS_DIR, exist_ok=True)

E_MOD = 1e6
NU    = 0.0
A_AMP = 1e-4

C11 = E_MOD / (1.0 - NU**2)          
C12 = E_MOD * NU / (1.0 - NU**2)
C33 = E_MOD / (2 * (1 + NU))

def u_exact(x, y):
    ux =  A_AMP * np.sin(np.pi*x) * np.sin(np.pi*y)
    uy = -A_AMP * np.sin(np.pi*x) * np.sin(np.pi*y)
    return ux, uy
# gradients 
def grad_u_exact(x, y):
    c = A_AMP * np.pi
    duxdx =  c * np.cos(np.pi*x) * np.sin(np.pi*y)
    duxdy =  c * np.sin(np.pi*x) * np.cos(np.pi*y)
    duydx = -c * np.cos(np.pi*x) * np.sin(np.pi*y)
    duydy = -c * np.sin(np.pi*x) * np.cos(np.pi*y)
    return duxdx, duxdy, duydx, duydy

# source :  f = -div(sigma)
def body_force(x, y):
    c = A_AMP * np.pi**2
    
    fx =  c * ( (C11 + C33)*np.sin(np.pi*x)*np.sin(np.pi*y) 
              + (C12 + C33)*np.cos(np.pi*x)*np.cos(np.pi*y) )
    
    fy = -c * ( (C12 + C33)*np.cos(np.pi*x)*np.cos(np.pi*y) 
              + (C11 + C33)*np.sin(np.pi*x)*np.sin(np.pi*y) )
    return fx, fy

# maillage 
def create_mesh(L, H, nx, ny):
    x = np.linspace(0, L, nx)
    y = np.linspace(0, H, ny)
    X, Y = np.meshgrid(x, y)
    positions = np.column_stack([X.ravel(), Y.ravel()])
    tris = []
    for j in range(ny-1):
        for i in range(nx-1):
            n0,n1 = j*nx+i,   j*nx+i+1
            n2,n3 = (j+1)*nx+i, (j+1)*nx+i+1
            tris += [[n0,n1,n2],[n1,n3,n2]]
    return positions, np.array(tris, dtype=int)

_GP = np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
_GW = np.array([1/3, 1/3, 1/3])

def tri_jacobian(p0, p1, p2):
    J = np.array([p1-p0, p2-p0])   
    return J, 0.5*abs(np.linalg.det(J))

# Forces nodales : int phi_i * f d Omega 

def nodal_body_force(positions, triangles):
    N = len(positions)
    F = np.zeros((N, 2))
    for tri in triangles:
        p0,p1,p2 = positions[tri]
        J, area = tri_jacobian(p0,p1,p2)
        for gp, gw in zip(_GP, _GW):
            xy = gp[0]*p0 + gp[1]*p1 + gp[2]*p2
            fx, fy = body_force(xy[0], xy[1])
            for li, ni in enumerate(tri):
                F[ni,0] += gw * gp[li] * fx * area
                F[ni,1] += gw * gp[li] * fy * area
    return F

def _add_plugins(root):
    root.addObject('RequiredPlugin', pluginName=[
        "Elasticity",
        "Sofa.Component.Constraint.Projective",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.MechanicalLoad",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Dynamic",
        "Sofa.Component.Visual",
    ])

def build_scene(root, L, H, nx, ny):
    _add_plugins(root)
    root.addObject('DefaultAnimationLoop')

    node = root.addChild('Domain')
    positions, triangles = create_mesh(L, H, nx, ny)
    N = len(positions)

    eps = 1e-9
    x_all, y_all = positions[:,0], positions[:,1]
    bnd = np.where(
        (x_all < eps) | (x_all > L-eps) |
        (y_all < eps) | (y_all > H-eps)
    )[0]

    dofs = node.addObject('MechanicalObject', name="dofs",
                          template="Vec2d",
                          position=positions.tolist())

    node.addObject('TriangleSetTopologyContainer', name="topo",
                   triangles=triangles.tolist())
    node.addObject('TriangleSetTopologyModifier')
    node.addObject('TriangleSetGeometryAlgorithms', template="Vec2d")

    node.addObject('SparseLDLSolver', name="solver",
                   template="CompressedRowSparseMatrixd")

    node.addObject('NewtonRaphsonSolver', name="newton",
                   printLog=False,
                   maxNbIterationsNewton=20,
                   absoluteResidualStoppingThreshold=1e-10)
    node.addObject('StaticSolver', name="static",
                   newtonSolver="@newton",
                   linearSolver="@solver")

    node.addObject('LinearSmallStrainFEMForceField',
               name="FEM", template="Vec2d",
               youngModulus=E_MOD, 
               poissonRatio=NU,
               topology="@topo")

    node.addObject('FixedConstraint', name="bc",
                   indices=bnd.tolist(), template="Vec2d")

    # On charge uniquement F_body ; SOFA gère K·u lui-même
    F_body = nodal_body_force(positions, triangles)
    F_body[bnd] = 0.0 

    print(f"  Norme F_body = {np.linalg.norm(F_body):.3e}")

    node.addObject('ConstantForceField', name="src",
                   indices=list(range(N)),
                   forces=F_body.tolist(), template="Vec2d")

    Sofa.Simulation.init(root)
    return dofs, positions, triangles, bnd

#  Erreurs L2 et H1 
def compute_errors(positions, triangles, u_sofa):
    l2_sq = h1_sq = denom = 0.0
    dN_ref = np.array([[-1., -1.],
                        [ 1.,  0.],
                        [ 0.,  1.]], dtype=float)  

    for tri in triangles:
        p0, p1, p2 = positions[tri]

        J = np.array([p1 - p0, p2 - p0])
        area = 0.5 * abs(np.linalg.det(J))
        Jinv = np.linalg.inv(J)

        
        dN = dN_ref @ Jinv.T          
        u_tri = u_sofa[tri]           
        grad_uh = dN.T @ u_tri        

        for gp, gw in zip(_GP, _GW):
            xy = gp[0]*p0 + gp[1]*p1 + gp[2]*p2
            ux_ex, uy_ex = u_exact(xy[0], xy[1])
            uh = gp @ u_tri

            l2_sq += gw * ((uh[0] - ux_ex)**2 + (uh[1] - uy_ex)**2) * area
            denom  += gw * (ux_ex**2 + uy_ex**2) * area

            duxdx, duxdy, duydx, duydy = grad_u_exact(xy[0], xy[1])
            g_ex = np.array([[duxdx, duydx],
                              [duxdy, duydy]])
            h1_sq += gw * np.sum((grad_uh - g_ex)**2) * area

    return np.sqrt(l2_sq), np.sqrt(l2_sq / denom), np.sqrt(h1_sq)

# Convergence 
def convergence_study():
    L, H = 1.0, 1.0
    grids = [(4,4),(8,8),(16,16),(32,32),(64,64)]
    results = []

    for nx, ny in grids:
        h = L / (nx-1)
        root = Sofa.Core.Node("root")
        dofs, positions, triangles, bnd = build_scene(root, L, H, nx, ny)
        Sofa.Simulation.animate(root, 1.0)

        pos_final = np.array(dofs.position.value)
        u_sofa    = pos_final - positions

        l2, l2r, h1 = compute_errors(positions, triangles, u_sofa)
        print(f"  L2 = {l2:.3e}   L2_rel = {l2r:.3e}   H1 = {h1:.3e}")
        results.append((h, l2, h1))
        Sofa.Simulation.unload(root)

    return np.array(results)

def plot_convergence(results):
    h, l2, h1 = results[:,0], results[:,1], results[:,2]

    # Calcul des pentes
    sl2 = np.polyfit(np.log(h), np.log(l2), 1)[0]
    sh1 = np.polyfit(np.log(h), np.log(h1), 1)[0]
    print(f"\nPente L2 = {sl2:.2f}  (attendu ~ 2.0)")
    print(f"Pente H1 = {sh1:.2f}  (attendu ~ 1.0)")

    fig, ax = plt.subplots(figsize=(7,5))
    ax.loglog(h, l2, 'o-', label=f'‖e‖_L2  (pente={sl2:.2f})', color='royalblue', lw=2)
    ax.loglog(h, h1, 's-', label=f'|e|_H1  (pente={sh1:.2f})', color='firebrick', lw=2)
    ax.loglog(h, l2[0]*(h/h[0])**2, '--', color='royalblue', alpha=0.4, label='O(h²)')
    ax.loglog(h, h1[0]*(h/h[0])**1, '--', color='firebrick', alpha=0.4, label='O(h)')
    ax.set_xlabel('h')
    ax.set_ylabel('Error ')
    ax.set_title('Convergence MMS P1 ')
    ax.legend()
    ax.grid(True, which='both', ls=':')
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "convergence.png")
    plt.savefig(out, dpi=200)
    plt.close()

if __name__ == "__main__":
    results = convergence_study()
    if len(results) >= 2:
        plot_convergence(results)