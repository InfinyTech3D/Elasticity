import numpy as np
import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime
import matplotlib.pyplot as plt
import os


RESULTS_DIR = "results_3d_hexa_err-neu"
os.makedirs(RESULTS_DIR, exist_ok=True)




L_DEFAULT  = 1.0
E_DEFAULT  = 1e6
NU_DEFAULT = 0.3

NX, NY, NZ = 6, 6, 6


def lame(E,nu):
    if abs(nu)<1e-12:
        return 0.0, E /2.0
    
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu  = E / (2 * (1 + nu))
    return lam, mu 

# ======= MMS & derivatives ============================== 


def lame(E, nu):
    if abs(nu) < 1e-12:
        return 0.0, E / 2.0
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu  = E / (2 * (1 + nu))
    return lam, mu

def ux_mms(x, y, z, L): return x / L
def uy_mms(x, y, z, L): return y / L
def uz_mms(x, y, z, L): return z / L


def stress(E, nu, L):
    lam, mu = lame(E, nu)
    eps = np.diag([1/L, 1/L, 1/L])
    return lam * np.trace(eps) * np.eye(3) + 2 * mu * eps


def traction_3d(x, y, z, nx_c, ny_c, nz_c, E, nu, L):
    t = stress(E, nu, L) @ np.array([nx_c, ny_c, nz_c])
    return t[0], t[1], t[2]






 
# ================================ Mesh =============================================

def get_nodes_3d(L, nx, ny, nz):
    dx = L / (nx - 1)
    dy = L / (ny - 1)
    dz = L / (nz - 1)
    return np.array(
        [[i*dx, j*dy, k*dz]
         for k in range(nz)
         for j in range(ny)
         for i in range(nx)],
        dtype=float
    )


def idx(i, j, k, nx, ny):
    return k * ny * nx + j * nx + i


def get_hexahedra(nx, ny, nz):
    hexas = []
    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                hexas.append([
                    idx(i,   j,   k,   nx, ny),
                    idx(i+1, j,   k,   nx, ny),
                    idx(i+1, j+1, k,   nx, ny),
                    idx(i,   j+1, k,   nx, ny),
                    idx(i,   j,   k+1, nx, ny),
                    idx(i+1, j,   k+1, nx, ny),
                    idx(i+1, j+1, k+1, nx, ny),
                    idx(i,   j+1, k+1, nx, ny),
                ])
    return np.array(hexas)


def get_surface_quads(nx, ny, nz):
    quads = []
    for k in range(nz-1):
        for j in range(ny-1):
            quads.append([idx(0,j,k,nx,ny), idx(0,j,k+1,nx,ny),
                          idx(0,j+1,k+1,nx,ny), idx(0,j+1,k,nx,ny)])
    for k in range(nz-1):
        for j in range(ny-1):
            quads.append([idx(nx-1,j,k,nx,ny), idx(nx-1,j+1,k,nx,ny),
                          idx(nx-1,j+1,k+1,nx,ny), idx(nx-1,j,k+1,nx,ny)])
    for k in range(nz-1):
        for i in range(nx-1):
            quads.append([idx(i,0,k,nx,ny), idx(i+1,0,k,nx,ny),
                          idx(i+1,0,k+1,nx,ny), idx(i,0,k+1,nx,ny)])
    for k in range(nz-1):
        for i in range(nx-1):
            quads.append([idx(i,ny-1,k,nx,ny), idx(i,ny-1,k+1,nx,ny),
                          idx(i+1,ny-1,k+1,nx,ny), idx(i+1,ny-1,k,nx,ny)])
    for j in range(ny-1):
        for i in range(nx-1):
            quads.append([idx(i,j,0,nx,ny), idx(i+1,j,0,nx,ny),
                          idx(i+1,j+1,0,nx,ny), idx(i,j+1,0,nx,ny)])
    for j in range(ny-1):
        for i in range(nx-1):
            quads.append([idx(i,j,nz-1,nx,ny), idx(i,j+1,nz-1,nx,ny),
                          idx(i+1,j+1,nz-1,nx,ny), idx(i+1,j,nz-1,nx,ny)])
    return quads

# ========== forces nodales ===================

def compute_nodal_forces_3d(nodes_3d, L, E, nu, nx, ny, nz):
    dx = L / (nx - 1)
    dy = L / (ny - 1)
    dz = L / (nz - 1)
    F  = np.zeros((len(nodes_3d), 3))

    gp = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    gw = np.array([1.0, 1.0])

    Jf = (dy/2) * (dz/2)
    for k in range(nz - 1):
        for j in range(ny - 1):
            y0, y1 = j*dy, (j+1)*dy
            z0, z1 = k*dz, (k+1)*dz
            face   = [idx(nx-1,j,  k,  nx,ny), idx(nx-1,j+1,k,  nx,ny),
                      idx(nx-1,j+1,k+1,nx,ny), idx(nx-1,j,  k+1,nx,ny)]
            eta_f  = [-1, 1, 1,-1]
            zeta_f = [-1,-1, 1, 1]
            for gi, p1 in enumerate(gp):
                for gj, p2 in enumerate(gp):
                    w  = gw[gi]*gw[gj]*Jf
                    yg = (y0+y1)/2 + p1*(dy/2)
                    zg = (z0+z1)/2 + p2*(dz/2)
                    Tx,Ty,Tz = traction_3d(L, yg, zg, 1.,0.,0., E, nu, L)
                    for a in range(4):
                        Na = (1+eta_f[a]*p1)*(1+zeta_f[a]*p2)/4.
                        F[face[a],0] += Na*Tx*w
                        F[face[a],1] += Na*Ty*w
                        F[face[a],2] += Na*Tz*w

    for k in range(nz - 1):
        for j in range(ny - 1):
            y0, y1 = j*dy, (j+1)*dy
            z0, z1 = k*dz, (k+1)*dz
            face   = [idx(0,j,  k,  nx,ny), idx(0,j+1,k,  nx,ny),
                      idx(0,j+1,k+1,nx,ny), idx(0,j,  k+1,nx,ny)]
            eta_f  = [-1, 1, 1,-1]
            zeta_f = [-1,-1, 1, 1]
            for gi, p1 in enumerate(gp):
                for gj, p2 in enumerate(gp):
                    w  = gw[gi]*gw[gj]*Jf
                    yg = (y0+y1)/2 + p1*(dy/2)
                    zg = (z0+z1)/2 + p2*(dz/2)
                    Tx,Ty,Tz = traction_3d(0., yg, zg, -1.,0.,0., E, nu, L)
                    for a in range(4):
                        Na = (1+eta_f[a]*p1)*(1+zeta_f[a]*p2)/4.
                        F[face[a],0] += Na*Tx*w
                        F[face[a],1] += Na*Ty*w
                        F[face[a],2] += Na*Tz*w

    
    Jf = (dx/2) * (dz/2)
    for k in range(nz - 1):
        for i in range(nx - 1):
            x0, x1 = i*dx, (i+1)*dx
            z0, z1 = k*dz, (k+1)*dz
            face   = [idx(i,  ny-1,k,  nx,ny), idx(i+1,ny-1,k,  nx,ny),
                      idx(i+1,ny-1,k+1,nx,ny), idx(i,  ny-1,k+1,nx,ny)]
            xi_f   = [-1, 1, 1,-1]
            zeta_f = [-1,-1, 1, 1]
            for gi, p1 in enumerate(gp):
                for gj, p2 in enumerate(gp):
                    w  = gw[gi]*gw[gj]*Jf
                    xg = (x0+x1)/2 + p1*(dx/2)
                    zg = (z0+z1)/2 + p2*(dz/2)
                    Tx,Ty,Tz = traction_3d(xg, L, zg, 0.,1.,0., E, nu, L)
                    for a in range(4):
                        Na = (1+xi_f[a]*p1)*(1+zeta_f[a]*p2)/4.
                        F[face[a],0] += Na*Tx*w
                        F[face[a],1] += Na*Ty*w
                        F[face[a],2] += Na*Tz*w

    for k in range(nz - 1):
        for i in range(nx - 1):
            x0, x1 = i*dx, (i+1)*dx
            z0, z1 = k*dz, (k+1)*dz
            face   = [idx(i,  0,k,  nx,ny), idx(i+1,0,k,  nx,ny),
                      idx(i+1,0,k+1,nx,ny), idx(i,  0,k+1,nx,ny)]
            for gi, p1 in enumerate(gp):
                for gj, p2 in enumerate(gp):
                    w  = gw[gi]*gw[gj]*Jf
                    xg = (x0+x1)/2 + p1*(dx/2)
                    zg = (z0+z1)/2 + p2*(dz/2)
                    Tx,Ty,Tz = traction_3d(xg, 0., zg, 0.,-1.,0., E, nu, L)
                    for a in range(4):
                        Na = (1+xi_f[a]*p1)*(1+zeta_f[a]*p2)/4.
                        F[face[a],0] += Na*Tx*w
                        F[face[a],1] += Na*Ty*w
                        F[face[a],2] += Na*Tz*w

    Jf = (dx/2) * (dy/2)
    for j in range(ny - 1):
        for i in range(nx - 1):
            x0, x1 = i*dx, (i+1)*dx
            y0, y1 = j*dy, (j+1)*dy
            face   = [idx(i,  j,  nz-1,nx,ny), idx(i+1,j,  nz-1,nx,ny),
                      idx(i+1,j+1,nz-1,nx,ny), idx(i,  j+1,nz-1,nx,ny)]
            xi_f2  = [-1, 1, 1,-1]
            eta_f2 = [-1,-1, 1, 1]
            for gi, p1 in enumerate(gp):
                for gj, p2 in enumerate(gp):
                    w  = gw[gi]*gw[gj]*Jf
                    xg = (x0+x1)/2 + p1*(dx/2)
                    yg = (y0+y1)/2 + p2*(dy/2)
                    Tx,Ty,Tz = traction_3d(xg, yg, L, 0.,0.,1., E, nu, L)
                    for a in range(4):
                        Na = (1+xi_f2[a]*p1)*(1+eta_f2[a]*p2)/4.
                        F[face[a],0] += Na*Tx*w
                        F[face[a],1] += Na*Ty*w
                        F[face[a],2] += Na*Tz*w

    
    for j in range(ny - 1):
        for i in range(nx - 1):
            x0, x1 = i*dx, (i+1)*dx
            y0, y1 = j*dy, (j+1)*dy
            face   = [idx(i,  j,  0,nx,ny), idx(i+1,j,  0,nx,ny),
                      idx(i+1,j+1,0,nx,ny), idx(i,  j+1,0,nx,ny)]
            for gi, p1 in enumerate(gp):
                for gj, p2 in enumerate(gp):
                    w  = gw[gi]*gw[gj]*Jf
                    xg = (x0+x1)/2 + p1*(dx/2)
                    yg = (y0+y1)/2 + p2*(dy/2)
                    Tx,Ty,Tz = traction_3d(xg, yg, 0., 0.,0.,-1., E, nu, L)
                    for a in range(4):
                        Na = (1+xi_f2[a]*p1)*(1+eta_f2[a]*p2)/4.
                        F[face[a],0] += Na*Tx*w
                        F[face[a],1] += Na*Ty*w
                        F[face[a],2] += Na*Tz*w

    # Vérification équilibre global
    res = np.abs(F.sum(axis=0))
    assert res.max() < 1e-6 * np.abs(F).max()

    return F

# =================== SOFA SCENE ========================================
# NEumann pur 


def create_scene_3d_neumann(rootNode, L=1.0, E=1e6, nu=0.3, nx=6, ny=6, nz=6, visual=False):
    plugins = [
        "Elasticity",
        "Sofa.Component.Constraint.Projective",
        "Sofa.Component.Engine.Select",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.MechanicalLoad",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Dynamic",
    ]
    if visual:
        plugins += ["Sofa.Component.Visual", "Sofa.GL.Component.Rendering3D"]

    rootNode.addObject('RequiredPlugin', pluginName=plugins)
    rootNode.addObject('DefaultAnimationLoop')
    if visual:
        rootNode.addObject('DefaultVisualManagerLoop')
        rootNode.addObject('VisualStyle', displayFlags='showVisualModels showWireframe')
    rootNode.gravity.value = [0, 0, 0]
    rootNode.dt.value = 1.0

    nodes_3d = get_nodes_3d(L, nx, ny, nz)
    hexas    = get_hexahedra(nx, ny, nz)

    solid = rootNode.addChild('Solid3D')

    solid.addObject('NewtonRaphsonSolver',
                    name="newtonSolver",
                    printLog=False,
                    warnWhenLineSearchFails=True,
                    maxNbIterationsNewton=1,
                    maxNbIterationsLineSearch=1,
                    lineSearchCoefficient=1,
                    relativeSuccessiveStoppingThreshold=0,
                    absoluteResidualStoppingThreshold=1e-7,
                    absoluteEstimateDifferenceThreshold=1e-12,
                    relativeInitialStoppingThreshold=1e-12,
                    relativeEstimateDifferenceThreshold=0)
    solid.addObject('SparseLDLSolver',
                    name="linearSolver",
                    template="CompressedRowSparseMatrixd")
    solid.addObject('StaticSolver',
                    name="staticSolver",
                    newtonSolver="@newtonSolver",
                    linearSolver="@linearSolver")

    dofs = solid.addObject('MechanicalObject', name="dofs", template="Vec3d",
                           position=nodes_3d.tolist())

    solid.addObject('HexahedronSetTopologyContainer', name="topology",
                    hexahedra=hexas.tolist())
    solid.addObject('HexahedronSetTopologyModifier')

    solid.addObject('LinearSmallStrainFEMForceField', name="FEM", template="Vec3d",
                    youngModulus=E, poissonRatio=nu, topology="@topology")

    eps = 1e-8

    
    solid.addObject('BoxROI', name="fixA",
                    box=[-eps, -eps, -eps, eps, eps, eps])
    solid.addObject('FixedProjectiveConstraint', name="bcA",
                    indices="@fixA.indices")

    
    solid.addObject('BoxROI', name="fixB",
                    box=[L-eps, -eps, -eps, L+eps, eps, eps])
    solid.addObject('PartialFixedProjectiveConstraint', name="bcB",
                    indices="@fixB.indices",
                    fixedDirections=[0, 1, 1])  

    
    solid.addObject('BoxROI', name="fixC",
                    box=[-eps, L-eps, -eps, eps, L+eps, eps])
    solid.addObject('PartialFixedProjectiveConstraint', name="bcC",
                    indices="@fixC.indices",
                    fixedDirections=[0, 0, 1])   
    
    F = compute_nodal_forces_3d(nodes_3d, L, E, nu, nx, ny, nz)

    solid.addObject('ConstantForceField', name="MMS_forces", template="Vec3d",
                    indices=list(range(len(nodes_3d))),
                    forces=F.tolist())

    if visual:
        quads = get_surface_quads(nx, ny, nz)
        visu  = solid.addChild('Visual')
        visu.addObject('OglModel', name="ogl",
                       position=nodes_3d.tolist(),
                       quads=quads,
                       color=[0.2, 0.6, 1.0, 0.9])
        visu.addObject('IdentityMapping')

    return dofs, nodes_3d, hexas




def createScene(rootNode):
    return create_scene_3d_neumann(
        rootNode, L=L_DEFAULT, E=E_DEFAULT, nu=NU_DEFAULT,
        nx=NX, ny=NY, nz=NZ,
        visual=True,
    )

 
# =============================== SIMU ==============================================

def run_simulation_3d(L, E, nu, nx, ny, nz):
    root = Sofa.Core.Node("root")
    root.dt.value = 1.0

    dofs, nodes_3d, hexas = create_scene_3d_neumann(root, L=L, E=E, nu=nu,
                                             nx=nx, ny=ny, nz=nz)
    Sofa.Simulation.init(root)
    pos_init = dofs.position.array().copy()

    Sofa.Simulation.animate(root, root.dt.value)

    pos_final = dofs.position.array().copy()
    Sofa.Simulation.unload(root)

    ux = pos_final[:, 0] - pos_init[:, 0]
    uy = pos_final[:, 1] - pos_init[:, 1]
    uz = pos_final[:, 2] - pos_init[:, 2]
    return nodes_3d, hexas, ux, uy, uz




_XI_LOC   = np.array([-1, 1, 1,-1,-1, 1, 1,-1], dtype=float)
_ETA_LOC  = np.array([-1,-1, 1, 1,-1,-1, 1, 1], dtype=float)
_ZETA_LOC = np.array([-1,-1,-1,-1, 1, 1, 1, 1], dtype=float)

_GP = np.array([-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)])
_GW = np.array([1.0, 1.0])

# =========================== derivees parametriques fonctions de formes ================================
def _shape_hexa(xi, eta, zeta):
    
    N        = (1 + _XI_LOC*xi) * (1 + _ETA_LOC*eta) * (1 + _ZETA_LOC*zeta) / 8.0
    dN_dxi   = _XI_LOC   * (1 + _ETA_LOC*eta)  * (1 + _ZETA_LOC*zeta) / 8.0
    dN_deta  = _ETA_LOC  * (1 + _XI_LOC*xi)   * (1 + _ZETA_LOC*zeta) / 8.0
    dN_dzeta = _ZETA_LOC * (1 + _XI_LOC*xi)   * (1 + _ETA_LOC*eta)  / 8.0
    return N, dN_dxi, dN_deta, dN_dzeta



#  ==================================  ERROR L2 ======================================


def compute_l2_3d(nodes_3d, ux, uy, uz, L, hexas):
   
    err2 = 0.0
    for hexa in hexas:
        xe   = nodes_3d[hexa, 0]
        ye   = nodes_3d[hexa, 1]
        ze   = nodes_3d[hexa, 2]
        ux_e = ux[hexa]
        uy_e = uy[hexa]
        uz_e = uz[hexa]

        for xi, wi in zip(_GP, _GW):
            for eta, wj in zip(_GP, _GW):
                for zeta, wk in zip(_GP, _GW):
                    N, dN_dxi, dN_deta, dN_dzeta = _shape_hexa(xi, eta, zeta)

                    J = np.array([
                        [dN_dxi @ xe,   dN_dxi @ ye,   dN_dxi @ ze],
                        [dN_deta @ xe,  dN_deta @ ye,  dN_deta @ ze],
                        [dN_dzeta @ xe, dN_dzeta @ ye, dN_dzeta @ ze],
                    ])
                    detJ = np.linalg.det(J)

                    xg = N @ xe;  yg = N @ ye;  zg = N @ ze

                    ux_h = N @ ux_e;  uy_h = N @ uy_e;  uz_h = N @ uz_e

                    err2 += (
                        (ux_h - xg/L)**2
                      + (uy_h - yg/L)**2
                      + (uz_h - zg/L)**2
                    ) * wi * wj * wk * detJ

    return np.sqrt(err2)

# =============================  semi norm H1 in 3D =====================================

def compute_h1_3d(nodes_3d, ux, uy, uz, L, hexas):
    
    grad_ex = np.eye(3) / L   

    err2 = 0.0
    for hexa in hexas:
        xe   = nodes_3d[hexa, 0]
        ye   = nodes_3d[hexa, 1]
        ze   = nodes_3d[hexa, 2]
        ux_e = ux[hexa]
        uy_e = uy[hexa]
        uz_e = uz[hexa]

        for xi, wi in zip(_GP, _GW):
            for eta, wj in zip(_GP, _GW):
                for zeta, wk in zip(_GP, _GW):
                    N, dN_dxi, dN_deta, dN_dzeta = _shape_hexa(xi, eta, zeta)

                    J = np.array([
                        [dN_dxi @ xe,   dN_dxi @ ye,   dN_dxi @ ze],
                        [dN_deta @ xe,  dN_deta @ ye,  dN_deta @ ze],
                        [dN_dzeta @ xe, dN_dzeta @ ye, dN_dzeta @ ze],
                    ])
                    detJ = np.linalg.det(J)
                    Jinv = np.linalg.inv(J)

                
                    dN_param = np.vstack([dN_dxi, dN_deta, dN_dzeta])  
                    dN_phys  = Jinv.T @ dN_param                        
                    

                    
                    grad_h = np.array([
                        [dN_phys[0] @ ux_e, dN_phys[1] @ ux_e, dN_phys[2] @ ux_e],
                        [dN_phys[0] @ uy_e, dN_phys[1] @ uy_e, dN_phys[2] @ uy_e],
                        [dN_phys[0] @ uz_e, dN_phys[1] @ uz_e, dN_phys[2] @ uz_e],
                    ])

                    diff = grad_h - grad_ex
                    err2 += np.sum(diff**2) * wi * wj * wk * detJ

    return np.sqrt(err2)




def convergence_study_3d(L, E, nu, nx_values, results_dir=RESULTS_DIR):
    
    os.makedirs(results_dir, exist_ok=True)

    hs, errs_l2, errs_h1 = [], [], []

    hdr = (f"{'nx':>5} | {'h':>8} | {'L2':>14} | {'ord_L2':>7}"
           f" | {'H1':>14} | {'ord_H1':>7}")
    txt_path = os.path.join(results_dir, f"convergence_3d_hexa_nu{nu}.txt")

    print(f"\n── Convergence MMS 3D : Hexa Q1  nu={nu} ──\n{hdr}")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Convergence MMS 3D — Hexa Q1  E={E}  nu={nu}\n{hdr}\n")

        for k, nx in enumerate(nx_values):
            ny = nz = nx
            h = L / (nx - 1)

            nodes_3d, hexas, ux, uy, uz = run_simulation_3d(L, E, nu, nx, ny, nz)

            l2 = compute_l2_3d(nodes_3d, ux, uy, uz, L, hexas)
            h1 = compute_h1_3d(nodes_3d, ux, uy, uz, L, hexas)

            hs.append(h)
            errs_l2.append(l2)
            errs_h1.append(h1)

            ord_l2 = (f"{np.log(l2 / errs_l2[k-1]) / np.log(h / hs[k-1]):.2f}"
                      if k > 0 else "   —  ")
            ord_h1 = (f"{np.log(h1 / errs_h1[k-1]) / np.log(h / hs[k-1]):.2f}"
                      if k > 0 else "   —  ")

            line = (f"{nx:5d} | {h:8.4f} | {l2:14.6e} | {ord_l2:>7}"
                    f" | {h1:14.6e} | {ord_h1:>7}")
            print(line)
            f.write(line + "\n")

# log-log plot 
    hs_a  = np.array(hs)
    l2_a  = np.array(errs_l2)
    h1_a  = np.array(errs_h1)
    h_ref = np.array([hs_a[0], hs_a[-1]])

    for errs, ylabel, slope, fname in [
        (l2_a, "L2 error",      2, f"convergence_L2_3d_nu{nu}.png"),
        (h1_a, " H1 semi norm",  1, f"convergence_H1_3d_nu{nu}.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog(hs_a, errs, "o-", lw=2, ms=7,
                  color="#1a73e8", label=f"Hexa Q1 3D  nu={nu}")
        ax.loglog(h_ref, errs[0] * (h_ref / hs_a[0])**slope,
                  ":", color="gray", lw=1.5, label=f"O(h^{slope})")
        ax.set_xlabel("h")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Convergence MMS 3D — Hexa Q1\n{ylabel}  (nu={nu})")
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, fname), dpi=150)
        plt.close(fig)

    return hs_a, l2_a, h1_a




def plot_displacement(nodes_3d, ux, uy, uz, L, nx, ny, nz, E, nu):
    mid_i = (nx - 1) // 2
    mid_j = (ny - 1) // 2
    mid_k = (nz - 1) // 2

    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    line_x = [idx(i, mid_j, mid_k, nx, ny) for i in range(nx)]
    xv = nodes_3d[line_x, 0]
    axes[0].plot(xv, ux[line_x], 'bo-', label='SOFA', markersize=7)
    axes[0].plot(np.linspace(0, L, 200), np.linspace(0, L, 200)/L, 'r--',
                 label='MMS exact', lw=2)
    axes[0].set(xlabel='x', ylabel='$u_x$',
                title=f'$u_x$(x)  y={nodes_3d[line_x[0],1]:.2f}, z={nodes_3d[line_x[0],2]:.2f}')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    line_y = [idx(mid_i, j, mid_k, nx, ny) for j in range(ny)]
    yv = nodes_3d[line_y, 1]
    axes[1].plot(yv, uy[line_y], 'go-', label='SOFA', markersize=7)
    axes[1].plot(np.linspace(0, L, 200), np.linspace(0, L, 200)/L, 'r--',
                 label='MMS exact', lw=2)
    axes[1].set(xlabel='y', ylabel='$u_y$',
                title=f'$u_y$(y)  x={nodes_3d[line_y[0],0]:.2f}, z={nodes_3d[line_y[0],2]:.2f}')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    line_z = [idx(mid_i, mid_j, k, nx, ny) for k in range(nz)]
    zv = nodes_3d[line_z, 2]
    axes[2].plot(zv, uz[line_z], 'ms-', label='SOFA', markersize=7)
    axes[2].plot(np.linspace(0, L, 200), np.linspace(0, L, 200)/L, 'r--',
                 label='MMS exact', lw=2)
    axes[2].set(xlabel='z', ylabel='$u_z$',
                title=f'$u_z$(z)  x={nodes_3d[line_z[0],0]:.2f}, y={nodes_3d[line_z[0],1]:.2f}')
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.suptitle(f'MMS 3D Hexa lin — coupes 1D  ({nx}*{ny}*{nz})', fontsize=13)
    plt.tight_layout()
    out1 = os.path.join(RESULTS_DIR, f'1d_nx{nx}.png')
    plt.savefig(out1, dpi=150); plt.close()

    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))

    slice_z = np.array([[idx(i, j, mid_k, nx, ny) for i in range(nx)]
                        for j in range(ny)])
    X2d = nodes_3d[slice_z, 0]
    Y2d = nodes_3d[slice_z, 1]

    im = axes2[0, 0].pcolormesh(X2d, Y2d, ux[slice_z], cmap='RdBu_r', shading='auto')
    axes2[0, 0].set(xlabel='x', ylabel='y',
                    title=f'$u_x$  plan z=mid ({nodes_3d[slice_z[0,0],2]:.2f})')
    plt.colorbar(im, ax=axes2[0, 0])

    im = axes2[1, 0].pcolormesh(X2d, Y2d, uy[slice_z], cmap='RdBu_r', shading='auto')
    axes2[1, 0].set(xlabel='x', ylabel='y', title='$u_y$  plan z=mid')
    plt.colorbar(im, ax=axes2[1, 0])

    slice_y = np.array([[idx(i, mid_j, k, nx, ny) for i in range(nx)]
                        for k in range(nz)])
    X2d = nodes_3d[slice_y, 0]
    Z2d = nodes_3d[slice_y, 2]

    im = axes2[0, 1].pcolormesh(X2d, Z2d, ux[slice_y], cmap='RdBu_r', shading='auto')
    axes2[0, 1].set(xlabel='x', ylabel='z',
                    title=f'$u_x$  plan y=mid ({nodes_3d[slice_y[0,0],1]:.2f})')
    plt.colorbar(im, ax=axes2[0, 1])

    im = axes2[1, 1].pcolormesh(X2d, Z2d, uz[slice_y], cmap='RdBu_r', shading='auto')
    axes2[1, 1].set(xlabel='x', ylabel='z', title='$u_z$  plan y=mid')
    plt.colorbar(im, ax=axes2[1, 1])

    slice_x = np.array([[idx(mid_i, j, k, nx, ny) for j in range(ny)]
                        for k in range(nz)])
    Y2d = nodes_3d[slice_x, 1]
    Z2d = nodes_3d[slice_x, 2]

    im = axes2[0, 2].pcolormesh(Y2d, Z2d, uy[slice_x], cmap='RdBu_r', shading='auto')
    axes2[0, 2].set(xlabel='y', ylabel='z',
                    title=f'$u_y$  plan x=mid ({nodes_3d[slice_x[0,0],0]:.2f})')
    plt.colorbar(im, ax=axes2[0, 2])

    im = axes2[1, 2].pcolormesh(Y2d, Z2d, uz[slice_x], cmap='RdBu_r', shading='auto')
    axes2[1, 2].set(xlabel='y', ylabel='z', title='$u_z$  plan x=mid')
    plt.colorbar(im, ax=axes2[1, 2])

    plt.suptitle(f'MMS 3D Hexa lin; Displacment  ({nx}*{ny}*{nz})', fontsize=13)
    plt.tight_layout()
    out2 = os.path.join(RESULTS_DIR, f'2d_nx{nx}.png')
    plt.savefig(out2, dpi=150); plt.close()




if __name__ == "__main__":
    L  = L_DEFAULT
    E  = E_DEFAULT
    nu = NU_DEFAULT

    nodes_3d, hexas, ux, uy, uz = run_simulation_3d(L, E, nu, NX, NY, NZ)
    plot_displacement(nodes_3d, ux, uy, uz, L, NX, NY, NZ, E, nu)

    
    l2 = compute_l2_3d(nodes_3d, ux, uy, uz, L, hexas)
    h1 = compute_h1_3d(nodes_3d, ux, uy, uz, L, hexas)
    
    print(f"  L2 error       = {l2:.4e}")
    print(f"  H1 semi error  = {h1:.4e}")

    
    convergence_study_3d(
        L          = L,
        E          = E,
        nu         = nu,
        nx_values  = [3, 4, 6, 8, 11, 20],
        results_dir= RESULTS_DIR,
    )

nodes_3d, hexas, ux, uy, uz = run_simulation_3d(L, E, nu, NX, NY, NZ)

err_x = np.abs(ux - nodes_3d[:, 0] / L)
err_y = np.abs(uy - nodes_3d[:, 1] / L)
err_z = np.abs(uz - nodes_3d[:, 2] / L)

print(f"max|ux - x/L| = {err_x.max():.4e}")
print(f"max|uy - y/L| = {err_y.max():.4e}")
print(f"max|uz - z/L| = {err_z.max():.4e}")