import numpy as np
import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime
import matplotlib.pyplot as plt
import os


RESULTS_DIR = "results_3d_hexa_sin"
os.makedirs(RESULTS_DIR, exist_ok=True)

L_DEFAULT  = 1.0
E_DEFAULT  = 1e6
NU_DEFAULT = 0.3

NX, NY, NZ = 6, 6, 6


def lame(E, nu):
    if abs(nu) < 1e-12:
        return 0.0, E / 2.0
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu  = E / (2 * (1 + nu))
    return lam, mu

# =============== mms & derivatives ========================= 

def ux_mms(x, y, z, L):
    return np.sin(np.pi*x/L) * np.sin(np.pi*y/L)

def uy_mms(x, y, z, L):
    return np.sin(np.pi*y/L) * np.sin(np.pi*z/L)

def uz_mms(x, y, z, L):
    return np.sin(np.pi*z/L) * np.sin(np.pi*x/L)




def dux_dx(x, y, z, L): return  (np.pi/L) * np.cos(np.pi*x/L) * np.sin(np.pi*y/L)
def dux_dy(x, y, z, L): return  (np.pi/L) * np.sin(np.pi*x/L) * np.cos(np.pi*y/L)
def dux_dz(x, y, z, L): return  np.zeros_like(x) if hasattr(x, '__len__') else 0.0

def duy_dx(x, y, z, L): return  np.zeros_like(x) if hasattr(x, '__len__') else 0.0
def duy_dy(x, y, z, L): return  (np.pi/L) * np.cos(np.pi*y/L) * np.sin(np.pi*z/L)
def duy_dz(x, y, z, L): return  (np.pi/L) * np.sin(np.pi*y/L) * np.cos(np.pi*z/L)

def duz_dx(x, y, z, L): return  (np.pi/L) * np.sin(np.pi*z/L) * np.cos(np.pi*x/L)
def duz_dy(x, y, z, L): return  np.zeros_like(x) if hasattr(x, '__len__') else 0.0
def duz_dz(x, y, z, L): return  (np.pi/L) * np.cos(np.pi*z/L) * np.sin(np.pi*x/L)

 

def _C1(E, nu): return E / (2.0 * (nu + 1.0))

def _C2(E, nu): return E / ((1.0 + nu) * (1.0 - 2.0*nu))


def traction_3d(x, y, z, face, E, nu, L):
     
    p = np.pi / L
    c1 = _C1(E, nu)   
    c2 = _C2(E,nu)
    
    if face == 'xp':   
        exx   =  p * np.cos(p*L) * np.sin(p*y)      
        eyy   =  p * np.cos(p*y) * np.sin(p*z)
        ezz   =  p * np.cos(p*z) * np.sin(p*L)
        tr    = exx + eyy + ezz
        Tx = nu*c2*tr + 2*c1*exx
        Ty = c1 * p * np.sin(p*L) * np.cos(p*y)     
        Tz = c1 * p * np.cos(p*L) * np.sin(p*z)     
        Tz = c1 * p * np.cos(p*L) * np.sin(p*z)
        return Tx, Ty, Tz

    if face == 'xm':   
        exx   =  p * np.cos(0.0) * np.sin(p*y)      
        eyy   =  p * np.cos(p*y) * np.sin(p*z)
        ezz   =  p * np.cos(p*z) * np.sin(0.0)      
        tr    = exx + eyy + ezz
        
        Tx = -(nu*c2*tr + 2*c1*exx)
        Ty = -(c1 * p * np.sin(0.0) * np.cos(p*y))  
        Tz = -(c1 * p * np.cos(0.0) * np.sin(p*z))  
        return Tx, Ty, Tz

    if face == 'yp':   
        exx   =  p * np.cos(p*x) * np.sin(p*L)
        eyy   =  p * np.cos(p*L) * np.sin(p*z)
        ezz   =  p * np.cos(p*z) * np.sin(p*x)
        tr    = exx + eyy + ezz
        Tx = c1 * p * np.sin(p*x) * np.cos(p*L)     
        Ty = nu*c2*tr + 2*c1*eyy
        Tz = c1 * p * np.cos(p*z) * np.sin(p*x) * 0.0  
        Tz = c1 * p * np.sin(p*L) * np.cos(p*z)    
        return Tx, Ty, Tz

    if face == 'ym':   
        exx   =  p * np.cos(p*x) * np.sin(0.0)      
        eyy   =  p * np.cos(0.0) * np.sin(p*z)
        ezz   =  p * np.cos(p*z) * np.sin(p*x)
        tr    = exx + eyy + ezz
        Tx = -(c1 * p * np.sin(p*x) * np.cos(0.0))  
        Ty = -(nu*c2*tr + 2*c1*eyy)
        Tz = -(c1 * p * np.sin(0.0) * np.cos(p*z)) 
        return Tx, Ty, Tz

    if face == 'zp':   
        exx   =  p * np.cos(p*x) * np.sin(p*y)
        eyy   =  p * np.cos(p*y) * np.sin(p*L)
        ezz   =  p * np.cos(p*L) * np.sin(p*x)
        tr    = exx + eyy + ezz
        
        Tx = c1 * p * np.sin(p*L) * np.cos(p*x)     
        Ty = c1 * p * np.sin(p*y) * np.cos(p*L)     
        Tz = nu*c2*tr + 2*c1*ezz
        return Tx, Ty, Tz

    if face == 'zm':   
        exx   =  p * np.cos(p*x) * np.sin(p*y)
        eyy   =  p * np.cos(p*y) * np.sin(0.0)      
        ezz   =  p * np.cos(0.0) * np.sin(p*x)
        tr    = exx + eyy + ezz
        Tx = -(c1 * p * np.sin(0.0) * np.cos(p*x))  
        Ty = -(c1 * p * np.sin(p*y) * np.cos(0.0))  
        Tz = -(nu*c2*tr + 2*c1*ezz)
        return Tx, Ty, Tz

    raise ValueError(f"Face inconnue : {face}")




def body_force_mms(x, y, z, E, nu, L):
   
    p  = np.pi / L
    denom = 2.0 * L**2 * (2.0*nu**2 + nu - 1.0)

    fx = (np.pi**2 * E / denom) * (
          (4.0*nu - 3.0) * np.sin(p*x) * np.sin(p*y)
        + np.cos(p*x) * np.cos(p*z)
    )
    fy = (np.pi**2 * E / denom) * (
          (4.0*nu - 3.0) * np.sin(p*y) * np.sin(p*z)
        + np.cos(p*x) * np.cos(p*y)
    )
    fz = (np.pi**2 * E / denom) * (
          (4.0*nu - 3.0) * np.sin(p*x) * np.sin(p*z)
        + np.cos(p*y) * np.cos(p*z)
    )
    return fx, fy, fz


# =============================== Mesh ============================================== 

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

 

def compute_nodal_forces_3d(nodes_3d, L, E, nu, nx, ny, nz):
    dx = L / (nx - 1)
    dy = L / (ny - 1)
    dz = L / (nz - 1)
    F  = np.zeros((len(nodes_3d), 3))

    gp = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    gw = np.array([1.0, 1.0])


    def integrate_face(face_nodes, coords_a, coords_b, a0, a1, b0, b1, Ja, Jb, face_tag):
        
        xi_nodes  = np.array([-1, 1, 1,-1], dtype=float)
        eta_nodes = np.array([-1,-1, 1, 1], dtype=float)
        Jsurf = Ja * Jb
        for gi, p1 in enumerate(gp):
            for gj, p2 in enumerate(gp):
                w   = gw[gi] * gw[gj] * Jsurf
                ag  = (a0+a1)/2 + p1*Ja
                bg  = (b0+b1)/2 + p2*Jb
                # coordonnées physiques du point de Gauss
                xg, yg, zg = coords_a(ag, bg), coords_b(ag, bg), 0.0
                xg, yg, zg = _face_coords(face_tag, ag, bg, L)
                Tx, Ty, Tz = traction_3d(xg, yg, zg, face_tag, E, nu, L)
                for a in range(4):
                    Na = (1 + xi_nodes[a]*p1) * (1 + eta_nodes[a]*p2) / 4.0
                    F[face_nodes[a], 0] += Na * Tx * w
                    F[face_nodes[a], 1] += Na * Ty * w
                    F[face_nodes[a], 2] += Na * Tz * w

    def _face_coords(tag, a, b, L):
        if tag == 'xm': return 0., a,  b
        if tag == 'xp': return L,  a,  b
        if tag == 'ym': return a,  0., b
        if tag == 'yp': return a,  L,  b
        if tag == 'zm': return a,  b,  0.
        if tag == 'zp': return a,  b,  L

    
    Jf = (dy/2)*(dz/2)
    for k in range(nz-1):
        for j in range(ny-1):
            y0,y1 = j*dy,(j+1)*dy
            z0,z1 = k*dz,(k+1)*dz
            fn = [idx(nx-1,j,  k,  nx,ny), idx(nx-1,j+1,k,  nx,ny),
                  idx(nx-1,j+1,k+1,nx,ny), idx(nx-1,j,  k+1,nx,ny)]
            xi_n = [-1,1,1,-1]; eta_n = [-1,-1,1,1]
            for gi,p1 in enumerate(gp):
                for gj,p2 in enumerate(gp):
                    w  = gw[gi]*gw[gj]*Jf
                    yg = (y0+y1)/2 + p1*(dy/2)
                    zg = (z0+z1)/2 + p2*(dz/2)
                    Tx,Ty,Tz = traction_3d(L,yg,zg,'xp',E,nu,L)
                    for a in range(4):
                        Na = (1+xi_n[a]*p1)*(1+eta_n[a]*p2)/4.
                        F[fn[a],0]+=Na*Tx*w; F[fn[a],1]+=Na*Ty*w; F[fn[a],2]+=Na*Tz*w

    
    for k in range(nz-1):
        for j in range(ny-1):
            y0,y1 = j*dy,(j+1)*dy
            z0,z1 = k*dz,(k+1)*dz
            fn = [idx(0,j,  k,  nx,ny), idx(0,j+1,k,  nx,ny),
                  idx(0,j+1,k+1,nx,ny), idx(0,j,  k+1,nx,ny)]
            xi_n = [-1,1,1,-1]; eta_n = [-1,-1,1,1]
            for gi,p1 in enumerate(gp):
                for gj,p2 in enumerate(gp):
                    w  = gw[gi]*gw[gj]*Jf
                    yg = (y0+y1)/2 + p1*(dy/2)
                    zg = (z0+z1)/2 + p2*(dz/2)
                    Tx,Ty,Tz = traction_3d(0.,yg,zg,'xm',E,nu,L)
                    for a in range(4):
                        Na = (1+xi_n[a]*p1)*(1+eta_n[a]*p2)/4.
                        F[fn[a],0]+=Na*Tx*w; F[fn[a],1]+=Na*Ty*w; F[fn[a],2]+=Na*Tz*w

    
    Jf = (dx/2)*(dz/2)
    for k in range(nz-1):
        for i in range(nx-1):
            x0,x1 = i*dx,(i+1)*dx
            z0,z1 = k*dz,(k+1)*dz
            fn = [idx(i,  ny-1,k,  nx,ny), idx(i+1,ny-1,k,  nx,ny),
                  idx(i+1,ny-1,k+1,nx,ny), idx(i,  ny-1,k+1,nx,ny)]
            xi_n = [-1,1,1,-1]; eta_n = [-1,-1,1,1]
            for gi,p1 in enumerate(gp):
                for gj,p2 in enumerate(gp):
                    w  = gw[gi]*gw[gj]*Jf
                    xg = (x0+x1)/2 + p1*(dx/2)
                    zg = (z0+z1)/2 + p2*(dz/2)
                    Tx,Ty,Tz = traction_3d(xg,L,zg,'yp',E,nu,L)
                    for a in range(4):
                        Na = (1+xi_n[a]*p1)*(1+eta_n[a]*p2)/4.
                        F[fn[a],0]+=Na*Tx*w; F[fn[a],1]+=Na*Ty*w; F[fn[a],2]+=Na*Tz*w

    
    for k in range(nz-1):
        for i in range(nx-1):
            x0,x1 = i*dx,(i+1)*dx
            z0,z1 = k*dz,(k+1)*dz
            fn = [idx(i,  0,k,  nx,ny), idx(i+1,0,k,  nx,ny),
                  idx(i+1,0,k+1,nx,ny), idx(i,  0,k+1,nx,ny)]
            xi_n = [-1,1,1,-1]; eta_n = [-1,-1,1,1]
            for gi,p1 in enumerate(gp):
                for gj,p2 in enumerate(gp):
                    w  = gw[gi]*gw[gj]*Jf
                    xg = (x0+x1)/2 + p1*(dx/2)
                    zg = (z0+z1)/2 + p2*(dz/2)
                    Tx,Ty,Tz = traction_3d(xg,0.,zg,'ym',E,nu,L)
                    for a in range(4):
                        Na = (1+xi_n[a]*p1)*(1+eta_n[a]*p2)/4.
                        F[fn[a],0]+=Na*Tx*w; F[fn[a],1]+=Na*Ty*w; F[fn[a],2]+=Na*Tz*w

    
    Jf = (dx/2)*(dy/2)
    for j in range(ny-1):
        for i in range(nx-1):
            x0,x1 = i*dx,(i+1)*dx
            y0,y1 = j*dy,(j+1)*dy
            fn = [idx(i,  j,  nz-1,nx,ny), idx(i+1,j,  nz-1,nx,ny),
                  idx(i+1,j+1,nz-1,nx,ny), idx(i,  j+1,nz-1,nx,ny)]
            xi_n = [-1,1,1,-1]; eta_n = [-1,-1,1,1]
            for gi,p1 in enumerate(gp):
                for gj,p2 in enumerate(gp):
                    w  = gw[gi]*gw[gj]*Jf
                    xg = (x0+x1)/2 + p1*(dx/2)
                    yg = (y0+y1)/2 + p2*(dy/2)
                    Tx,Ty,Tz = traction_3d(xg,yg,L,'zp',E,nu,L)
                    for a in range(4):
                        Na = (1+xi_n[a]*p1)*(1+eta_n[a]*p2)/4.
                        F[fn[a],0]+=Na*Tx*w; F[fn[a],1]+=Na*Ty*w; F[fn[a],2]+=Na*Tz*w

    # Face z = 0
    for j in range(ny-1):
        for i in range(nx-1):
            x0,x1 = i*dx,(i+1)*dx
            y0,y1 = j*dy,(j+1)*dy
            fn = [idx(i,  j,  0,nx,ny), idx(i+1,j,  0,nx,ny),
                  idx(i+1,j+1,0,nx,ny), idx(i,  j+1,0,nx,ny)]
            xi_n = [-1,1,1,-1]; eta_n = [-1,-1,1,1]
            for gi,p1 in enumerate(gp):
                for gj,p2 in enumerate(gp):
                    w  = gw[gi]*gw[gj]*Jf
                    xg = (x0+x1)/2 + p1*(dx/2)
                    yg = (y0+y1)/2 + p2*(dy/2)
                    Tx,Ty,Tz = traction_3d(xg,yg,0.,'zm',E,nu,L)
                    for a in range(4):
                        Na = (1+xi_n[a]*p1)*(1+eta_n[a]*p2)/4.
                        F[fn[a],0]+=Na*Tx*w; F[fn[a],1]+=Na*Ty*w; F[fn[a],2]+=Na*Tz*w
 
    Jvol = (dx/2)*(dy/2)*(dz/2)
    hexa_xi   = np.array([-1,1,1,-1,-1,1,1,-1], dtype=float)
    hexa_eta  = np.array([-1,-1,1,1,-1,-1,1,1], dtype=float)
    hexa_zeta = np.array([-1,-1,-1,-1,1,1,1,1], dtype=float)
    for k in range(nz-1):
        for j in range(ny-1):
            for i in range(nx-1):
                x0,x1 = i*dx,(i+1)*dx
                y0,y1 = j*dy,(j+1)*dy
                z0,z1 = k*dz,(k+1)*dz
                hn = [idx(i,  j,  k,  nx,ny), idx(i+1,j,  k,  nx,ny),
                      idx(i+1,j+1,k,  nx,ny), idx(i,  j+1,k,  nx,ny),
                      idx(i,  j,  k+1,nx,ny), idx(i+1,j,  k+1,nx,ny),
                      idx(i+1,j+1,k+1,nx,ny), idx(i,  j+1,k+1,nx,ny)]
                for gi,p1 in enumerate(gp):
                    for gj,p2 in enumerate(gp):
                        for gk,p3 in enumerate(gp):
                            w   = gw[gi]*gw[gj]*gw[gk]*Jvol
                            xg  = (x0+x1)/2 + p1*(dx/2)
                            yg  = (y0+y1)/2 + p2*(dy/2)
                            zg  = (z0+z1)/2 + p3*(dz/2)
                            fx,fy,fz = body_force_mms(xg,yg,zg,E,nu,L)
                            for a in range(8):
                                Na = ((1+hexa_xi[a]*p1)
                                     *(1+hexa_eta[a]*p2)
                                     *(1+hexa_zeta[a]*p3)/8.)
                                F[hn[a],0]+=Na*fx*w
                                F[hn[a],1]+=Na*fy*w
                                F[hn[a],2]+=Na*fz*w

    return F
# ================= SOFA scene uniquement avec dirichlet ============================
def create_scene_3d(rootNode, L=1.0, E=1e6, nu=0.3, nx=6, ny=6, nz=6, visual=False):
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
                    maxNbIterationsNewton=10,
                    maxNbIterationsLineSearch=5,
                    lineSearchCoefficient=0.5,
                    relativeSuccessiveStoppingThreshold=1e-10,
                    absoluteResidualStoppingThreshold=1e-10,
                    absoluteEstimateDifferenceThreshold=1e-14,
                    relativeInitialStoppingThreshold=1e-10,
                    relativeEstimateDifferenceThreshold=1e-10)
    solid.addObject('SparseLDLSolver',
                    name="linearSolver",
                    template="CompressedRowSparseMatrixd")
    solid.addObject('StaticSolver',
                    name="staticSolver",
                    newtonSolver="@newtonSolver",
                    linearSolver="@linearSolver")
 
    eps     = 1e-8
    bnd     = []
    init_pos = nodes_3d.copy()

    for node_id, (xi, yi, zi) in enumerate(nodes_3d):
        on_bnd = (abs(xi) < eps or abs(xi - L) < eps or
                  abs(yi) < eps or abs(yi - L) < eps or
                  abs(zi) < eps or abs(zi - L) < eps)
        if on_bnd:
            bnd.append(node_id)
            init_pos[node_id, 0] += ux_mms(xi, yi, zi, L)
            init_pos[node_id, 1] += uy_mms(xi, yi, zi, L)
            init_pos[node_id, 2] += uz_mms(xi, yi, zi, L)

    dofs = solid.addObject('MechanicalObject', name="dofs", template="Vec3d",
                           rest_position=nodes_3d.tolist(),   
                           position=init_pos.tolist())         

    solid.addObject('HexahedronSetTopologyContainer', name="topology",
                    hexahedra=hexas.tolist())
    solid.addObject('HexahedronSetTopologyModifier')

    solid.addObject('LinearSmallStrainFEMForceField', name="FEM", template="Vec3d",
                    youngModulus=E, poissonRatio=nu, topology="@topology")

    
    solid.addObject('FixedProjectiveConstraint', name="bc_boundary",
                    indices=bnd)

    dx = L / (nx - 1); dy = L / (ny - 1); dz = L / (nz - 1)
    F  = np.zeros((len(nodes_3d), 3))
    gp = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    gw = np.array([1.0, 1.0])
    Jvol     = (dx/2)*(dy/2)*(dz/2)
    hexa_xi   = np.array([-1, 1, 1,-1,-1, 1, 1,-1], dtype=float)
    hexa_eta  = np.array([-1,-1, 1, 1,-1,-1, 1, 1], dtype=float)
    hexa_zeta = np.array([-1,-1,-1,-1, 1, 1, 1, 1], dtype=float)

    for k in range(nz-1):
        for j in range(ny-1):
            for i in range(nx-1):
                x0, x1 = i*dx, (i+1)*dx
                y0, y1 = j*dy, (j+1)*dy
                z0, z1 = k*dz, (k+1)*dz
                hn = [idx(i,   j,   k,   nx, ny), idx(i+1, j,   k,   nx, ny),
                      idx(i+1, j+1, k,   nx, ny), idx(i,   j+1, k,   nx, ny),
                      idx(i,   j,   k+1, nx, ny), idx(i+1, j,   k+1, nx, ny),
                      idx(i+1, j+1, k+1, nx, ny), idx(i,   j+1, k+1, nx, ny)]
                for gi, p1 in enumerate(gp):
                    for gj, p2 in enumerate(gp):
                        for gk, p3 in enumerate(gp):
                            w  = gw[gi]*gw[gj]*gw[gk]*Jvol
                            xg = (x0+x1)/2 + p1*(dx/2)
                            yg = (y0+y1)/2 + p2*(dy/2)
                            zg = (z0+z1)/2 + p3*(dz/2)
                            fx, fy, fz = body_force_mms(xg, yg, zg, E, nu, L)
                            for a in range(8):
                                Na = ((1 + hexa_xi[a]*p1)
                                     *(1 + hexa_eta[a]*p2)
                                     *(1 + hexa_zeta[a]*p3) / 8.)
                                F[hn[a], 0] += Na * fx * w
                                F[hn[a], 1] += Na * fy * w
                                F[hn[a], 2] += Na * fz * w

    
    F[bnd, :] = 0.0

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
    return create_scene_3d(
        rootNode, L=L_DEFAULT, E=E_DEFAULT, nu=NU_DEFAULT,
        nx=NX, ny=NY, nz=NZ, visual=True,
    )

 

def run_simulation_3d(L, E, nu, nx, ny, nz):
    root = Sofa.Core.Node("root")
    root.dt.value = 1.0

    dofs, nodes_3d, hexas = create_scene_3d(root, L=L, E=E, nu=nu,
                                             nx=nx, ny=ny, nz=nz)
    Sofa.Simulation.init(root)
    pos_init  = nodes_3d.copy()   
    pos_mms   = dofs.position.array().copy()  

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

_GP = np.array([-1.0/np.sqrt(3), 1.0/np.sqrt(3)])
_GW = np.array([1.0, 1.0])


def _shape_hexa(xi, eta, zeta):
    N        = (1+_XI_LOC*xi)*(1+_ETA_LOC*eta)*(1+_ZETA_LOC*zeta)/8.
    dN_dxi   = _XI_LOC  *(1+_ETA_LOC*eta) *(1+_ZETA_LOC*zeta)/8.
    dN_deta  = _ETA_LOC *(1+_XI_LOC*xi)   *(1+_ZETA_LOC*zeta)/8.
    dN_dzeta = _ZETA_LOC*(1+_XI_LOC*xi)   *(1+_ETA_LOC*eta)  /8.
    return N, dN_dxi, dN_deta, dN_dzeta


# ============================== L2 ===============================================


def compute_l2_3d(nodes_3d, ux, uy, uz, L, hexas):

    err2 = 0.0
    for hexa in hexas:
        xe = nodes_3d[hexa,0]; ye = nodes_3d[hexa,1]; ze = nodes_3d[hexa,2]
        ux_e = ux[hexa]; uy_e = uy[hexa]; uz_e = uz[hexa]
        for xi, wi in zip(_GP, _GW):
            for eta, wj in zip(_GP, _GW):
                for zeta, wk in zip(_GP, _GW):
                    N, dNx, dNe, dNz = _shape_hexa(xi, eta, zeta)
                    J = np.array([
                        [dNx@xe, dNx@ye, dNx@ze],
                        [dNe@xe, dNe@ye, dNe@ze],
                        [dNz@xe, dNz@ye, dNz@ze],
                    ])
                    detJ = np.linalg.det(J)
                    xg = N@xe; yg = N@ye; zg = N@ze
                    err2 += (
                        (N@ux_e - ux_mms(xg,yg,zg,L))**2
                      + (N@uy_e - uy_mms(xg,yg,zg,L))**2
                      + (N@uz_e - uz_mms(xg,yg,zg,L))**2
                    ) * wi*wj*wk * detJ
    return np.sqrt(err2)


# ================== H1 =======================================

def compute_h1_3d(nodes_3d, ux, uy, uz, L, hexas):
     
    err2 = 0.0
    for hexa in hexas:
        xe = nodes_3d[hexa,0]; ye = nodes_3d[hexa,1]; ze = nodes_3d[hexa,2]
        ux_e = ux[hexa]; uy_e = uy[hexa]; uz_e = uz[hexa]
        for xi, wi in zip(_GP, _GW):
            for eta, wj in zip(_GP, _GW):
                for zeta, wk in zip(_GP, _GW):
                    N, dNx, dNe, dNz = _shape_hexa(xi, eta, zeta)
                    J = np.array([
                        [dNx@xe, dNx@ye, dNx@ze],
                        [dNe@xe, dNe@ye, dNe@ze],
                        [dNz@xe, dNz@ye, dNz@ze],
                    ])
                    detJ = np.linalg.det(J)
                    Jinv = np.linalg.inv(J)
                    dNp  = np.vstack([dNx, dNe, dNz])   
                    dNph = Jinv.T @ dNp                   
                    xg = N@xe; yg = N@ye; zg = N@ze

                    
                    grad_h = np.array([
                        [dNph[0]@ux_e, dNph[1]@ux_e, dNph[2]@ux_e],
                        [dNph[0]@uy_e, dNph[1]@uy_e, dNph[2]@uy_e],
                        [dNph[0]@uz_e, dNph[1]@uz_e, dNph[2]@uz_e],
                    ])
                    
                    grad_ex = np.array([
                        [dux_dx(xg,yg,zg,L), dux_dy(xg,yg,zg,L), dux_dz(xg,yg,zg,L)],
                        [duy_dx(xg,yg,zg,L), duy_dy(xg,yg,zg,L), duy_dz(xg,yg,zg,L)],
                        [duz_dx(xg,yg,zg,L), duz_dy(xg,yg,zg,L), duz_dz(xg,yg,zg,L)],
                    ])
                    diff = grad_h - grad_ex
                    err2 += np.sum(diff**2) * wi*wj*wk * detJ
    return np.sqrt(err2)


# ============================= Conv study =============================

def convergence_study_3d(L, E, nu, nx_values, results_dir=RESULTS_DIR):
    os.makedirs(results_dir, exist_ok=True)

    hs, errs_l2, errs_h1 = [], [], []
    hdr = (f"{'nx':>5} | {'h':>8} | {'L2':>14} | {'ord_L2':>7}"
           f" | {'H1':>14} | {'ord_H1':>7}")
    txt_path = os.path.join(results_dir, f"convergence_3d_sinus_nu{nu}.txt")

    print(f"\n── Convergence MMS 3D sinus : Hexa Q1  nu={nu} ──\n{hdr}")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Convergence MMS 3D sinus — Hexa Q1  E={E}  nu={nu}\n{hdr}\n")
        for k, nx in enumerate(nx_values):
            ny = nz = nx
            h  = L / (nx - 1)
            nodes_3d, hexas, ux, uy, uz = run_simulation_3d(L, E, nu, nx, ny, nz)
            l2 = compute_l2_3d(nodes_3d, ux, uy, uz, L, hexas)
            h1 = compute_h1_3d(nodes_3d, ux, uy, uz, L, hexas)
            hs.append(h); errs_l2.append(l2); errs_h1.append(h1)
            ord_l2 = (f"{np.log(l2/errs_l2[k-1])/np.log(h/hs[k-1]):.2f}"
                      if k > 0 else "   —  ")
            ord_h1 = (f"{np.log(h1/errs_h1[k-1])/np.log(h/hs[k-1]):.2f}"
                      if k > 0 else "   —  ")
            line = (f"{nx:5d} | {h:8.4f} | {l2:14.6e} | {ord_l2:>7}"
                    f" | {h1:14.6e} | {ord_h1:>7}")
            print(line); f.write(line+"\n")

    hs_a = np.array(hs); l2_a = np.array(errs_l2); h1_a = np.array(errs_h1)
    h_ref = np.array([hs_a[0], hs_a[-1]])

    for errs, ylabel, slope, fname in [
        (l2_a, "err L2",     2, f"convergence_L2_3d_sinus_nu{nu}.png"),
        (h1_a, "err H1", 1, f"convergence_H1_3d_sinus_nu{nu}.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog(hs_a, errs, "o-", lw=2, ms=7, color="#1a73e8",
                  label=f"Hexa Q1 3D sinus  nu={nu}")
        ax.loglog(h_ref, errs[0]*(h_ref/hs_a[0])**slope,
                  ":", color="gray", lw=1.5, label=f"O(h^{slope})")
        ax.set_xlabel("h"); ax.set_ylabel(ylabel)
        ax.set_title(f"Convergence MMS 3D sinusoïdal — Hexa Q1\n{ylabel}  (nu={nu})")
        ax.legend(); ax.grid(True, alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, fname), dpi=150)
        plt.close(fig)
    return hs_a, l2_a, h1_a

 

def plot_displacement(nodes_3d, ux, uy, uz, L, nx, ny, nz, E, nu):
    mid_i = (nx-1)//2; mid_j = (ny-1)//2; mid_k = (nz-1)//2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    line_x = [idx(i, mid_j, mid_k, nx, ny) for i in range(nx)]
    xv = nodes_3d[line_x, 0]
    xf = np.linspace(0, L, 200)
    axes[0].plot(xv, ux[line_x], 'bo-', label='SOFA', markersize=7)
    axes[0].plot(xf, ux_mms(xf, nodes_3d[line_x[0],1], nodes_3d[line_x[0],2], L),
                 'r--', label='MMS exact', lw=2)
    axes[0].set(xlabel='x', ylabel='$u_x$',
                title=f'$u_x$(x)  y={nodes_3d[line_x[0],1]:.2f}, z={nodes_3d[line_x[0],2]:.2f}')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    line_y = [idx(mid_i, j, mid_k, nx, ny) for j in range(ny)]
    yv = nodes_3d[line_y, 1]
    axes[1].plot(yv, uy[line_y], 'go-', label='SOFA', markersize=7)
    axes[1].plot(xf, uy_mms(nodes_3d[line_y[0],0], xf, nodes_3d[line_y[0],2], L),
                 'r--', label='MMS exact', lw=2)
    axes[1].set(xlabel='y', ylabel='$u_y$',
                title=f'$u_y$(y)  x={nodes_3d[line_y[0],0]:.2f}, z={nodes_3d[line_y[0],2]:.2f}')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    line_z = [idx(mid_i, mid_j, k, nx, ny) for k in range(nz)]
    zv = nodes_3d[line_z, 2]
    axes[2].plot(zv, uz[line_z], 'ms-', label='SOFA', markersize=7)
    axes[2].plot(xf, uz_mms(nodes_3d[line_z[0],0], nodes_3d[line_z[0],1], xf, L),
                 'r--', label='MMS exact', lw=2)
    axes[2].set(xlabel='z', ylabel='$u_z$',
                title=f'$u_z$(z)  x={nodes_3d[line_z[0],0]:.2f}, y={nodes_3d[line_z[0],1]:.2f}')
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.suptitle(f'MMS 3D sinus — Hexa Q1 )', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'1d_sinus_nx{nx}.png'), dpi=150)
    plt.close()


if __name__ == "__main__":
    L  = L_DEFAULT
    E  = E_DEFAULT
    nu = NU_DEFAULT

    nodes_3d, hexas, ux, uy, uz = run_simulation_3d(L, E, nu, NX, NY, NZ)
    plot_displacement(nodes_3d, ux, uy, uz, L, NX, NY, NZ, E, nu)

    l2 = compute_l2_3d(nodes_3d, ux, uy, uz, L, hexas)
    h1 = compute_h1_3d(nodes_3d, ux, uy, uz, L, hexas)
    print(f"\nMaillage {NX}*{NY}*{NZ}")
    print(f"  Err L2      = {l2:.4e}")
    print(f"  Err H1  = {h1:.4e}")

    convergence_study_3d(
        L          = L,
        E          = E,
        nu         = nu,
        nx_values  = [3, 4, 6, 8, 11, 20],
        results_dir= RESULTS_DIR,
    )