"""
2D MMS Verification - Quadratic Solution
Plane Strain ===>    template Vec3d  
Plane Stress ===>    template Vec2d
Solution: u_x = x*(L-x)*y / (L^2·H),  u_y = 0
Domain: [0,L] * [0,H] rectangular beam
"""

import numpy as np
import os
import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

RESULTS_DIR = "results_2d"

# MM solution 
def u_mms(x, y, L, H):
    ux = x * (L - x) * y / (L**2 * H)
    uy = np.zeros_like(x)
    return ux, uy


def strain_mms(x, y, L, H):
    exx = (L - 2*x) * y / (L**2 * H)
    eyy = np.zeros_like(x)
    exy = 0.5 * x * (L - x) / (L**2 * H)
    return exx, eyy, exy


def stress_mms(x, y, L, H, E, nu, formulation):
    exx, eyy, exy = strain_mms(x, y, L, H)
    if formulation == "planeStrain":
        c = E / ((1 + nu) * (1 - 2*nu))
        sxx = c * ((1 - nu)*exx + nu*eyy)
        syy = c * (nu*exx + (1 - nu)*eyy)
        sxy = E / (2*(1 + nu)) * 2*exy
    else:  # planeStress
        c = E / (1 - nu**2)
        sxx = c * (exx + nu*eyy)
        syy = c * (nu*exx + eyy)
        sxy = E / (2*(1 + nu)) * 2*exy
    return sxx, syy, sxy


def body_force_mms(x, y, L, H, E, nu, formulation):
    
    if formulation == "planeStrain":
        c = E / ((1 + nu) * (1 - 2*nu))
        fx = c * (1 - nu) * 2*y / (L**2 * H)
    else:
        c = E / (1 - nu**2)
        fx = c * 2*y / (L**2 * H)
    fy = np.zeros_like(x)
    return fx, fy



def generate_mesh_gmsh_v2(L, H, nx, ny, output_file):

    x_arr = np.linspace(0, L, nx)
    y_arr = np.linspace(0, H, ny)
    X, Y  = np.meshgrid(x_arr, y_arr)         
    nodes_xy = np.column_stack([X.ravel(), Y.ravel()])
    N = len(nodes_xy)

    def idx(i, j):   
        return j * nx + i

    tris = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            n00 = idx(i,   j);   n10 = idx(i+1, j)
            n01 = idx(i,   j+1); n11 = idx(i+1, j+1)
            tris.append([n00, n10, n11])
            tris.append([n00, n11, n01])
    tris = np.array(tris, dtype=int)
    M = len(tris)

    with open(output_file, 'w') as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write(f"$Nodes\n{N}\n")
        for k, (x, y) in enumerate(nodes_xy):
            f.write(f"{k+1} {x:.10f} {y:.10f} 0.0\n")
        f.write("$EndNodes\n")
        f.write(f"$Elements\n{M}\n")
        for k, tri in enumerate(tris):
            f.write(f"{k+1} 2 2 1 1 {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
        f.write("$EndElements\n")

    nodes3d = np.column_stack([nodes_xy, np.zeros(N)])
    return nodes3d, tris


def nodal_body_forces(nodes, triangles, L, H, E, nu, formulation):
    N = len(nodes)
    F = np.zeros((N, 2))
    for tri in triangles:
        n0, n1, n2 = tri
        pts  = nodes[[n0, n1, n2], :2]
        v1   = pts[1] - pts[0];  v2 = pts[2] - pts[0]
        area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
        xc, yc = pts.mean(axis=0)
        fx, fy = body_force_mms(np.array([xc]), np.array([yc]),
                                L, H, E, nu, formulation)
        for nid in [n0, n1, n2]:
            F[nid, 0] += (1/3) * fx[0] * area
            F[nid, 1] += (1/3) * fy[0] * area
    return F



def build_scene_2d(root, nodes, triangles, L, H, E, nu, formulation):
    
    if formulation == "planeStrain":
        template = "Vec3d"
        dof_dim  = 3
    else:
        template = "Vec2d"
        dof_dim  = 2

    
    root.addObject('RequiredPlugin', pluginName=[
        "Elasticity",
        "Sofa.Component.Constraint.Projective",
        "Sofa.Component.Engine.Select",
        "Sofa.Component.IO.Mesh",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.MechanicalLoad",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Dynamic",
    ])
    root.addObject('DefaultAnimationLoop')
    root.gravity = [0, -1, 0]

    
    N = len(nodes)
    x_all, y_all = nodes[:, 0], nodes[:, 1]
    eps_tol = 1e-9
    is_boundary = (
        (x_all < eps_tol) | (x_all > L - eps_tol) |
        (y_all < eps_tol) | (y_all > H - eps_tol)
    )
    boundary_idx = np.where(is_boundary)[0].tolist()
    interior_idx = np.where(~is_boundary)[0].tolist()

    
    F_nodal = nodal_body_forces(nodes, triangles, L, H, E, nu, formulation)

    
    ux_mms, uy_mms = u_mms(x_all, y_all, L, H)

    if dof_dim == 3:
        pos_init = np.column_stack([nodes[:, 0], nodes[:, 1], np.zeros(N)])
    else:
        pos_init = nodes[:, :2].copy()

    for i in boundary_idx:
        pos_init[i, 0] += ux_mms[i]
        pos_init[i, 1] += uy_mms[i]

    pos_list = pos_init.tolist()
    tris_list = triangles.tolist()   

    if dof_dim == 3:
        forces_list = [[F_nodal[i, 0], F_nodal[i, 1], 0.0] for i in interior_idx]
    else:
        forces_list = [[F_nodal[i, 0], F_nodal[i, 1]] for i in interior_idx]

    Beam = root.addChild('Beam')

    Beam.addObject('NewtonRaphsonSolver',
                   name="newtonSolver",
                   maxNbIterationsNewton=30,
                   absoluteResidualStoppingThreshold=1e-12,
                   printLog=False)
    Beam.addObject('SparseLDLSolver',
                   name="linearSolver",
                   template="CompressedRowSparseMatrixd")
    Beam.addObject('StaticSolver',
                   name="staticSolver",
                   newtonSolver="@newtonSolver",
                   linearSolver="@linearSolver")

    dofs = Beam.addObject('MechanicalObject',
                          name="dofs",
                          template=template,
                          position=pos_list)

    Beam.addObject('TriangleSetTopologyContainer',
                   name="topology",
                   triangles=tris_list)
    Beam.addObject('TriangleSetTopologyModifier')

    # Pas d'attribut 'formulation' ici !
    Beam.addObject('LinearSmallStrainFEMForceField',
                   name="FEM",
                   template=template,
                   youngModulus=E,
                   poissonRatio=nu,
                   topology="@topology")

    if interior_idx:
        Beam.addObject('ConstantForceField',
                       name="bodyForce",
                       template=template,
                       indices=interior_idx,
                       forces=forces_list)

    Beam.addObject('FixedProjectiveConstraint',
                   name="dirichlet",
                   template=template,
                   indices=boundary_idx)

    return dofs, pos_init


def compute_errors(nodes, triangles, u_sofa, L, H):
    x_all, y_all = nodes[:, 0], nodes[:, 1]
    ux_ref, uy_ref = u_mms(x_all, y_all, L, H)
    err = u_sofa - np.column_stack([ux_ref, uy_ref])

    l2_sq = 0.0; l2_ref_sq = 0.0
    xi = np.array([[2/3, 1/6, 1/6],
                   [1/6, 2/3, 1/6],
                   [1/6, 1/6, 2/3]])
    w = np.array([1/3, 1/3, 1/3])

    for tri in triangles:
        n0, n1, n2 = tri
        pts  = nodes[[n0, n1, n2], :2]
        v1   = pts[1] - pts[0];  v2 = pts[2] - pts[0]
        area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
        for k in range(3):
            lam = xi[k]
            xg  = lam @ pts[:, 0]; yg = lam @ pts[:, 1]
            ex  = lam[0]*err[n0,0] + lam[1]*err[n1,0] + lam[2]*err[n2,0]
            ey  = lam[0]*err[n0,1] + lam[1]*err[n1,1] + lam[2]*err[n2,1]
            uxe, uye = u_mms(np.array([xg]), np.array([yg]), L, H)
            l2_sq     += w[k] * area * (ex**2 + ey**2)
            l2_ref_sq += w[k] * area * (uxe[0]**2 + uye[0]**2)

    return np.sqrt(l2_sq), np.sqrt(l2_sq / max(l2_ref_sq, 1e-30))



def plot_results(nodes, triangles, u_sofa, L, H, formulation, out_dir):
    x, y = nodes[:, 0], nodes[:, 1]
    ux_ref, uy_ref = u_mms(x, y, L, H)
    triang = mtri.Triangulation(x, y, triangles)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(f"2D MMS Quadratique — {formulation}", fontsize=14)

    def _plot(ax, vals, title, cmap='viridis'):
        tc = ax.tripcolor(triang, vals, shading='gouraud', cmap=cmap)
        plt.colorbar(tc, ax=ax)
        ax.set_title(title); ax.set_aspect('equal')
        ax.set_xlabel('x'); ax.set_ylabel('y')

    _plot(axes[0,0], u_sofa[:,0],          'SOFA : u_x')
    _plot(axes[0,1], ux_ref,               'Exact : u_x')
    _plot(axes[0,2], u_sofa[:,0] - ux_ref, 'Erreur u_x', 'RdBu')
    _plot(axes[1,0], u_sofa[:,1],          'SOFA : u_y')
    _plot(axes[1,1], uy_ref,               'Exact : u_y')
    _plot(axes[1,2], u_sofa[:,1] - uy_ref, 'Erreur u_y', 'RdBu')

    plt.tight_layout()
    out = os.path.join(out_dir, f"mms_2d_{formulation}.png")
    plt.savefig(out, dpi=200); plt.close()
    

#    CONVERGENCE
def convergence_study(L, H, E, nu, formulation, mesh_sizes):
    h_list, l2_list = [], []
    for (nx, ny) in mesh_sizes:
        msh = os.path.join(RESULTS_DIR, f"mesh_{nx}x{ny}.msh")
        nodes, tris = generate_mesh_gmsh_v2(L, H, nx, ny, msh)

        root = Sofa.Core.Node("root")
        dofs_obj, pos0 = build_scene_2d(root, nodes, tris, L, H, E, nu, formulation)
        Sofa.Simulation.init(root)
        Sofa.Simulation.animate(root, root.dt.value)

        pos_final = np.array(dofs_obj.position.toList())
        u_sofa = pos_final[:, :2] - pos0[:, :2]

        l2_abs, l2_rel = compute_errors(nodes, tris, u_sofa, L, H)
        h = max(L/(nx-1), H/(ny-1))
        h_list.append(h); l2_list.append(l2_abs)
        Sofa.Simulation.unload(root)
        print(f"  nx={nx:3d} ny={ny:2d}  h={h:.4f}  L2={l2_abs:.4e}  L2_rel={l2_rel:.4e}")

    if len(h_list) > 1:
        orders = [np.log(l2_list[i]/l2_list[i-1]) / np.log(h_list[i]/h_list[i-1])
                  for i in range(1, len(h_list))]
        print(f"  Ordres : {[f'{o:.2f}' for o in orders]}")

    plt.figure(figsize=(6, 5))
    h_arr = np.array(h_list)
    plt.loglog(h_arr, l2_list, 'o-', label='L2 error')
    plt.loglog(h_arr, l2_list[0]*(h_arr/h_arr[0])**2, '--', label='slope 2')
    plt.xlabel('h'); plt.ylabel('||e||_L2')
    plt.title(f'Convergence MMS 2D — {formulation}')
    plt.legend(); plt.grid(True, which='both'); plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"convergence_{formulation}.png")
    plt.savefig(out, dpi=200); plt.close()





def traction_mms_2d(x, y, L, H, E, nu, formulation, normal):
    """t = σ · n sur une frontière de normale n"""
    sxx, syy, sxy = stress_mms(x, y, L, H, E, nu, formulation)
    nx, ny = normal
    tx = sxx * nx + sxy * ny
    ty = sxy * nx + syy * ny
    return tx, ty
    



def main():
    L, H        = 1.0, 0.5
    E, nu       = 1e6, 0.3
    formulation = "planeStrain"   
    nx, ny      = 80,20

    mesh_file = os.path.join(RESULTS_DIR, f"mesh_{nx}x{ny}.msh")

    

    nodes, triangles = generate_mesh_gmsh_v2(L, H, nx, ny, mesh_file)
    print(f"  Maillage : {len(nodes)} nœuds, {len(triangles)} triangles")

    root = Sofa.Core.Node("root")
    dofs_obj, pos0 = build_scene_2d(root, nodes, triangles, L, H, E, nu, formulation)

    Sofa.Simulation.init(root)
    Sofa.Simulation.animate(root, root.dt.value)

    pos_final = np.array(dofs_obj.position.toList())
    u_sofa    = pos_final[:, :2] - pos0[:, :2]   

    l2_abs, l2_rel = compute_errors(nodes, triangles, u_sofa, L, H)
    print(f"\n  Erreur L2 absolue : {l2_abs:.6e}")
    print(f"  Erreur L2 relative: {l2_rel:.6e}")

    plot_results(nodes, triangles, u_sofa, L, H, formulation, RESULTS_DIR)

    Sofa.Simulation.unload(root)


    convergence_study(L, H, E, nu, formulation,
                  mesh_sizes=[(5,2),(10,3),(20,5),(40,10),(80,20)])

    


if __name__ == "__main__":
    main()