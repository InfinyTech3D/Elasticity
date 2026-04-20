"""
2D MMS Verification - Quadratic Solution
Plane Strain (Vec3d) / Plane Stress (Vec2d)

manufactured solution :
    u_x = x*(L-x)*y / (L^2·H)
    u_y = 0
Domaine : [0,L] * [0,H] rectangular beam   
"""

import numpy as np
import os
import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime
import matplotlib.pyplot as plt

RESULTS_DIR = "results_2d"
os.makedirs(RESULTS_DIR, exist_ok=True)



def _add_plugins(root):
    root.addObject('RequiredPlugin', pluginName=[
        "Elasticity",
        "Sofa.Component.Constraint.Projective",
        "Sofa.Component.Engine.Select",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.MechanicalLoad",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Grid",    
        "Sofa.Component.Topology.Container.Dynamic",  
        "Sofa.Component.Visual",
    ])



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


def traction_mms_2d(x, y, L, H, E, nu, formulation, normal):
    sxx, syy, sxy = stress_mms(x, y, L, H, E, nu, formulation)
    nx, ny = normal
    tx = sxx * nx + sxy * ny
    ty = sxy * nx + syy * ny
    return tx, ty


def compute_nodal_areas(triangles, positions):

    N = len(positions)
    nodal_area = np.zeros(N)
    for tri in triangles:
        n0, n1, n2 = tri
        pts = positions[[n0, n1, n2], :2]
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
        nodal_area[n0] += area / 3.0
        nodal_area[n1] += area / 3.0
        nodal_area[n2] += area / 3.0
    return nodal_area


def build_boundary_edges(triangles, boundary_idx_set):

    edge_set = set()
    for tri in triangles:
        for k in range(3):
            i = tri[k]
            j = tri[(k + 1) % 3]
            if i in boundary_idx_set and j in boundary_idx_set:
                edge = (min(i, j), max(i, j))
                edge_set.add(edge)
    return list(edge_set)


def integrate_neumann_on_edges(edges, positions, x_all, y_all,
                                L, H, E, nu, formulation, normal,
                                dof_dim):
    
    force_dict = {}

    for (i, j) in edges:
        xi, yi = x_all[i], y_all[i]
        xj, yj = x_all[j], y_all[j]
        L_e = np.sqrt((xj - xi)**2 + (yj - yi)**2)

        txi, tyi = traction_mms_2d(np.array([xi]), np.array([yi]),
                                    L, H, E, nu, formulation, normal)
        txj, tyj = traction_mms_2d(np.array([xj]), np.array([yj]),
                                    L, H, E, nu, formulation, normal)

        for node, tx, ty in [(i, txi[0], tyi[0]), (j, txj[0], tyj[0])]:
            if node not in force_dict:
                force_dict[node] = np.zeros(dof_dim)
            force_dict[node][0] += 0.5 * L_e * tx
            force_dict[node][1] += 0.5 * L_e * ty

    indices = list(force_dict.keys())
    forces  = [force_dict[n].tolist() for n in indices]
    return indices, forces


def build_scene_2d(root, L, H, nx, ny, E, nu, formulation):
    
    if formulation == "planeStrain":
        template = "Vec3d"
        dof_dim  = 3
    else:
        template = "Vec2d"
        dof_dim  = 2

    root_geo = Sofa.Core.Node("root_geo")
    _add_plugins(root_geo)
    root_geo.addObject('DefaultAnimationLoop')

    Beam_geo = root_geo.addChild('Beam')
    Beam_geo.addObject('RegularGridTopology',
                       name="grid",
                       min=[0.0, 0.0, 0.0],
                       max=[L,   H,   0.0],
                       n=[nx, ny, 1])

    Topo_geo = Beam_geo.addChild('Topo')
    Topo_geo.addObject('TriangleSetTopologyContainer', name="triangles", src="@../grid")
    Topo_geo.addObject('TriangleSetTopologyModifier')
    Topo_geo.addObject('TriangleSetGeometryAlgorithms', template=template)

    dofs_geo = Beam_geo.addObject('MechanicalObject',
                                   name="dofs", template=template, src="@grid")

    Sofa.Simulation.init(root_geo)

    positions = np.array(dofs_geo.position.value)
    tri_cont  = Topo_geo.getObject("triangles")
    triangles = np.array(tri_cont.triangles.value, dtype=int)

    Sofa.Simulation.unload(root_geo)

    x_all = positions[:, 0]
    y_all = positions[:, 1]
    N     = len(positions)
    eps   = 1e-9

    left_idx   = np.where(x_all < eps)[0].tolist()
    right_idx  = np.where(x_all > L - eps)[0].tolist()
    bottom_idx = np.where(y_all < eps)[0].tolist()
    top_idx    = np.where(y_all > H - eps)[0].tolist()

    nodal_area             = compute_nodal_areas(triangles, positions)
    fx_density, fy_density = body_force_mms(x_all, y_all, L, H, E, nu, formulation)
    fx_nodal = fx_density * nodal_area
    fy_nodal = fy_density * nodal_area

    if dof_dim == 3:
        body_forces = [[fx_nodal[i], fy_nodal[i], 0.0] for i in range(N)]
    else:
        body_forces = [[fx_nodal[i], fy_nodal[i]] for i in range(N)]

    right_set  = set(right_idx)
    bottom_set = set(bottom_idx) - set(left_idx)
    top_set    = set(top_idx)    - set(left_idx)

    neumann_indices = []
    neumann_forces  = []
    for node_set, normal in [
        (right_set,  [ 1.0,  0.0]),
        (bottom_set, [ 0.0, -1.0]),
        (top_set,    [ 0.0,  1.0]),
    ]:
        if not node_set:
            continue
        edges = build_boundary_edges(triangles, node_set)
        if not edges:
            continue
        idxs, forces = integrate_neumann_on_edges(
            edges, positions, x_all, y_all,
            L, H, E, nu, formulation, normal, dof_dim)
        if dof_dim == 3:
            forces = [[f[0], f[1], 0.0] for f in forces]
        neumann_indices.extend(idxs)
        neumann_forces.extend(forces)


    _add_plugins(root)
    root.addObject('VisualStyle', displayFlags="showWireframe")
    root.addObject('DefaultAnimationLoop')

    Beam = root.addChild('Beam')
    Beam.addObject('RegularGridTopology',
                   name="grid",
                   min=[0.0, 0.0, 0.0],
                   max=[L,   H,   0.0],
                   n=[nx, ny, 1])

    TopoNode = Beam.addChild('Topo')
    TopoNode.addObject('TriangleSetTopologyContainer', name="triangles", src="@../grid")
    TopoNode.addObject('TriangleSetTopologyModifier')
    TopoNode.addObject('TriangleSetGeometryAlgorithms', template=template)

    dofs = Beam.addObject('MechanicalObject',
                          name="dofs", template=template, src="@grid")

    # Solveurs
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

    Beam.addObject('LinearSmallStrainFEMForceField',
                   name="FEM",
                   template=template,
                   youngModulus=E,
                   poissonRatio=nu,
                   topology="@Topo/triangles")

    Beam.addObject('ConstantForceField',
                   name="bodyForce",
                   template=template,
                   indices=list(range(N)),
                   forces=body_forces)

    # Dirichlet 
    if left_idx:
        Beam.addObject('FixedProjectiveConstraint',
                       template=template,
                       indices=left_idx)

    # Neumann 
    if neumann_indices:
        Beam.addObject('ConstantForceField',
                       name="neumannBC",
                       template=template,
                       indices=neumann_indices,
                       forces=neumann_forces)

    
    Sofa.Simulation.init(root)

    return dofs, positions, triangles


#  errors 

def compute_errors_2d(positions, triangles, u_sofa, L, H):
    
    x_all = positions[:, 0]
    y_all = positions[:, 1]

    ux_ref, uy_ref = u_mms(x_all, y_all, L, H)
    err = u_sofa - np.column_stack([ux_ref, uy_ref])

    gauss_lam = np.array([[2/3, 1/6, 1/6],
                          [1/6, 2/3, 1/6],
                          [1/6, 1/6, 2/3]])
    gauss_w = np.array([1/6, 1/6, 1/6])

    l2_sq     = 0.0
    l2_ref_sq = 0.0

    for tri in triangles:
        n0, n1, n2 = tri
        pts  = positions[[n0, n1, n2], :2]
        v1   = pts[1] - pts[0]
        v2   = pts[2] - pts[0]
        area = 0.5 * abs(v1[0]*v2[1] - v1[1]*v2[0])
        jac  = 2.0 * area

        for k in range(3):
            lam = gauss_lam[k]
            xg  = lam @ pts[:, 0]
            yg  = lam @ pts[:, 1]

            ex = lam[0]*err[n0,0] + lam[1]*err[n1,0] + lam[2]*err[n2,0]
            ey = lam[0]*err[n0,1] + lam[1]*err[n1,1] + lam[2]*err[n2,1]

            uxe, uye = u_mms(np.array([xg]), np.array([yg]), L, H)

            l2_sq     += gauss_w[k] * jac * (ex**2 + ey**2)
            l2_ref_sq += gauss_w[k] * jac * (uxe[0]**2 + uye[0]**2)

    l2_abs = np.sqrt(l2_sq)
    l2_rel = np.sqrt(l2_sq / max(l2_ref_sq, 1e-30))
    

    return l2_abs, l2_rel

def plot_2d_results(positions, triangles, u_sofa, L, H, formulation, out_dir):
    
    import matplotlib.tri as mtri

    x = positions[:, 0]
    y = positions[:, 1]
    ux_ref, uy_ref = u_mms(x, y, L, H)

    triang = mtri.Triangulation(x, y, triangles)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(f"2D MMS Quadratic — {formulation}", fontsize=14)

    def _plot(ax, vals, title, cmap='viridis'):
        tc = ax.tripcolor(triang, vals, shading='gouraud', cmap=cmap)
        plt.colorbar(tc, ax=ax)
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    _plot(axes[0,0], u_sofa[:,0],          'SOFA: u_x')
    _plot(axes[0,1], ux_ref,               'Exact: u_x')
    _plot(axes[0,2], u_sofa[:,0] - ux_ref, 'Error u_x', 'RdBu')
    _plot(axes[1,0], u_sofa[:,1],          'SOFA: u_y')
    _plot(axes[1,1], uy_ref,               'Exact: u_y')
    _plot(axes[1,2], u_sofa[:,1] - uy_ref, 'Error u_y', 'RdBu')

    plt.tight_layout()
    out = os.path.join(out_dir, f"mms_2d_{formulation}.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"  Plot saved: {out}")



def convergence_study_2d(L, H, E, nu, formulation, mesh_configs):
    
    h_list    = []
    l2_list   = []
    

    print(f"\n{'nx':>4} | {'ny':>4} | {'h':>8} | {'L2 error':>12} |  {'Rate L2':>8}")
    

    for nx, ny in mesh_configs:
        root = Sofa.Core.Node("root")
        dofs_obj, positions, triangles = build_scene_2d(
            root, L, H, nx, ny, E, nu, formulation)

        Sofa.Simulation.animate(root, 1.0)

        pos_final = np.array(dofs_obj.position.value)   
        u_sofa    = pos_final[:, :2] - positions[:, :2]

        l2_abs, l2_rel = compute_errors_2d(
            positions, triangles, u_sofa, L, H)

        h = max(L / (nx - 1), H / (ny - 1))
        h_list.append(h)
        l2_list.append(l2_abs)
        

        Sofa.Simulation.unload(root)

        rate = (np.log(l2_list[-1] / l2_list[-2]) /
                np.log(h_list[-1] / h_list[-2])) if len(h_list) > 1 else 0.0

        print(f"{nx:4d} | {ny:4d} | {h:8.4f} | {l2_abs:12.6e} | {rate:8.2f}")


    if len(h_list) > 1:
        h_arr  = np.array(h_list)
        l2_arr = np.array(l2_list)

        plt.figure(figsize=(6, 5))
        plt.loglog(h_arr, l2_arr,
                   'bo-', label='L2 error', markersize=8)
        plt.loglog(h_arr, l2_arr[0] * (h_arr / h_arr[0])**2,
                   'k--', label='O(h²)', linewidth=2)
        plt.xlabel('Element size h')
        plt.ylabel('||e||_L2')
        plt.title(f'Convergence MMS 2D — {formulation}')
        plt.legend()
        plt.grid(True, alpha=0.3, which='both')
        plt.tight_layout()

        out = os.path.join(RESULTS_DIR, f"convergence_{formulation}.png")
        plt.savefig(out, dpi=200)
        plt.close()




def main():
    L   = 1.0
    H   = 0.5
    E   = 1e6
    nu  = 0.3
    formulation = "planeStress"   # or "planeStrain"
    nx, ny = 20, 10

    
    
    print(f"Mesh: {nx}×{ny} nodes")
    

    # Simulation unique
    root = Sofa.Core.Node("root")
    dofs_obj, positions, triangles = build_scene_2d(
        root, L, H, nx, ny, E, nu, formulation)

    Sofa.Simulation.animate(root, 1.0)

    pos_final = np.array(dofs_obj.position.value)   
    u_sofa    = pos_final[:, :2] - positions[:, :2]

    l2_abs, l2_rel= compute_errors_2d(
        positions, triangles, u_sofa, L, H)

    print(f"\n  L2 error (abs) : {l2_abs:.6e}")
    print(f"  L2 error (rel) : {l2_rel:.6e}")
    

    plot_2d_results(positions, triangles, u_sofa, L, H, formulation, RESULTS_DIR)

    Sofa.Simulation.unload(root)

    # Étude de convergence
    mesh_configs = [ (10, 5), (20, 10), (40, 20), (80, 40), (100,50), ]
    convergence_study_2d(L, H, E, nu, formulation, mesh_configs)


if __name__ == "__main__":
    main()