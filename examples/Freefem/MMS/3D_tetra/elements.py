"""Element strategies for MMS scenes.
...
"""

import numpy as np

from fem import (
    assemble_nodal_forces,
    assemble_traction,
    l2_error,
    h1_semi_error,
    quad_q1_rule,
    tri_p1_rule,
    hex_q1_rule,
    tet_p1_rule,
    edge_line_rule,
    quad_face_rule,
    tri_face_rule,          # <-- AJOUT : nécessaire pour _TetElement
)


# ---------------------------------------------------------------------------
# Boundary-facet helpers
# ---------------------------------------------------------------------------

def _boundary_edges(nx, ny):
    bottom = [(i, i + 1)                                    for i in range(nx - 1)]
    top    = [((ny - 1) * nx + i, (ny - 1) * nx + i + 1)    for i in range(nx - 1)]
    left   = [(j * nx, (j + 1) * nx)                        for j in range(ny - 1)]
    right  = [(j * nx + (nx - 1), (j + 1) * nx + (nx - 1))  for j in range(ny - 1)]
    return bottom, top, left, right


def _boundary_quads(nx, ny, nz):
    def idx(i, j, k):
        return i + j * nx + k * nx * ny

    xm = [(idx(0, j, k),    idx(0, j+1, k),    idx(0, j+1, k+1),    idx(0, j, k+1))
          for k in range(nz - 1) for j in range(ny - 1)]
    xp = [(idx(nx-1, j, k), idx(nx-1, j+1, k), idx(nx-1, j+1, k+1), idx(nx-1, j, k+1))
          for k in range(nz - 1) for j in range(ny - 1)]
    ym = [(idx(i, 0, k),    idx(i+1, 0, k),    idx(i+1, 0, k+1),    idx(i, 0, k+1))
          for k in range(nz - 1) for i in range(nx - 1)]
    yp = [(idx(i, ny-1, k), idx(i+1, ny-1, k), idx(i+1, ny-1, k+1), idx(i, ny-1, k+1))
          for k in range(nz - 1) for i in range(nx - 1)]
    zm = [(idx(i, j, 0),    idx(i+1, j, 0),    idx(i+1, j+1, 0),    idx(i, j+1, 0))
          for j in range(ny - 1) for i in range(nx - 1)]
    zp = [(idx(i, j, nz-1), idx(i+1, j, nz-1), idx(i+1, j+1, nz-1), idx(i, j+1, nz-1))
          for j in range(ny - 1) for i in range(nx - 1)]
    return xm, xp, ym, yp, zm, zp


def _boundary_triangles(nx, ny, nz, diagonal="main"):
    """Découpe chaque quad de _boundary_quads en 2 triangles.

    diagonal="main" coupe (0,1,2) & (0,2,3) ; "anti" coupe (1,2,3) & (1,3,0).
    Le sens qui matche la vraie diagonale interne de
    Hexa2TetraTopologicalMapping(swapping=True) n'est pas garanti a priori
    -> il faut tester les deux et garder celui qui stabilise le taux H1.
    """
    xm, xp, ym, yp, zm, zp = _boundary_quads(nx, ny, nz)

    def split(quads):
        tris = []
        for q in quads:
            if diagonal == "main":
                tris.append((q[0], q[1], q[2]))
                tris.append((q[0], q[2], q[3]))
            else:
                tris.append((q[1], q[2], q[3]))
                tris.append((q[1], q[3], q[0]))
        return tris

    return [split(f) for f in (xm, xp, ym, yp, zm, zp)]


# ---------------------------------------------------------------------------
# 2D elements (Vec2d / plane stress  OR  Vec3d / plane strain via dim arg)
# ---------------------------------------------------------------------------

class _ElementBase2D:
    @classmethod
    def compute_nodal_forces(cls, nodes_2d, conn, mms, L, E, nu, nx, ny, dim):
        xy = nodes_2d[:, :2]

        F = assemble_nodal_forces(
            lambda x, y: mms.source(x, y, E, nu, L, dim),
            xy, conn, cls._source_rule(mms))

        bottom, top, left, right = _boundary_edges(nx, ny)
        sides = [(bottom, 0.0, -1.0),
                 (top,    0.0, +1.0),
                 (left,  -1.0,  0.0),
                 (right, +1.0,  0.0)]
        edge_rule = edge_line_rule(2)
        for edges, nrm_x, nrm_y in sides:
            F += assemble_traction(
                lambda x, y, nx=nrm_x, ny=nrm_y:
                    mms.traction(x, y, nx, ny, E, nu, L, dim),
                xy, edges, edge_rule)
        return F

    @classmethod
    def compute_l2(cls, sol, mms, L):
        xy = sol.nodes[:, :2]
        return l2_error(
            xy, sol.conn, np.column_stack([sol.ux, sol.uy]),
            lambda x, y: mms.u_ex(x, y, L),
            cls.ELEMENT_RULE)

    @classmethod
    def compute_h1(cls, sol, mms, L):
        xy = sol.nodes[:, :2]
        return h1_semi_error(
            xy, sol.conn, np.column_stack([sol.ux, sol.uy]),
            lambda x, y: mms.grad_u_ex(x, y, L),
            cls.ELEMENT_RULE)


class _QuadElement(_ElementBase2D):
    LABEL        = "Q1 quad"
    ELEMENT_RULE = staticmethod(quad_q1_rule(2))

    @staticmethod
    def _source_rule(mms):
        rule = mms.source_quadrature_quad
        if rule is None:
            raise ValueError(
                f"{type(mms).__name__}.source_quadrature_quad must be set")
        return rule

    @staticmethod
    def add_topology(Beam):
        topology = Beam.addObject("QuadSetTopologyContainer", name="topology",
                                  quads="@../Grid/grid.quads",
                                  position="@../Grid/grid.position")
        Beam.addObject("QuadSetTopologyModifier")
        return topology

    @staticmethod
    def read_connectivity(topology):
        return topology.quads.array().copy()

    @staticmethod
    def to_triangles(conn):
        tris = []
        for q in conn:
            tris.append([q[0], q[1], q[2]])
            tris.append([q[0], q[2], q[3]])
        return np.array(tris)


class _TriElement(_ElementBase2D):
    LABEL        = "P1 tri"
    ELEMENT_RULE = staticmethod(tri_p1_rule(3))

    @staticmethod
    def _source_rule(mms):
        rule = mms.source_quadrature_tri
        if rule is None:
            raise ValueError(
                f"{type(mms).__name__}.source_quadrature_tri must be set")
        return rule

    @staticmethod
    def add_topology(Beam):
        topology = Beam.addObject("TriangleSetTopologyContainer", name="topology")
        Beam.addObject("Quad2TriangleTopologicalMapping",
                       input="@../Grid/grid", output="@topology")
        Beam.addObject("TriangleSetTopologyModifier")
        return topology

    @staticmethod
    def read_connectivity(topology):
        return topology.triangles.array().copy()

    @staticmethod
    def to_triangles(conn):
        return conn


# ---------------------------------------------------------------------------
# 3D elements (Vec3d / full Hooke)
# ---------------------------------------------------------------------------

class _ElementBase3D:
    @classmethod
    def compute_nodal_forces(cls, nodes_3d, conn, mms, L, E, nu, nx, ny, nz):
        xyz = nodes_3d[:, :3]

        F = assemble_nodal_forces(
            lambda x, y, z: mms.source(x, y, z, E, nu, L),
            xyz, conn, cls._source_rule(mms))

        xm, xp, ym, yp, zm, zp = _boundary_quads(nx, ny, nz)
        sides = [(xm, -1.0, 0.0, 0.0),
                 (xp, +1.0, 0.0, 0.0),
                 (ym,  0.0, -1.0, 0.0),
                 (yp,  0.0, +1.0, 0.0),
                 (zm,  0.0, 0.0, -1.0),
                 (zp,  0.0, 0.0, +1.0)]
        face_rule = quad_face_rule(2)
        for quads, nrm_x, nrm_y, nrm_z in sides:
            F += assemble_traction(
                lambda x, y, z, nx=nrm_x, ny=nrm_y, nz=nrm_z:
                    mms.traction(x, y, z, nx, ny, nz, E, nu, L),
                xyz, quads, face_rule)
        return F

    @classmethod
    def compute_l2(cls, sol, mms, L):
        return l2_error(
            sol.nodes, sol.conn,
            np.column_stack([sol.ux, sol.uy, sol.uz]),
            lambda x, y, z: mms.u_ex(x, y, z, L),
            cls.ELEMENT_RULE)

    @classmethod
    def compute_h1(cls, sol, mms, L):
        return h1_semi_error(
            sol.nodes, sol.conn,
            np.column_stack([sol.ux, sol.uy, sol.uz]),
            lambda x, y, z: mms.grad_u_ex(x, y, z, L),
            cls.ELEMENT_RULE)


class _HexElement(_ElementBase3D):
    LABEL        = "Q1 hex"
    ELEMENT_RULE = staticmethod(hex_q1_rule(2))

    @staticmethod
    def _source_rule(mms):
        rule = mms.source_quadrature_hex
        if rule is None:
            raise ValueError(
                f"{type(mms).__name__}.source_quadrature_hex must be set")
        return rule

    @staticmethod
    def add_topology(Solid):
        topology = Solid.addObject("HexahedronSetTopologyContainer",
                                   name="topology",
                                   hexahedra="@../Grid/grid.hexahedra",
                                   position="@../Grid/grid.position")
        Solid.addObject("HexahedronSetTopologyModifier")
        return topology

    @staticmethod
    def read_connectivity(topology):
        return topology.hexahedra.array().copy()


# ---------------------------------------------------------------------------
# P1 tet — NOTE : compute_nodal_forces est SURCHARGÉE ici (pas héritée de
# _ElementBase3D) car la traction Neumann doit être assemblée sur des
# facettes TRIANGULAIRES (tri_face_rule / _boundary_triangles), cohérentes
# avec la base P1 réelle de l'élément — pas sur des facettes quad Q1
# (quad_face_rule / _boundary_quads) comme le fait la version héritée.
# ---------------------------------------------------------------------------

class _TetElement(_ElementBase3D):
    LABEL        = "P1 tet"
    ELEMENT_RULE = staticmethod(tet_p1_rule(4))

    @staticmethod
    def _source_rule(mms):
        rule = mms.source_quadrature_tet
        if rule is None:
            raise ValueError(
                f"{type(mms).__name__}.source_quadrature_tet must be set")
        return rule

    @staticmethod
    def add_topology(Solid):
        topology = Solid.addObject("TetrahedronSetTopologyContainer", name="topology")
        Solid.addObject("Hexa2TetraTopologicalMapping",
                input="@../Grid/grid", output="@topology", swapping=True)
        Solid.addObject("TetrahedronSetTopologyModifier")
        return topology

    @staticmethod
    def read_connectivity(topology):
        return topology.tetrahedra.array().copy()

    @classmethod
    def compute_nodal_forces(cls, nodes_3d, conn, mms, L, E, nu, nx, ny, nz,
                             diagonal="main"):
        xyz = nodes_3d[:, :3]

        F = assemble_nodal_forces(
            lambda x, y, z: mms.source(x, y, z, E, nu, L),
            xyz, conn, cls._source_rule(mms))
        
        sides = _boundary_triangles_from_conn(conn, xyz, L)
        xm, xp, ym, yp, zm, zp = (sides["xm"], sides["xp"],
                          sides["ym"], sides["yp"],
                          sides["zm"], sides["zp"])

        tri_rule = tri_face_rule(3)
        for tris, nrm_x, nrm_y, nrm_z in sides:
            F += assemble_traction(
                lambda x, y, z, nx=nrm_x, ny=nrm_y, nz=nrm_z:
                    mms.traction(x, y, z, nx, ny, nz, E, nu, L),
                xyz, tris, tri_rule)
        return F
    


def _boundary_triangles_from_conn(conn, xyz, L, eps=1e-9):
    """Extrait les vraies faces de bord du maillage tet à partir de sa
    connectivité (une face partagée par un seul tet = face de bord),
    avec orientation sortante calculée géométriquement, puis les groupe
    par face du cube. Contrairement à `_boundary_triangles`, ceci est
    garanti cohérent avec la triangulation réelle produite par
    Hexa2TetraTopologicalMapping, quel que soit `swapping`.
    """
    face_local = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    count = {}
    for tet in conn:
        for loc in face_local:
            key = tuple(sorted(tet[i] for i in loc))
            count[key] = count.get(key, 0) + 1

    sides = {"xm": [], "xp": [], "ym": [], "yp": [], "zm": [], "zp": []}
    for tet in conn:
        for loc in face_local:
            tri = [tet[i] for i in loc]
            key = tuple(sorted(tri))
            if count[key] != 1:
                continue
            opp = [v for v in tet if v not in tri][0]
            p0, p1, p2 = xyz[tri[0]], xyz[tri[1]], xyz[tri[2]]
            n = np.cross(p1 - p0, p2 - p0)
            if np.dot(n, xyz[opp] - p0) > 0:
                tri = [tri[0], tri[2], tri[1]]
                n = -n
            nn = n / np.linalg.norm(n)
            c  = (xyz[tri[0]] + xyz[tri[1]] + xyz[tri[2]]) / 3.0
            if   abs(c[0])     < eps and nn[0] < 0: sides["xm"].append(tuple(tri))
            elif abs(c[0] - L) < eps and nn[0] > 0: sides["xp"].append(tuple(tri))
            elif abs(c[1])     < eps and nn[1] < 0: sides["ym"].append(tuple(tri))
            elif abs(c[1] - L) < eps and nn[1] > 0: sides["yp"].append(tuple(tri))
            elif abs(c[2])     < eps and nn[2] < 0: sides["zm"].append(tuple(tri))
            elif abs(c[2] - L) < eps and nn[2] > 0: sides["zp"].append(tuple(tri))
    return sides


element_quad = _QuadElement()
element_tri  = _TriElement()
element_hex  = _HexElement()
element_tet  = _TetElement()