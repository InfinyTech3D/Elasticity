"""Central map from tool-agnostic vocabulary to SOFA components."""

# Spatial dimension (DOF/embedding space) -> MechanicalObject / force-field vector template.
# This is also what selects the constitutive branch: SOFA derives the Lame parameters from
# DataTypes::spatial_dimensions, so Vec2d on a 2D element is plane stress and Vec3d is plane strain.
VEC_BY_SPATIAL_DIM = {1: "Vec1d", 2: "Vec2d", 3: "Vec3d"}

# Element kind -> its own (topological) dimension, independent of the space it is embedded in.
TOPOLOGICAL_DIM = {"edge": 1, "tri": 2, "quad": 2, "tet": 3, "hexa": 3}

# Element kind -> (topology container, RegularGridTopology connectivity field).
CONTAINER = {
    "edge": ("EdgeSetTopologyContainer",        "edges"),
    "tri":  ("TriangleSetTopologyContainer",    "triangles"),
    "quad": ("QuadSetTopologyContainer",        "quads"),
    "tet":  ("TetrahedronSetTopologyContainer", "tetrahedra"),
    "hexa": ("HexahedronSetTopologyContainer",  "hexahedra"),
}

# Element kind -> SOFA geometry element name, for compound component templates ("Vec3d,Hexahedron").
ELEMENT_CPP = {
    "edge": "Edge",
    "tri":  "Triangle",
    "quad": "Quad",
    "tet":  "Tetrahedron",
    "hexa": "Hexahedron",
}

# Element kind -> topology Data holding its boundary facets (edge: a facet is a vertex).
FACET_FIELD = {
    "edge": None,
    "tri":  "edges",
    "quad": "edges",
    "tet":  "triangles",
    "hexa": "quads",
}

BOUNDARY_KIND = {
    "tri":  "edge",
    "quad": "edge",
    "tet":  "tri",
    "hexa": "quad",
}

# Element kind -> topological mapping generating it from the grid, or None if grid-native.
MAPPING = {
    "edge": None,
    "quad": None,
    "hexa": None,
    "tri":  "Quad2TriangleTopologicalMapping",
    "tet":  "Hexa2TetraTopologicalMapping",
}
