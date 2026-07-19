"""Central map from tool-agnostic vocabulary to SOFA components."""

# Spatial dimension -> MechanicalObject / force-field vector template.
VEC_BY_DIM = {1: "Vec1d", 2: "Vec2d", 3: "Vec3d"}

# Element kind -> (topology container, RegularGridTopology connectivity field).
CONTAINER = {
    "edge": ("EdgeSetTopologyContainer",        "edges"),
    "tri":  ("TriangleSetTopologyContainer",    "triangles"),
    "quad": ("QuadSetTopologyContainer",        "quads"),
    "tet":  ("TetrahedronSetTopologyContainer", "tetrahedra"),
    "hexa": ("HexahedronSetTopologyContainer",  "hexahedra"),
}

# Element kind -> topology Data holding its boundary facets (edge: a facet is a vertex).
FACET_FIELD = {
    "edge": None,
    "tri":  "edges",
    "quad": "edges",
    "tet":  "triangles",
    "hexa": "quads",
}

# Element kind -> topological mapping generating it from the grid, or None if grid-native.
MAPPING = {
    "edge": None,
    "quad": None,
    "hexa": None,
    "tri":  "Quad2TriangleTopologicalMapping",
    "tet":  "Hexa2TetraTopologicalMapping",
}
