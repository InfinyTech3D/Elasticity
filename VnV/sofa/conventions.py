"""Central map from tool-agnostic vocabulary to SOFA components."""

# Spatial dimension -> MechanicalObject / force-field vector template.
VEC_BY_DIM = {1: "Vec1d", 2: "Vec2d", 3: "Vec3d"}

# Element kind -> (topology container, RegularGridTopology connectivity field).
CONTAINER = {
    "edge": ("EdgeSetTopologyContainer",       "edges"),
    "quad": ("QuadSetTopologyContainer",       "quads"),
    "hexa": ("HexahedronSetTopologyContainer", "hexahedra"),
}
