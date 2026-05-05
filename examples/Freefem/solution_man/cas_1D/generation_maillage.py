import gmsh
import os

def generate_bar1D(length, n, filename):
    """
    Génère un maillage 1D d'une barre [0, L] avec n éléments P1.
    Équivalent FreeFEM : meshL Th = segment(n, [x * L]);
    """
    gmsh.initialize()
    gmsh.model.add("bar1d")

    # Points extrémités
    gmsh.model.geo.addPoint(0,      0, 0, tag=1)  # x=0, label 1 (encastrement)
    gmsh.model.geo.addPoint(length, 0, 0, tag=2)  # x=L, label 2

    # Ligne = la barre
    line = gmsh.model.geo.addLine(1, 2, tag=1)
    gmsh.model.geo.synchronize()

    # Physical groups — points d'extrémité (dim=0) et la barre (dim=1)
    gmsh.model.addPhysicalGroup(0, [1], tag=1, name="Left")   # u=0 en x=0
    gmsh.model.addPhysicalGroup(0, [2], tag=2, name="Right")  # u=0 en x=L
    gmsh.model.addPhysicalGroup(1, [line], tag=10, name="Bar")

    # Maillage uniforme avec n+1 noeuds
    gmsh.model.mesh.setTransfiniteCurve(line, n + 1)
    gmsh.model.mesh.generate(1)
    gmsh.model.mesh.setOrder(1)

    os.makedirs("results", exist_ok=True)
    msh_path = os.path.join("results", filename)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(msh_path)
    gmsh.finalize()

    print(f"Maillage écrit : {msh_path}  ({n} éléments, {n+1} noeuds)")
    return msh_path


if __name__ == "__main__":
    generate_bar1D(length=1.0, n=16, filename="bar1d.msh")