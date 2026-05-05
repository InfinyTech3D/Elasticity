"""
Debug approfondi : teste plusieurs valeurs de nx pour identifier
la formule exacte du facteur de raideur de LinearSmallStrainFEMForceField Vec1d.
"""
import math
import os
import tempfile
import numpy as np
import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime

# ---- génère un fichier .msh Gmsh v2 pour une barre 1D ----
def write_msh(filepath, n_elems, length=1.0):
    """Génère un fichier .msh Gmsh v2 pour une barre 1D avec n_elems éléments."""
    n_nodes = n_elems + 1
    h = length / n_elems
    with open(filepath, 'w') as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write("$Nodes\n")
        f.write(f"{n_nodes}\n")
        for i in range(n_nodes):
            f.write(f"{i+1} {i*h:.10f} 0.0 0.0\n")
        f.write("$EndNodes\n")
        f.write("$Elements\n")
        # 2 éléments ponctuels (labels) + n_elems éléments lignes
        n_total = 2 + n_elems
        f.write(f"{n_total}\n")
        f.write(f"1 15 2 1 1 1\n")           # nœud x=0, label=1
        f.write(f"2 15 2 2 2 {n_nodes}\n")   # nœud x=L, label=2
        for i in range(n_elems):
            f.write(f"{i+3} 1 2 0 1 {i+1} {i+2}\n")
        f.write("$EndElements\n")


class SimpleExporter(Sofa.Core.Controller):
    def __init__(self, dofs_node, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dofs   = dofs_node
        self.x0     = None
        self.u      = None

    def onSimulationInitDoneEvent(self, e):
        self.x0 = self.dofs.position.array().flatten().copy()

    def onAnimateEndEvent(self, e):
        self.u = self.dofs.position.array().flatten() - self.x0


def run_sofa(msh_file, n_elems, E=1000.0, nu=0.3, L=1.0):
    """Lance SOFA avec force F=E en x=L, retourne u(L)."""
    eps  = 1e-8
    root = Sofa.Core.Node("root")

    plugins = ["Elasticity","Sofa.Component.Constraint.Projective",
               "Sofa.Component.Engine.Select","Sofa.Component.IO.Mesh",
               "Sofa.Component.LinearSolver.Direct","Sofa.Component.MechanicalLoad",
               "Sofa.Component.ODESolver.Backward","Sofa.Component.StateContainer",
               "Sofa.Component.Topology.Container.Dynamic","Sofa.Component.Visual",
               "Sofa.GL.Component.Rendering3D"]
    root.addObject('RequiredPlugin', pluginName=plugins)
    root.addObject('DefaultAnimationLoop')

    with root.addChild('Bar') as Bar:
        Bar.addObject('NewtonRaphsonSolver', name="ns", printLog=False,
                      maxNbIterationsNewton=1,
                      absoluteResidualStoppingThreshold=1e-12,
                      relativeInitialStoppingThreshold=1e-14)
        Bar.addObject('SparseLDLSolver', name="ls", template="CompressedRowSparseMatrixd")
        Bar.addObject('StaticSolver', name="ss", newtonSolver="@ns", linearSolver="@ls")
        Bar.addObject('MeshGmshLoader', name="loader", filename=msh_file)
        dofs = Bar.addObject('MechanicalObject', name="dofs", template="Vec1d", src="@loader")

        with Bar.addChild('edges') as Edges:
            Edges.addObject('EdgeSetTopologyContainer', name="topo", src="@../loader")
            Edges.addObject('LinearSmallStrainFEMForceField', name="FEM", template="Vec1d",
                            youngModulus=E, poissonRatio=nu, topology="@topo")

        Bar.addObject('BoxROI', name="fix", template="Vec1d",
                      box=[-eps,-eps,-eps, eps,eps,eps])
        Bar.addObject('FixedProjectiveConstraint', indices="@fix.indices")
        Bar.addObject('BoxROI', name="load", template="Vec1d",
                      box=[L-eps,-eps,-eps, L+eps,eps,eps])
        Bar.addObject('ConstantForceField', indices="@load.indices", forces=str(E), showArrowSize=0)

    exp = root.addObject(SimpleExporter(dofs_node=dofs, name="exp"))
    Sofa.Simulation.init(root)
    Sofa.Simulation.animate(root, root.dt.value)

    idx = np.argsort(exp.x0)
    return exp.x0[idx], exp.u[idx]


if __name__ == "__main__":
    import sys, json

    with open("params.json") as f:
        cfg = json.load(f)
    E  = float(cfg["youngModulus"])
    nu = float(cfg["poissonRatio"])
    L  = float(cfg["length"])

    print(f"\n{'n_elems':>8}  {'u(L) SOFA':>12}  {'u(L) exact':>12}  {'facteur':>10}  {'1/facteur':>10}")
    print("-" * 60)

    tmpdir = tempfile.mkdtemp()
    for n in [1, 2, 4, 9, 10, 20, 100]:
        msh = os.path.join(tmpdir, f"bar_{n}.msh")
        write_msh(msh, n, L)
        x, u = run_sofa(msh, n, E, nu, L)
        uL    = u[-1]
        exact = L   # u(L) = F*L/E = E*L/E = L
        fac   = uL / exact
        print(f"{n:>8}  {uL:>12.8f}  {exact:>12.8f}  {fac:>10.6f}  {1/fac:>10.6f}")

    print("\nConclusion : le facteur doit être constant si c'est un scaling global,")
    print("ou dépendre de n si c'est lié à l'assemblage.")