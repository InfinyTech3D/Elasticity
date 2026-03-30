"""
beam_2d_sofa.py
===============
Equivalent Python de beam-2d.scn (SOFA)
Poutre 2D sous gravité — formulation déformations planes (Vec3d)
Equivalent exact de beam-2d.edp (FreeFEM)


Export :
    Génère sofa_results.txt avec x, y, ux, uy pour chaque nœud
"""

import Sofa
import Sofa.Core
import numpy as np
import csv
import os
import time


# ─── PARAMÈTRES
E       = 21.5
NU      = 0.29
GRAVITY = -0.05
NX, NY  = 141, 36
TOTAL_MASS = 20.0
OUTPUT_FILE = "sofa_results.txt"



class ExportController(Sofa.Core.Controller):
    """
    Contrôleur SOFA : exporte les positions déformées après convergence.
    Répond à onAnimateEndEvent (fin de chaque pas de temps).
    """

    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.exported = False
        self.node     = kwargs.get("node", None)
        self.iteration = 0  # compter les itérations

    def onAnimateEndEvent(self, event):  
        self.iteration += 1
        
        if self.exported:
            return
        
        # Laisser le temps à la simulation de converger
        # 50 itérations suffisent généralement
        if self.iteration < 50:
            return

        root = self.getContext().getRootContext()
        strain = root.getChild("beam_plane_strain")
        if strain is None:
            print("[ExportController] Nœud beam_plane_strain introuvable")
            return

        dofs = strain.getObject("dofs")
        if dofs is None:
            print("[ExportController] MechanicalObject 'dofs' introuvable")
            return

        # positions déformées
        pos_def = np.array(dofs.position.value)
        
        # positions initiales
        try:
            pos_rest = np.array(dofs.rest_position.value)
        except:
            pos_rest = _build_initial_grid(NX, NY, y_offset=3.0)

        disp = pos_def - pos_rest

        # Afficher la convergence
        max_uy = np.abs(disp[:, 1]).max()
        print(f"[CONVERGENCE] NX={NX} NY={NY} nœuds={NX*NY} max|uy|={max_uy:.8f}")

        # export TXT
        try:
            with open(OUTPUT_FILE, "w") as f:
                f.write("x y ux uy\n")
                for i in range(len(pos_def)):
                    x = pos_rest[i, 0]
                    y = pos_rest[i, 1] - 3.0  # ramène à [0,2]
                    ux = disp[i, 0]
                    uy = disp[i, 1]
                    f.write(f"{x:.6f} {y:.6f} {ux:.8f} {uy:.8f}\n")
            
            print(f"[ExportController] ✓ Exporté {len(pos_def)} nœuds → {OUTPUT_FILE}")
            self.exported = True
            root.animate.value = False
            
        except Exception as e:
            print(f"[ExportController] Erreur d'écriture: {e}")


def _build_initial_grid(nx, ny, y_offset=0.0):
    """Reconstruit la grille régulière SOFA si rest_position n'est pas dispo."""
    xs = np.linspace(0, 10, nx)
    ys = np.linspace(y_offset, y_offset + 2, ny)
    pts = []
    for y in ys:
        for x in xs:
            pts.append([x, y, 0.0])
    return np.array(pts)


def createScene(rootNode):
    """Point d'entrée SOFA — construit le graphe de scène."""

    rootNode.name    = "root"
    rootNode.gravity = [0, GRAVITY, 0]
    rootNode.dt      = 1.0

    # ── Plugins 
    rootNode.addObject("RequiredPlugin", name="Elasticity")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Constraint.Projective")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Engine.Select")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.LinearSolver.Direct")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Mass")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.ODESolver.Backward")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.StateContainer")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Topology.Container.Dynamic")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Topology.Container.Grid")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Visual")
    rootNode.addObject("RequiredPlugin", name="Sofa.GL.Component.Rendering3D")

    rootNode.addObject("DefaultAnimationLoop")
    rootNode.addObject("VisualStyle",
        displayFlags="showBehaviorModels showForceFields showWireframe")

    # ── Contrôleur d'export 
    rootNode.addObject(ExportController(name="exporter", node=rootNode))

    # ── Nœud beam_plane_strain (Vec3d) 
    strain = rootNode.addChild("beam_plane_strain")

    # Réduire les logs pour plus de clarté
    strain.addObject("NewtonRaphsonSolver",
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
        relativeEstimateDifferenceThreshold=0,
    )
    strain.addObject("SparseLDLSolver",
        name="linearSolver",
        template="CompressedRowSparseMatrixd")
    strain.addObject("StaticSolver",
        name="staticSolver",
        newtonSolver="@newtonSolver",
        linearSolver="@linearSolver")

    # Grille régulière avec NX × NY points
    strain.addObject("RegularGridTopology",
        name="grid",
        min=[0, 3, 0],
        max=[10, 5, 0],
        n=[NX, NY, 1])
    strain.addObject("MechanicalObject",
        template="Vec3d",
        name="dofs")

    # ── Sous-nœud triangles 
    triangles = strain.addChild("triangles")
    triangles.addObject("TriangleSetTopologyContainer",
        name="topology", src="@../grid")
    triangles.addObject("TriangleSetTopologyModifier")
    triangles.addObject("TriangleSetGeometryAlgorithms",
        template="Vec3d", drawTriangles=True)
    triangles.addObject("MeshMatrixMass",
        totalMass=TOTAL_MASS, topology="@topology")
    triangles.addObject("LinearSmallStrainFEMForceField",
        name="FEM",
        youngModulus=E,
        poissonRatio=NU,
        topology="@topology")

    # ── Condition aux limites : encastrement à gauche 
    strain.addObject("BoxROI",
        name="fixed_roi",
        template="Vec3d",
        box=[-0.01, 2.99, -1.0, 0.01, 5.01, 1.0],
        drawBoxes=True)
    strain.addObject("FixedProjectiveConstraint",
        template="Vec3d",
        indices="@fixed_roi.indices")
    
    print(f"[createScene] ✓ Scène créée avec NX={NX}, NY={NY}, nœuds={NX*NY}")

    return rootNode


start = time.time()

# simulation

end = time.time()
print("Temps de calcul :", end - start)


if __name__ == "__main__":
    # Pour exécution directe (non-SOFA)
    print(f"Paramètres: NX={NX}, NY={NY}, output={OUTPUT_FILE}")
    print(f"Total nœuds: {NX*NY}")