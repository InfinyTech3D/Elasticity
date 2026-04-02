import numpy as np
from pyfreefem import FreeFemRunner
import os, shutil

project_dir = os.path.abspath(".")
edp_path    = os.path.join(project_dir, "3d-beam.edp")
mesh_path   = os.path.join(project_dir, "beam.msh")


runner = FreeFemRunner(edp_path)

results = runner.execute({
    'youngModulus': '210e9',
    'poissonRatio': '0.3',
    'rhoMat':       '7800',
    'grav':         '9.81',
})

ux      = results['dispX']
uy      = results['dispY']
uz      = results['dispZ']
xcoords = results['coordX']
ycoords = results['coordY']
zcoords = results['coordZ']

print(f"Nb noeuds     : {len(ux)}")
print(f"uz_max simulé : {abs(uz).max():.6e} m")
idx = abs(uz).argmax()
print(f"Position max  : x={xcoords[idx]:.3f}, y={ycoords[idx]:.3f}, z={zcoords[idx]:.3f}")