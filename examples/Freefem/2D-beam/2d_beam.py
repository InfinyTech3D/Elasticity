import numpy as np
from pyfreefem import FreeFemRunner
import os

edp_path = os.path.abspath("2d_beam.edp")
runner = FreeFemRunner(edp_path)

results = runner.execute({
    'youngModulus': '210000',
    'rho': '7.85e-6',
    'g': '9810',
    'poissonRatio': '0.3',
    'nx': '140',
    'ny': '35'
})


uu      = results['uu[]']    
vv      = results['vv[]']      
xcoords = results['xcoords']
ycoords = results['ycoords']

print(f"Déplacement max ux : {uu.max():.6f}")
print(f"Déplacement max uy : {vv.min():.6f}")