"""
1D Bar Simulation - Traction Load - Comparison File
"""
import json
import os
import sys
import math
import numpy as np
from sofa_bar_traction import sofaRun
from pyfreefem import FreeFemRunner

if __name__ == "__main__":

    # --- Read parameters from JSON file ---
    config_file = sys.argv[1] if len(sys.argv) > 1 else "params.json"
    with open(config_file) as f:
        cfg = json.load(f)

    length        = float(cfg["length"])
    nx            = int(cfg["nx"])
    force         = float(cfg["force"])
    young_modulus = float(cfg["youngModulus"])
    poisson_ratio = float(cfg["poissonRatio"])

    # --- Run FreeFEM ---
    runner = FreeFemRunner("freefem_bar_traction.edp")
    exports = runner.execute({
        'youngModulus': young_modulus,
        'force': force,
        'nx': nx,
        'length': length,
    })
    # Extract coordinates and displacements computed by FreeFEM
    x_ff = exports['xcoords']
    u_ff = exports['u[]']
    x_ff_final = x_ff + u_ff

    os.makedirs("results", exist_ok=True)
    np.savetxt(
        os.path.join("results", "freefem_results.txt"),
        np.column_stack([x_ff, x_ff_final, u_ff]),
        header="x_initial x_final u_x"
    )

    # --- Run SOFA ---
    x_sofa, u_sofa = sofaRun(length=length
            , force=force
            , young_modulus=young_modulus
            , poisson_ratio=poisson_ratio
            , nx=nx)

    # --- Compare Results ---
    with open("results/comparison_results.txt", 'w') as f:
        # --- Per Node Displacement ---
        f.write(f"{'x':>8}  {'u_ff':>10}  {'u_sofa':>10}\n")
        for x, uff, us in zip(x_ff, u_ff, u_sofa):
            f.write(f"{x:8.4f}  {uff:10.6f}  {us:10.6f}\n")
        # --- L2 Norm ---
        norm = np.linalg.norm(u_ff - u_sofa) / x_ff.size
        f.write(f"L2 Norm - ||u_ff - u_sofa||_2\n")
        f.write(f"{norm:>8}\n")

