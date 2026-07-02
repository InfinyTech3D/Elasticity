"""Diagnostic : convergence P1 tet avec Dirichlet complet (isole le Neumann)."""

from convergence import run_convergence_series
from output      import plot_convergence
from solid       import load_params, RESULTS_DIR, element_tet
from sinusoidal  import mms as sinusoidal_mms
from sinusoidal_dirichlet import solve_dirichlet


if __name__ == "__main__":
    cfg  = load_params()
    L, E = cfg["length"], cfg["youngModulus"]
    ff   = cfg["forceField"]
    ls   = cfg["linearSolver"]
    conv = cfg["convergence"]

    nx_vals = conv["nx_values"][sinusoidal_mms.name]
    nu      = conv["nu_values"][0]

    hs, errors = run_convergence_series(
        nx_values = nx_vals,
        run_fn    = lambda nx: solve_dirichlet(
            element_tet, sinusoidal_mms, L, E, nu, nx, nx, nx,
            force_field=ff, linear_solver=ls),
        h_fn      = lambda nx: L / (nx - 1),
        error_fns = {
            "L2": lambda sol: element_tet.compute_l2(sol, sinusoidal_mms, L),
            "H1": lambda sol: element_tet.compute_h1(sol, sinusoidal_mms, L),
        },
        banner    = f"-- P1 tet DIRICHLET-only  {sinusoidal_mms.name}  nu={nu} --",
        results_dir = RESULTS_DIR,
        table_stem  = f"convergence_{sinusoidal_mms.name}_P1tet_DIRICHLET_nu{nu}",
    )

    plot_convergence(
        f"convergence_{sinusoidal_mms.name}_DIRICHLET_nu{nu}",
        hs, [{"label": "P1 tet L² (Dirichlet)", "errors": errors["L2"], "style": "g^-"},
             {"label": "P1 tet H¹ (Dirichlet)", "errors": errors["H1"], "style": "md--"}],
        title=f"Convergence — {sinusoidal_mms.name} DIRICHLET nu={nu}",
        results_dir=RESULTS_DIR)