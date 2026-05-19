# 1D MMS — refactor plan

## Problem

`build_bar_scene` (in `bar.py`) uses `RegularGridTopology` to define the mesh,
but the body-force vector for `ConstantForceField BodyForce` was being
assembled from a *separate* Python-side mesh (`np.linspace(0,1,nx)` and
consecutive edge pairs). Two parallel mesh representations — they happen to
agree, but nothing enforces it. The MMS verification could silently miss a
topology mismatch.

## Current state (intermediate)

`build_bar_scene` now creates `BodyForce` with zero forces. `solve_bar`
performs the assembly *after* `Sofa.Simulation.init`, reading nodes and edges
back from the SOFA topology (`Bar.dofs.rest_position`, `Bar.topology.edges`)
and writing the result into `Bar.BodyForce.forces` before animating.

**Limitation:** the assembly lives in `solve_bar`, not in the scene
construction path. Any consumer that calls `build_bar_scene` directly — e.g.
`runSofa` loading the file via a `createScene` entry point — would run with
zero body force.

## Next step

Move the post-init assembly into a small `Sofa.Core.Controller` attached
inside `build_bar_scene`, so it runs regardless of how the scene is driven.

Sketch:

- Add a `BodyForceAssembler(Sofa.Core.Controller)` defined in `bar.py`.
  - Constructor takes the `Bar` node (or direct refs to `dofs`, `topology`,
    `BodyForce`), the source function `f_body`, and the quadrature rule.
  - On `onSimulationInitDoneEvent` (preferred) — or on the first
    `onAnimateBeginEvent` with a one-shot guard — read `rest_position` and
    `edges`, call `assemble_nodal_forces`, and write into
    `BodyForce.forces` via `writeableArray()`.
- Attach the controller inside `build_bar_scene` after the `BodyForce`
  object is created.
- Drop the manual assembly block from `solve_bar`; it only needs to init,
  animate, and snapshot the result.

## Acceptance

- `build_bar_scene` is self-contained: loading the scene via `runSofa`
  yields a correctly loaded body force without any external driver code.
- `solve_bar` reduces to scene lifecycle + snapshot.
- Convergence rates from `run_convergence.py` are unchanged (sanity check).
