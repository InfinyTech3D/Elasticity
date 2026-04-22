# Solving 1D Elasticity Problems Using the Method of Manufactured Solutions (MMS)

2026-04-22

## Introduction and Objective

This document presents the application of the **Method of Manufactured
Solutions (MMS)** to the validation of a finite element code for linear
elasticity problems in one dimension. The objective is to verify that
the numerical implementation (here in the SOFA software) produces
results consistent with theoretical expectations, in particular
regarding convergence rates.

### General Problem

We consider an elastic bar of length *L*, Young’s modulus *E*, subjected
to a body force *f*(*x*). The local equilibrium equation reads:

*E* *u*<sup>″</sup>(*x*) + *f*(*x*) = 0  ∀*x* ∈ \[0, *L*\]

where *u*(*x*) is the axial displacement. This equation is complemented
by boundary conditions (Dirichlet and/or Neumann).

The MMS consists of choosing an arbitrary exact solution
*u*<sub>ex</sub>(*x*), computing the corresponding right-hand side
*f*(*x*) by substitution into the equilibrium equation, and then
imposing consistent boundary conditions. The numerical solution is then
compared to the exact solution to evaluate the error and verify the
order of convergence.

------------------------------------------------------------------------

## Case 1: Sinusoidal Solution

### Problem Statement

We choose a sinusoidal exact solution:

*u*<sub>ex</sub>(*x*) = sin (*π**x*)

For a bar of length *L* = 1, the Young’s modulus is set to
*E* = 10<sup>6</sup>. Substituting this solution into the equilibrium
equation gives the body force:

*f*(*x*) = *E* *π*<sup>2</sup>sin (*π**x*)

The boundary conditions are:

- Zero displacement at the clamped end: *u*(0) = 0 (Dirichlet)
- Prescribed normal force at the free end:
  *E* *u*<sup>′</sup>(*L*) = *E* *π*cos (*π**L*) (Neumann)

### Numerical Implementation

The code builds a mesh using linear P1 finite elements. The body force
is integrated using **2-point Gauss quadrature** per element, which is
exact for the sinusoidal function. The nonlinear solver (Newton-Raphson)
is limited to 15 iterations, but since the problem is linear, a single
iteration is sufficient.

### Evaluated Quantities

Two types of error are computed:

- **Nodal L² error**: evaluated at the nodes, it exhibits
  superconvergence (order 2)
- **L² error at element midpoints**: evaluated by linear interpolation,
  it gives the actual convergence order
- **H¹ semi-norm error**: computed by Gauss quadrature at integration
  points, used to verify the convergence of the strain field

### Results and Convergence

The convergence study for mesh sizes ranging from 10 to 160 nodes shows:

- The L² error (midpoints) decays as *O*(*h*<sup>2</sup>)
- The H¹ semi-norm error decays as *O*(*h*<sup>1</sup>)

These orders are consistent with P1 finite element theory.

------------------------------------------------------------------------

## Case 2: Quadratic Solution

### Problem Statement

We now choose a quadratic exact solution:

$$
u\_{\text{ex}}(x) = \frac{x(L-x)}{L^2}
$$

For a bar of length *L* = 10 and *E* = 10<sup>6</sup>. The resulting
body force is constant:

$$
f(x) = \frac{2E}{L^2}
$$

The boundary conditions are:

- Zero displacement at the left end: *u*(0) = 0
- Prescribed strain at the right end: *u*<sup>′</sup>(*L*) = −1/*L*,
  i.e., a normal force *F*<sub>*N*</sub> = −*E*/*L*

### Implementation

The constant body force is integrated analytically: it translates into
nodal forces equal to *f* ⋅ *h* for interior nodes and *f* ⋅ *h*/2 at
the endpoints. The Neumann condition is applied directly as a point
force at the right end.

### Results and Convergence

The convergence study confirms:

- The nodal L² error is zero (perfect superconvergence, since the exact
  solution is quadratic and the nodes are exact)
- The L² error at element midpoints decays as *O*(*h*<sup>2</sup>)

The interpolation error at the midpoint of an element is proportional to
the curvature of the solution.

------------------------------------------------------------------------

## Case 3: Cubic Solution

### Choice of the Manufactured Solution

We select a cubic displacement field:

$$
u\_{\text{ex}}(x) = \frac{x^2 (L - x)}{L^2}, \qquad x \in \[0, L\]
$$

This function satisfies the homogeneous Dirichlet condition at *x* = 0:

$$
u\_{\text{ex}}(0) = \frac{0^2 (L - 0)}{L^2} = 0
$$

### First Derivative (Strain)

The strain field is obtained by differentiation:

$$
u\_{\text{ex}}'(x) = \frac{d}{dx} \left( \frac{x^2 L - x^3}{L^2} \right) = \frac{2x L - 3x^2}{L^2}
$$

**Check at the right boundary** (*x* = *L*):

$$
u\_{\text{ex}}'(L) = \frac{2L^2 - 3L^2}{L^2} = -1
$$

This gives the Neumann boundary condition.

### Second Derivative (Curvature)

$$
u\_{\text{ex}}''(x) = \frac{2L - 6x}{L^2}
$$

### Body Force via the Equilibrium Equation

The 1D equilibrium equation reads:

$$
E \frac{d^2 u}{dx^2} + f(x) = 0 \quad \forall x \in (0, L)
$$

Rearranging and substituting the second derivative:

$$
f(x) = -E \\ u\_{\text{ex}}''(x) = \frac{E (6x - 2L)}{L^2}
$$

**Properties of *f*(*x*):**

- Linear function in *x*
- At *x* = 0: $f(0) = -\dfrac{2E}{L}$ (negative, force to the left)
- At *x* = *L*: $f(L) = \dfrac{4E}{L}$ (positive, force to the right)
- Zero crossing at *x* = *L*/3

### Boundary Conditions

**Dirichlet (left end):** *u*(0) = 0, imposed as a
`FixedProjectiveConstraint` at node 0.

**Neumann (right end):** For a unit cross-section (*A* = 1):

*F*<sub>*N*</sub>(*L*) = *E* ⋅ *u*<sup>′</sup>(*L*) = *E* ⋅ (−1) = −*E*

The negative sign indicates a compressive force (pointing to the left).

### Weak Formulation

Multiplying the strong form by a test function *v*(*x*) with *v*(0) = 0
and integrating by parts:

∫<sub>0</sub><sup>*L*</sup>*E* *u*<sup>′</sup>(*x*) *v*<sup>′</sup>(*x*) *d**x* = ∫<sub>0</sub><sup>*L*</sup>*f*(*x*) *v*(*x*) *d**x* − *E* *v*(*L*)

This is the weak form implemented in the finite element code.

### Finite Element Discretization (P1 Elements)

#### Mesh and Shape Functions

The interval \[0, *L*\] is divided into *N* elements with
*n*<sub>*x*</sub> = *N* + 1 nodes. Each element *e* spans
\[*x*<sub>*i*</sub>, *x*<sub>*i* + 1</sub>\] of length *h* = *L*/*N*.
The linear shape functions are:

$$
\phi_i^{(e)}(x) = \frac{x\_{i+1} - x}{h}, \quad \phi\_{i+1}^{(e)}(x) = \frac{x - x_i}{h}
$$

#### Element Stiffness Matrix

Since *ϕ*<sub>*i*</sub><sup>′</sup> = −1/*h* and
*ϕ*<sub>*i* + 1</sub><sup>′</sup> = 1/*h*:

$$
\mathbf{k}^{(e)} = \frac{E}{h} \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}
$$

#### Consistent Nodal Forces (Body Force)

The element force vector is:

$$
\mathbf{f}\_{\text{body}}^{(e)} = \int\_{x_i}^{x\_{i+1}} f(x) \begin{bmatrix} \phi_i(x) \\ \phi\_{i+1}(x) \end{bmatrix} dx
$$

Since *f*(*x*) is linear, we interpolate it as
*f*(*x*) ≈ *f*(*x*<sub>*i*</sub>)*ϕ*<sub>*i*</sub> + *f*(*x*<sub>*i* + 1</sub>)*ϕ*<sub>*i* + 1</sub>
and use the exact integrals:

$$
\int\_{x_i}^{x\_{i+1}} \phi_i^2 \\ dx = \frac{h}{3}, \qquad \int\_{x_i}^{x\_{i+1}} \phi_i \phi\_{i+1} \\ dx = \frac{h}{6}
$$

This gives the **consistent nodal forces**:

$$
\mathbf{f}\_{\text{body}}^{(e)} = \frac{h}{6} \begin{bmatrix} 2 f(x_i) + f(x\_{i+1}) \\ f(x_i) + 2 f(x\_{i+1}) \end{bmatrix}
$$

#### Assembly of the Global Force Vector

After assembly, for an interior node *i* (1 ≤ *i* ≤ *N* − 1):

$$
F_i = \frac{h}{6} \left\[ f(x\_{i-1}) + 4f(x_i) + f(x\_{i+1}) \right\]
$$

Substituting $f(x_j) = \dfrac{E(6x_j - 2L)}{L^2}$ and simplifying, this
reduces to:

*F*<sub>*i*</sub> = *h* ⋅ *f*(*x*<sub>*i*</sub>)

**Conclusion:** for a linear *f*(*x*), the consistent nodal forces
simplify to *F*<sub>*i*</sub> = *h* ⋅ *f*(*x*<sub>*i*</sub>), which is
exactly the **nodal quadrature** used in the code.

The full discrete system reads:

**K****u** = **F**<sub>body</sub> + **F**<sub>Neumann</sub>

with boundary contributions:

$$
F_0 = \frac{h}{2} f(x_0), \quad F_i = h \\ f(x_i) \quad (1 \le i \le N-1), \quad F_N = \frac{h}{2} f(x_N) - E
$$

### Results and Convergence

For P1 elements with a linear body force, the finite element solution
satisfies
*u*<sub>*i*</sub><sup>FEM</sup> = *u*<sub>ex</sub>(*x*<sub>*i*</sub>) at
every node — this is the **superconvergence** phenomenon. The
convergence study (meshes from 2 to 64 nodes) confirms:

- L² error at midpoints: order 2
- Nodal L² error: significantly lower, illustrating superconvergence

------------------------------------------------------------------------

## Summary of Results

The three validation cases cover different loading types and exact
solutions:

<table>
<colgroup>
<col style="width: 8%" />
<col style="width: 22%" />
<col style="width: 20%" />
<col style="width: 32%" />
<col style="width: 14%" />
</colgroup>
<thead>
<tr>
<th>Case</th>
<th>Exact solution</th>
<th>Body force</th>
<th>L² order (midpoints)</th>
<th>H¹ order</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>Sinusoidal</td>
<td>Sinusoidal</td>
<td>2.0</td>
<td>1.0</td>
</tr>
<tr>
<td>2</td>
<td>Quadratic</td>
<td>Constant</td>
<td>2.0</td>
<td>—</td>
</tr>
<tr>
<td>3</td>
<td>Cubic</td>
<td>Affine</td>
<td>2.0</td>
<td>—</td>
</tr>
</tbody>
</table>

In all cases, the observed convergence order for the L² error evaluated
at element midpoints is consistent with P1 finite element theory:
*O*(*h*<sup>2</sup>). The H¹ semi-norm error (computed for the
sinusoidal case) yields *O*(*h*<sup>1</sup>), validating the correct
implementation of the strain field (gradient).

Superconvergence at the nodes is also verified: for the quadratic and
cubic solutions, the nodal error is significantly lower than the
midpoint error, and for the sinusoidal case, the nodal error remains
below the midpoint error.

------------------------------------------------------------------------

## Conclusion

The Method of Manufactured Solutions has enabled the quantitative
validation of the P1 finite element implementation in SOFA for 1D
elasticity problems. The theoretical convergence rates are perfectly
recovered, confirming the absence of major programming errors. This
approach can be extended to 2D or 3D geometries and more complex
constitutive laws.

The developed codes automatically produce convergence plots and error
tables, facilitating systematic validation campaigns.
