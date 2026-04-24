# Verification of 2D Linear Elasticity Using the Method of Manufactured Solutions (MMS)

## Mathematical Problem

### 2D linear elasticity (small strain)

$$
\begin{cases}
\displaystyle \frac{\partial \sigma_{xx}}{\partial x} + \frac{\partial \sigma_{xy}}{\partial y} + f_x(x,y) = 0 \\[10pt]
\displaystyle \frac{\partial \sigma_{xy}}{\partial x} + \frac{\partial \sigma_{yy}}{\partial y} + f_y(x,y) = 0
\end{cases}
\quad \forall (x,y) \in [0, L] \times [0, L]
$$

### Constitutive law (plane stress, $\nu = 0$)

$$
\sigma_{xx} = E \,\varepsilon_{xx}, \quad
\sigma_{yy} = E \,\varepsilon_{yy}, \quad
\sigma_{xy} = 2G \,\varepsilon_{xy} = E \,\varepsilon_{xy}
$$

### Strain-displacement relations

$$
\varepsilon_{xx} = \frac{\partial u_x}{\partial x}, \quad
\varepsilon_{yy} = \frac{\partial u_y}{\partial y}, \quad
\varepsilon_{xy} = \frac{1}{2}\left( \frac{\partial u_x}{\partial y} + \frac{\partial u_y}{\partial x} \right)
$$

### Variables and parameters

- $u_x(x,y)$, $u_y(x,y)$: displacement components
- $x$, $y$: spatial coordinates
- $L$: domain side length
- $E$: Young's modulus
- $f_x(x,y)$, $f_y(x,y)$: body forces per unit area

### Boundary conditions

- Dirichlet conditions on $x = 0$ and $x = L$: $u_x = 0$, $u_y = 0$
- Neumann conditions on $y = 0$ and $y = L$: prescribed traction $\mathbf{t} = \boldsymbol{\sigma} \cdot \mathbf{n}$

---

## Discretization

- Standard Galerkin finite element formulation
- Linear triangular elements (P1)
- Uniform structured mesh on $[0, L] \times [0, L]$, divided into $2 \times (n_x-1) \times (n_y-1)$ triangles

Integrals over each triangle $T$ of area $A_T$ are approximated by:

$$
\int_T g(x,y)\,dA \approx A_T \cdot g(\mathbf{x}_c)
$$

where $\mathbf{x}_c$ is the triangle centroid.

### Body Force

The nodal force vector associated with the body force is defined as:

$$
\mathbf{F}_i = \int_\Omega \mathbf{f}(x,y) \, \phi_i(x,y) \, dA
$$

where $\phi_i(x,y)$ are the finite element shape functions.

For a structured mesh with node spacing $\Delta x = L/(n_x-1)$, $\Delta y = L/(n_y-1)$, the integration weights are:

- Interior nodes: $w = \Delta x \cdot \Delta y$
- Edge nodes (not corners): $w = \Delta x \cdot \Delta y / 2$
- Corner nodes: $w = \Delta x \cdot \Delta y / 4$

Thus:

$$
\mathbf{F}_i = \mathbf{f}(x_i, y_i) \cdot w_i
$$

### Neumann Boundary Conditions (Traction)

On boundaries with outward normal $\mathbf{n}$, the traction is $\mathbf{t} = \boldsymbol{\sigma} \cdot \mathbf{n}$. The nodal force contribution is:

$$
\mathbf{F}_i = \int_{\partial \Omega_N} \mathbf{t}(x,y) \, \phi_i(x,y) \, ds
$$

For edges on $y = 0$ or $y = L$, with integration weight $w = \Delta x$ (interior) or $\Delta x/2$ (endpoints). Note that nodes at $x = 0$ and $x = L$ on these boundaries are subject to Dirichlet conditions and therefore excluded from the Neumann assembly:

$$
\mathbf{F}_i = \mathbf{t}(x_i, y_i) \cdot w_i
$$

### Error Measures

- $\mathbf{u}_h$: finite element solution
- $\mathbf{u}_{ex}$: manufactured solution

#### $L^2$ norm

$$
\| \mathbf{u}_h - \mathbf{u}_{ex} \|_{L^2}^2 \approx \sum_T A_T \, \left\| \mathbf{u}_h(\mathbf{x}_c) - \mathbf{u}_{ex}(\mathbf{x}_c) \right\|^2
$$

evaluated at triangle centroids.

---

## Manufactured Solution

### Displacement field

$$
u_{x,\text{ex}}(x,y) = \frac{x^2 (L - x)}{L^2}
$$

$$
u_{y,\text{ex}}(x,y) = \frac{x (L - x) y}{L^2}
$$

### Source Terms

From equilibrium $\nabla \cdot \boldsymbol{\sigma} + \mathbf{f} = 0$, with $\nu = 0$.

For this manufactured solution, $\partial u_x / \partial y = 0$, so:

$$
\sigma_{xy} = E\,\varepsilon_{xy} = \frac{E}{2}\,\frac{\partial u_y}{\partial x}
$$

The resulting body forces are:

$$
f_x(x,y) = -\frac{\partial \sigma_{xx}}{\partial x} - \frac{\partial \sigma_{xy}}{\partial y}
= -\frac{E(2L - 6x)}{L^2} - \frac{E(L - 2x)}{2L^2}
$$

$$
f_y(x,y) = -\frac{\partial \sigma_{xy}}{\partial x} - \frac{\partial \sigma_{yy}}{\partial y}
= \frac{E y}{L^2}
$$

### Boundary Conditions

#### Dirichlet (imposed displacement)

On $x = 0$ and $x = L$:

$$
u_x = 0, \quad u_y = 0
$$

#### Neumann (imposed traction)

On $y = 0$ (outward normal $\mathbf{n} = (0, -1)^T$):

$$
t_x = -\sigma_{xy}(x, 0), \quad t_y = -\sigma_{yy}(x, 0)
$$

On $y = L$ (outward normal $\mathbf{n} = (0, +1)^T$):

$$
t_x = \sigma_{xy}(x, L), \quad t_y = \sigma_{yy}(x, L)
$$

where:

$$
\sigma_{xx} = E \,\frac{\partial u_x}{\partial x}, \quad
\sigma_{yy} = E \,\frac{\partial u_y}{\partial y}, \quad
\sigma_{xy} = \frac{E}{2} \,\frac{\partial u_y}{\partial x}
$$

### Source Term Discretization

With a 1-point quadrature rule (trapezoidal rule):

- Interior nodes: $\mathbf{F}_i = \mathbf{f}(x_i, y_i) \cdot \Delta x \,\Delta y$
- Edge nodes (not corners): $\mathbf{F}_i = \mathbf{f}(x_i, y_i) \cdot \Delta x \,\Delta y / 2$
- Corner nodes: $\mathbf{F}_i = \mathbf{f}(x_i, y_i) \cdot \Delta x \,\Delta y / 4$

### Traction Discretization

On $y = \text{constant}$ boundaries (excluding Dirichlet corners at $x=0$, $x=L$), using 1-point quadrature:

- Interior edge nodes: $\mathbf{F}_i = \mathbf{t}(x_i, y_i) \cdot \Delta x$
- Endpoint nodes ($x = 0$ or $x = L$): excluded (Dirichlet boundary)