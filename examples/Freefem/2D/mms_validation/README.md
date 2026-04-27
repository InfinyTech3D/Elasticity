# Verification of 2D Linear Elasticity Using the Method of Manufactured Solutions (MMS)

## Mathematical Problem

### Variables and parameters

- $u_x(x,y)$, $u_y(x,y)$: displacement components
- $x$, $y$: spatial coordinates
- $L$: domain side length
- $E$: Young's modulus
- $f_x(x,y)$, $f_y(x,y)$: body forces per unit area

### 2D linear elasticity (small strain)

$$
\begin{aligned}
\frac{\partial \sigma_{xx}}{\partial x} + \frac{\partial \sigma_{xy}}{\partial y} + f_x(x,y) &= 0 \\
\frac{\partial \sigma_{xy}}{\partial x} + \frac{\partial \sigma_{yy}}{\partial y} + f_y(x,y) &= 0
\end{aligned}
\quad \forall (x,y) \in [0, L] \times [0, L]
$$

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

For a structured mesh with node spacing $\Delta x = L/(n_x-1)$, $\Delta y = L/(n_y-1)$, the integration weights are:

- Interior nodes: $w = \Delta x \cdot \Delta y$
- Edge nodes (not corners): $w = \Delta x \cdot \Delta y / 2$
- Corner nodes: $w = \Delta x \cdot \Delta y / 4$

Thus:

$$
\mathbf{F}_i = \mathbf{f}(x_i, y_i) \cdot w_i
$$

### Neumann Boundary Conditions (Traction)

On boundaries with outward normal $\mathbf{n}$, the traction is $\mathbf{t} = \boldsymbol{\sigma} \cdot \mathbf{n}$. For edges on $y = 0$ or $y = L$, with integration weight $w = \Delta x$ (interior) or $\Delta x/2$ (endpoints). Nodes at $x = 0$ and $x = L$ are Dirichlet and excluded from Neumann assembly:

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

### Source Term Discretization

With a 1-point quadrature rule (trapezoidal rule):

- Interior nodes: $\mathbf{F}_i = \mathbf{f}(x_i, y_i) \cdot \Delta x \,\Delta y$
- Edge nodes (not corners): $\mathbf{F}_i = \mathbf{f}(x_i, y_i) \cdot \Delta x \,\Delta y / 2$
- Corner nodes: $\mathbf{F}_i = \mathbf{f}(x_i, y_i) \cdot \Delta x \,\Delta y / 4$

### Traction Discretization

On $y = \text{constant}$ boundaries (excluding Dirichlet corners at $x=0$, $x=L$), using 1-point quadrature:

- Interior edge nodes: $\mathbf{F}_i = \mathbf{t}(x_i, y_i) \cdot \Delta x$
- Endpoint nodes ($x = 0$ or $x = L$): excluded (Dirichlet boundary)