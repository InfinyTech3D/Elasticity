"""Manufactured fields, and the solution one makes once it is paired with a material."""

from abc import ABC, abstractmethod

import numpy as np
import sympy as sp

from VnV.verification.materials import compile_laws

_COORDINATES = sp.symbols("x y z")

_COMPONENT_NAMES = ("u_x", "u_y", "u_z")


class ManufacturedField(ABC):
    """An exact displacement field, declared symbolically, and where its BCs are applied.

    A field is stated on its own and knows nothing about the material it will be a solution of: the
    two are independent axes of a study, and pairing them is `ManufacturedSolution`'s job.
    """

    prescribe_displacement_on = {}    # {region: direction mask} where u is prescribed (fixed comps)
    traction_on = ()                  # regions where the derived traction is applied

    def __init__(self, spec, geometry):
        # The deck's "function" block, kept whole: what a field needs beyond an amplitude is that
        # field's own business, so a new parameter is a deck key rather than a constructor argument.
        self.spec = spec
        # Scale of the field, and with it how far the mesh moves for a given h: a deck asking for an
        # amplitude near its own extents deforms elements past inversion.
        self.amplitude = spec["amplitude"]
        # Read off the geometry rather than kept as a reference to it: these two are all a field
        # needs, and a stored geometry is a wider surface than that.
        self.extents = geometry.named_extents
        self.dim = geometry.dim

    @abstractmethod
    def displacement(self, coordinates):
        """Exact displacement as sympy expressions, one per component of the field's own dimension."""

    def wavenumber(self, extent):
        """2 pi / one of the geometry's named extents, the wavenumber of a full period across it.

        Exact rather than the deck's float, so pi reaches `equation` as pi: a field printing 6.2832
        where it means 2 pi hides the wave it describes, and recovering the multiple afterwards is
        guesswork over what is known here.
        """
        return 2 * sp.pi / sp.Rational(str(self.extents[extent]))


class ManufacturedSolution:
    """A field and the material it solves: the gradient, stress, source and energies all derived.

    Nothing here is stated twice. The displacement comes from the field and the strain energy from the
    material, and every other quantity is differentiated out of one or the other, so no two of them can
    disagree. The source is -div of the stress in the mesh coordinates, which reads as -div(sigma) for
    a small-strain material and -Div(P) for a hyperelastic one without a branch: both are the same
    derivative of the same dpsi/d(grad u).
    """

    def __init__(self, field, material, spatial_dimensions):
        self.field = field
        self.material = material
        # The embedding space belongs to neither the field nor the material: a field states itself in
        # its own dimension and a material is a law, so the solution is where the two meet a mesh.
        dimensions = spatial_dimensions

        self.coordinates = sp.Matrix(_COORDINATES[:dimensions])
        # A field is handed its own coordinates and states its own components, so it can neither vary
        # nor displace off-manifold, and the embedding pads the rest with zeros. An extra component is
        # rejected rather than padded over: the scene clamps the off-manifold directions to zero, so
        # the study would go on to measure a field the solve is not solving.
        components = list(field.displacement(_COORDINATES[:field.dim]))
        if len(components) != field.dim:
            raise ValueError(f"{type(field).__name__}: displacement must state {field.dim} "
                             f"components, one per dimension of the field, got {len(components)}")
        components += [0] * (dimensions - field.dim)
        # Kept symbolic as well as compiled: `equation` is derived from this, so what a figure or a
        # write-up states is the field that was solved rather than a second, hand-written copy of it.
        self.displacement_expression = displacement = sp.Matrix(components)
        gradient = displacement.jacobian(self.coordinates)
        stress = material.stress(gradient)
        source = -sp.Matrix([sum(sp.diff(stress[i, j], self.coordinates[j])
                                 for j in range(dimensions))
                             for i in range(dimensions)])

        self.u = self._over_points(displacement, (dimensions,))
        self.grad_u = self._over_points(gradient, (dimensions, dimensions))
        self.stress = self._over_points(stress, (dimensions, dimensions))
        self.source = self._over_points(source, (dimensions,))
        self.energy_density, self.tangent = compile_laws(material, dimensions)

    # Where the BCs go is the field's statement, forwarded so the scene holds one object.
    @property
    def prescribe_displacement_on(self):
        return self.field.prescribe_displacement_on

    @property
    def traction_on(self):
        return self.field.traction_on

    @property
    def equation(self):
        """The manufactured displacement in math form, one line per component.

        Inline LaTeX, which is also what matplotlib's mathtext reads, so the same string titles a
        figure and drops into a write-up. It names the field a run verified, which is what a reader of
        either wants -- a deck path says which file was run, not which problem was solved.
        """
        names = ("u",) if len(self.displacement_expression) == 1 else _COMPONENT_NAMES
        return "\n".join(f"${name} = {sp.latex(component)}$"
                         for name, component in zip(names, self.displacement_expression))

    def _over_points(self, expression, shape):
        """Compile an expression of the coordinates into f(points) -> ndarray of (*batch, *shape).

        `points` is (..., spatial_dimensions), so one point or a whole mesh of quadrature points go
        through the same callable. Each component is lambdified on its own rather than the matrix as
        a whole: a component that does not depend on the coordinates lambdifies to a *scalar*, and
        stacking scalars with array-valued siblings is exactly where the obvious version breaks.
        """
        components = [sp.lambdify(list(self.coordinates), entry, "numpy") for entry in expression]

        def evaluate(points):
            points = np.asarray(points, dtype=float)
            batch = points.shape[:-1]
            columns = [points[..., d] for d in range(len(self.coordinates))]
            values = np.empty(batch + shape, dtype=float)
            for index, component in zip(np.ndindex(*shape), components):
                values[(...,) + index] = np.broadcast_to(
                    np.asarray(component(*columns), dtype=float), batch)
            return values

        return evaluate


# --- The fields themselves, one class per deck "function"; registry.py keys them by (dim, name). ---

# In-plane direction masks; the scene pads them to the embedding space (see MMSScene.apply_bcs).
_IN_PLANE_MASKS = {"left": [1, 0], "right": [1, 0], "bottom": [0, 1], "top": [0, 1]}


class Quadratic1D(ManufacturedField):
    """u(x) = [A x^2]: body force constant, so the nodal source is the source; clamped at 'left'."""

    prescribe_displacement_on = {"left": [1]}
    traction_on = ("right",)

    def displacement(self, coordinates):
        return [self.amplitude * coordinates[0] ** 2]


class Trigonometric1D(ManufacturedField):
    """u(x) = [A sin(k x)], k = 2 pi / L: prescribed (clamped) at 'left', traction at 'right'."""

    prescribe_displacement_on = {"left": [1]}
    traction_on = ("right",)

    def displacement(self, coordinates):
        return [self.amplitude * sp.sin(self.wavenumber("length") * coordinates[0])]


class Quadratic2D(ManufacturedField):
    """u = A[x^2, y^2]: body force constant and traction linear, so both are nodally exact."""

    # Each face prescribes only the component that is constant on it -- u_x on the x-faces is
    # A x^2 at fixed x -- so the nodal values carry the Dirichlet data without error either.
    prescribe_displacement_on = _IN_PLANE_MASKS
    traction_on = ("left", "right", "bottom", "top")

    def displacement(self, coordinates):
        x, y = coordinates[0], coordinates[1]
        return [self.amplitude * x**2, self.amplitude * y**2]


class Trigonometric2D(ManufacturedField):
    """u = A[sin(kx x)cos(ky y), cos(kx x)sin(ky y)], kx=2pi/L, ky=2pi/W: ux fixed on x-faces, uy on y-faces, traction elsewhere."""

    prescribe_displacement_on = _IN_PLANE_MASKS
    traction_on = ("left", "right", "bottom", "top")

    def displacement(self, coordinates):
        kx, ky = self.wavenumber("length"), self.wavenumber("width")
        x, y = coordinates[0], coordinates[1]
        return [self.amplitude * sp.sin(kx * x) * sp.cos(ky * y),
                self.amplitude * sp.cos(kx * x) * sp.sin(ky * y)]


class Trigonometric3D(ManufacturedField):
    """u_i oscillates along all axes; each component fixed on its own faces, traction elsewhere."""

    prescribe_displacement_on = {"left": [1, 0, 0], "right": [1, 0, 0],
                                 "bottom": [0, 1, 0], "top": [0, 1, 0],
                                 "front": [0, 0, 1], "back": [0, 0, 1]}
    traction_on = ("left", "right", "bottom", "top", "front", "back")

    def displacement(self, coordinates):
        kx, ky, kz = (self.wavenumber("length"), self.wavenumber("width"),
                      self.wavenumber("height"))
        x, y, z = coordinates
        return [self.amplitude * sp.sin(kx * x) * sp.cos(ky * y) * sp.cos(kz * z),
                self.amplitude * sp.cos(kx * x) * sp.sin(ky * y) * sp.cos(kz * z),
                self.amplitude * sp.cos(kx * x) * sp.cos(ky * y) * sp.sin(kz * z)]
