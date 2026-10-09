import pytest
import torch

# Import library modules
from neurom.quadratures import TwoPoints1D, QuadratureRule
from neurom.reference_elements.bar import Bar
from neurom.shape_functions import LinearBar
from neurom.meshes import Mesh, Topology
from neurom.function_space import FunctionSpace
from neurom.fields import Field
from neurom.field_layout import FieldLayout
from neurom.interpolation import (
    FieldInterpolator,
    QuadratureContext,
    QuadratureAssembly,
    IntegrationDomain,
)
from neurom.physics import SolidElasticEnergy
from neurom.math import jacobian, second_derivative
from neurom.elements import P1_BAR, HERMITE, VectorElement

torch.set_default_dtype(torch.float32)


class Endpoints(QuadratureRule):
    """
    Quadrature points at the two vertices of the element, to probe nodal values.
    """

    def __init__(self):
        super().__init__(Bar())
        self.register_buffer(
            "points_barycentric", torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        )
        self.register_buffer("weights_ref", torch.tensor([1.0, 1.0]))


def _mesh(x_nodes):
    """
    Build a 1-D mesh on the given node positions (P1 geometry).
    """
    n = len(x_nodes)
    elements = torch.vstack([torch.arange(n - 1), torch.arange(1, n)]).T
    topology = Topology(elements)
    geometry = FunctionSpace(topology, VectorElement(P1_BAR, 1))
    return Mesh(topology, Field(geometry, x_nodes.unsqueeze(-1)))


def _cubic_hermite_layout(quad):
    """
    Hermite field w = x^3 with physical DOFs (w, w') on the non-uniform mesh [0, 0.25, 1].
    """
    field_layout = FieldLayout()
    x_nodes = torch.tensor([0.0, 0.25, 1.0])
    mesh = _mesh(x_nodes)

    n = len(x_nodes)
    # Hermite DOFs interleaved per vertex as (w, w'); the Hermite element numbers
    # them [w0, w'0, w1, w'1, ...], matching this values tensor.
    values = torch.empty(2 * n, 1)
    values[0::2, 0] = x_nodes**3
    values[1::2, 0] = 3 * x_nodes**2

    w = field_layout.add(Field(FunctionSpace(mesh.topology, HERMITE), values, name="w"))

    ctx = QuadratureContext(mesh, quad)
    assembly = QuadratureAssembly(ctx, w)
    IntegrationDomain([assembly]).interpolate_all(field_layout)
    return field_layout, w


class TestQuadratureAssemblyDofTransformation:
    """
    Physical DOFs are mapped to reference DOFs before interpolation

    Attributes:
        relative_tolerance (float): The relative tolerance used to compare floats.
    """

    relative_tolerance: float = 1e-5

    def test_hermite_slope_is_continuous_on_non_uniform_mesh(self):
        """
        w' of w = x^3 at x = 0.25 is the same from both elements, and exact
        """
        field_layout, _ = _cubic_hermite_layout(Endpoints())
        result = field_layout["w"]

        # (N_e, N_q): N_q = end points of each element
        dw = jacobian(result.x, result.u).values.squeeze()

        assert dw[0, 1].item() == pytest.approx(0.1875, rel=self.relative_tolerance)
        assert dw[1, 0].item() == pytest.approx(0.1875, rel=self.relative_tolerance)

    def test_hermite_bending_energy_on_non_uniform_mesh(self):
        """
        1/2 int_0^1 (w'')^2 = 6 for w = x^3, which Hermite reproduces exactly
        """
        field_layout, w = _cubic_hermite_layout(TwoPoints1D())

        energy = SolidElasticEnergy(
            w, strain=second_derivative, stress_point=lambda k: k
        ).integrand(field_layout)

        assert energy.sum().item() == pytest.approx(6.0, rel=self.relative_tolerance)

    def test_lagrange_is_unchanged(self):
        """
        Lagrange DOFs go through the identity: same values as the bare interpolator
        """
        field_layout = FieldLayout()
        mesh = _mesh(torch.tensor([0.0, 0.25, 1.0]))
        u = field_layout.add(
            Field(
                FunctionSpace(mesh.topology, P1_BAR),
                torch.tensor([[1.0], [-2.0], [3.0]]),
                name="u",
            )
        )
        ctx = QuadratureContext(mesh, TwoPoints1D())

        u_q = QuadratureAssembly(ctx, u).interpolate().u.values
        u_q_bare = FieldInterpolator(LinearBar(), u).at_reference(
            ctx.interpolate.xi_back.values
        )

        assert torch.equal(u_q, u_q_bare)
