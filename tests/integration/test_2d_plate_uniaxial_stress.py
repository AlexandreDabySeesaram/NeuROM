import math

import pytest
import torch

# Import library modules
from neurom.quadratures import MidPoint2D
from neurom.shape_functions import LinearTriangle
from neurom.geometry import IsoparametricMapping2D
from neurom.meshes import Mesh, Connectivity
from neurom.meshes.validity import is_valid_mesh
from neurom.constraints import Dirichlet
from neurom.fields import Field, TrainableField
from neurom.field_layout import FieldLayout
from neurom.interpolation import (
    QuadratureContext,
    QuadratureAssembly,
    IntegrationDomain,
)
from neurom.physics import SolidElasticEnergy
from neurom.physics.tensors import (
    green_lagrange_strain,
    linear_elastic_stress,
    linear_elastic_stress_point,
    stress_deviator,
    stress_von_mises,
)
from neurom.physics_loss import PhysicsLoss
from neurom.fem_model import FEMModel

torch.set_default_dtype(torch.float32)


def build_plate_mesh(l_x: float, l_y: float, n_x: int, n_y: int):
    """Build a structured triangular mesh of the rectangle ``[0, l_x] x [0, l_y]``.

    Each cell of the ``n_x`` by ``n_y`` grid is split into two
    counter-clockwise triangles, so the resulting mesh is valid by construction.

    Returns:
        A tuple ``(connectivity, points)`` with the
        :class:`~neurom.meshes.connectivity.Connectivity` and the node positions
        of shape ``(n_x * n_y, 2)``.
    """
    xs = torch.linspace(0.0, l_x, n_x)
    ys = torch.linspace(0.0, l_y, n_y)

    points = torch.stack([torch.tensor([x, y]) for y in ys for x in xs])

    def idx(i, j):
        return j * n_x + i

    elements = []
    for j in range(n_y - 1):
        for i in range(n_x - 1):
            n0 = idx(i, j)
            n1 = idx(i + 1, j)
            n2 = idx(i + 1, j + 1)
            n3 = idx(i, j + 1)
            # Two counter-clockwise triangles.
            elements.append([n0, n1, n2])
            elements.append([n0, n2, n3])

    elements = torch.tensor(elements)
    nodes = torch.arange(0, points.shape[0])
    connectivity = Connectivity(nodes, elements)
    return connectivity, points


class Test2dPlateUniaxialStress:
    """Integration test verifying the 2D solid-mechanics pipeline against theory.

    This is a *patch test* for a uniform uniaxial-stress state. A linear
    displacement field corresponding to a uniaxial compression with free lateral
    contraction is prescribed on the whole boundary; the interior nodes are
    solved by energy minimization. Because linear triangles reproduce a linear
    field exactly, the converged solution must be the analytical homogeneous
    field, so the strain, stress tensor and von Mises stress can be checked
    element-by-element against closed-form linear elasticity.

    The constitutive law implemented by the library is the 2D Hooke law

    .. math::
        \\sigma = \\lambda\\,\\mathrm{tr}(\\varepsilon)\\,I + 2\\mu\\,\\varepsilon

    For a uniform strain :math:`\\varepsilon = \\mathrm{diag}(\\alpha, \\beta)` the
    lateral strain that makes the side faces traction-free
    (:math:`\\sigma_{xx} = 0`) is :math:`\\alpha = -\\lambda\\beta/(\\lambda + 2\\mu)`,
    which leaves a pure uniaxial stress :math:`\\sigma_{yy}`.

    Attributes:
        relative_tolerance (float): Relative tolerance used to compare stresses.
    """

    relative_tolerance: float = 1e-3

    def test_uniaxial_stress_patch(self):
        # --- Parameters ---
        l_x, l_y = 4.0, 4.0
        n_x, n_y = 6, 6
        lame_lambda = 1.25
        lame_mu = 1.0
        n_epochs = 10

        # --- Analytical uniform state (uniaxial stress, sigma_xx = 0) ---
        eps_yy = -0.02  # axial (compressive) strain imposed along y
        eps_xx = -lame_lambda * eps_yy / (lame_lambda + 2.0 * lame_mu)

        tr_eps = eps_xx + eps_yy
        sigma_xx = lame_lambda * tr_eps + 2.0 * lame_mu * eps_xx  # == 0 by design
        sigma_yy = lame_lambda * tr_eps + 2.0 * lame_mu * eps_yy

        # Von Mises with the library convention (2D trace/identity, 1/3 factor).
        tr_sigma = sigma_xx + sigma_yy
        sdev_xx = sigma_xx - tr_sigma / 3.0
        sdev_yy = sigma_yy - tr_sigma / 3.0
        von_mises_expected = math.sqrt(1.5 * (sdev_xx**2 + sdev_yy**2))

        # --- Mesh ---
        connectivity, points = build_plate_mesh(l_x, l_y, n_x, n_y)
        N = points.shape[0]

        # --- Prescribe the linear field u = (eps_xx * X, eps_yy * Y) on the
        #     boundary; interior nodes are free (trainable). ---
        x_coord, y_coord = points[:, 0], points[:, 1]
        on_boundary = (
            (x_coord == 0.0) | (x_coord == l_x) | (y_coord == 0.0) | (y_coord == l_y)
        )
        # Masking arange keeps node indices ascending, which is what
        # Dirichlet.expand assumes for its imposed values.
        nodes_bc = connectivity.nodes_indices[on_boundary]
        u_bc = torch.stack(
            [eps_xx * x_coord[on_boundary], eps_yy * y_coord[on_boundary]], dim=1
        )

        # Start the interior nodes away from the answer (only the prescribed
        # boundary values are used for the constrained nodes) so the solver
        # genuinely has to recover the field by energy minimization.
        u_init = torch.zeros(N, 2)

        sf = LinearTriangle()
        quad = MidPoint2D()

        field_layout = FieldLayout()
        u = field_layout.add(
            TrainableField(
                name="displacement",
                connectivity=connectivity,
                init_values=u_init,
                constraint=Dirichlet(nodes=nodes_bc, values_imposed=u_bc),
            )
        )
        x = field_layout.add(
            Field(name="positions", connectivity=connectivity, values=points)
        )

        mesh = Mesh(connectivity=connectivity, nodes_positions=x)
        assert is_valid_mesh(mesh)

        mapping = IsoparametricMapping2D(sf, mesh)
        ctx = QuadratureContext(mesh, quad, mapping)
        assembly_u = QuadratureAssembly(ctx, sf, u)
        domain = IntegrationDomain([assembly_u])

        def stress_point(strain):
            return linear_elastic_stress_point(strain, lame_lambda, lame_mu)

        physics = SolidElasticEnergy(
            field=u,
            strain=green_lagrange_strain,
            stress_point=stress_point,
        )
        physics_loss = PhysicsLoss(physics=physics, field_layout=field_layout)

        model = FEMModel(
            mesh=mesh,
            field_layout=field_layout,
            integration_domain=domain,
            loss=physics_loss,
        )

        optimizer = torch.optim.LBFGS(
            model.parameters(), lr=1.0, max_iter=50, line_search_fn="strong_wolfe"
        )

        def closure():
            optimizer.zero_grad()
            loss = model()
            if loss.requires_grad:
                loss.backward()
            return loss

        for _ in range(n_epochs):
            optimizer.step(closure)

        # --- The solve must reproduce the analytical linear field everywhere ---
        u_full = u.full_values()
        u_analytical = torch.stack([eps_xx * x_coord, eps_yy * y_coord], dim=1)
        assert u_full.detach().numpy() == pytest.approx(u_analytical.numpy(), abs=1e-4)

        # --- Strain / stress at the quadrature points (one per element) ---
        result = assembly_u.interpolate()
        strain = green_lagrange_strain(result.x, result.u)
        sigma = linear_elastic_stress(strain, lame_lambda, lame_mu)
        von_mises = stress_von_mises(stress_deviator(sigma))

        n_elements = connectivity.n_elements
        strain_e = strain.values.mean(dim=1)  # (N_e, 2, 2)
        sigma_e = sigma.values.mean(dim=1)  # (N_e, 2, 2)
        von_mises_e = von_mises.values.mean(dim=1)  # (N_e, 1)

        # Reference homogeneous tensors, broadcast to every element.
        strain_expected = torch.tensor([[eps_xx, 0.0], [0.0, eps_yy]]).expand(
            n_elements, 2, 2
        )
        sigma_expected = torch.tensor([[sigma_xx, 0.0], [0.0, sigma_yy]]).expand(
            n_elements, 2, 2
        )

        scale = abs(sigma_yy)

        # Strain matches diag(eps_xx, eps_yy) in every element.
        assert strain_e.detach().numpy() == pytest.approx(
            strain_expected.numpy(), abs=self.relative_tolerance * abs(eps_yy)
        )

        # Stress tensor matches Hooke's law in every element ...
        assert sigma_e.detach().numpy() == pytest.approx(
            sigma_expected.numpy(), abs=self.relative_tolerance * scale
        )

        # ... in particular the lateral stress vanishes (traction-free sides) ...
        assert sigma_e[:, 0, 0].detach().numpy() == pytest.approx(
            0.0, abs=self.relative_tolerance * scale
        )

        # ... and the axial stress equals the closed-form value.
        assert sigma_e[:, 1, 1].detach().numpy() == pytest.approx(
            sigma_yy, rel=self.relative_tolerance
        )

        # Von Mises stress matches the analytical value in every element.
        assert von_mises_e.squeeze(-1).detach().numpy() == pytest.approx(
            von_mises_expected, rel=self.relative_tolerance
        )
