import pytest
import torch
import torch.nn as nn

# Import library modules
from neurom.quadratures import MidPoint1D, TwoPoints1D
from neurom.shape_functions import LinearSegment
from neurom.geometry import IsoparametricMapping1D
from neurom.meshes import Mesh, Topology
from neurom.fields import Field, TrainableField
from neurom.constraints import NoConstraint, Dirichlet
from neurom.field_layout import FieldLayout
from neurom.interpolation import (
    PointWiseInterpolator,
    Interpolator,
    FieldInterpolator,
    QuadratureInterpolationResult,
)

from neurom.physics import ElasticEnergy, LoadPotential
from neurom.physics_loss import PhysicsLoss
from neurom.fem_model import FEMModel

torch.set_default_dtype(torch.float32)


class AnalyticalSolution:
    """
    Analytical solution to the 1D beam displacement field problem.
    .. math::
        \frac{1}{2} f (x - x_\text{min}) (x - x_\text{max})
    """

    def __init__(self, f, x_min, x_max):
        self.f = f
        self.x_min = x_min
        self.x_max = x_max

    def eval(self, x):
        return 0.5 * self.f(x) * (x - self.x_min) * (x - self.x_max)


class Test1dBeamDeflection:
    """
    Test training on a 1D beam under a uniform load with both ends fixed.

    Attributes:
        relative_tolerance (float): The relative tolerance used to compare floats.
    """

    relative_tolerance: float = 1e-2

    def test_beam_mid_point_quadrature(self):
        """
        Test that the displacement field corresponds to the analytical one.

        Use quadratures.TwoPoints1D.
        """
        # --- Prepare parameters ---
        # Domain dimensions
        x_min = 0.0
        x_max = 10.0
        # Number of points in the domain
        N = 100

        # Load applied to the beam
        def f(x):
            return 1000.0

        # Number of training steps
        n_epochs = 5000
        # Learning rate
        lr = 10.0

        # Generate vertices and connectivity
        x_array = torch.linspace(x_min, x_max, N).unsqueeze(-1)
        nodes = torch.arange(0, N)
        elements = torch.vstack([torch.arange(0, N - 1), torch.arange(1, N)]).T

        # Initialize displacement values
        u_init = 0.5 * torch.ones(N, 1)

        # Generate topology
        topology = Topology(nodes, elements)

        # Shape function
        sf = LinearSegment()
        # Quadrature strategy
        quad = MidPoint1D()
        # Mapping from/to reference/physical coordinates
        mapping = IsoparametricMapping1D(sf)

        # Prepare Field layout and fill it with actual fields
        field_layout = FieldLayout()

        # Displacement
        u = field_layout.add(
            TrainableField(
                name="displacement",
                topology=topology,
                init_values=u_init,
                constraint=Dirichlet(
                    nodes=[0, N - 1], values_imposed=torch.zeros(2, 1)
                ),
            )
        )

        # Positions
        x = field_layout.add(Field(name="positions", topology=topology, values=x_array))

        # Generate mesh
        mesh = Mesh(topology=topology, nodes_positions=x)

        # Define interpolator
        interpolator = Interpolator(mesh, quad, mapping, [FieldInterpolator(sf, u)])

        # Define physics to solve
        physics = ElasticEnergy(field=u) - LoadPotential(field=u, f=f)

        # The loss to use is purely based on physics
        physics_loss = PhysicsLoss(physics=physics, field_layout=field_layout)

        # Define FEM model
        model = FEMModel(
            mesh=mesh,
            field_layout=field_layout,
            interpolator=interpolator,
            loss=physics_loss,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for i in range(n_epochs):
            loss = model()
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        # Evaluate at quadrature points
        result = field_layout["displacement"]

        # Compute analytical solution
        solution = AnalyticalSolution(f=f, x_min=x_min, x_max=x_max)
        u_sol_train = solution.eval(result.x)

        # Check values
        assert result.u.detach().numpy() == pytest.approx(
            u_sol_train.detach().numpy(), rel=self.relative_tolerance
        )

        # Generate test points and interpolate
        # This also tests the boundary condition
        x_test = torch.linspace(x_min, x_max, 30)
        pwi = PointWiseInterpolator(mesh, sf, u, mapping)
        u_test = pwi.at_position(x_test).squeeze()

        # Compute analytical solution
        u_sol_test = solution.eval(x_test)

        # Check values
        # Note: absolute tolerance because there are 0 at boundaries
        assert u_test.detach().numpy() == pytest.approx(
            u_sol_test, abs=self.relative_tolerance * max(abs(u_sol_test))
        )

    def test_beam_two_points_quadrature(self):
        """
        Test that the displacement field corresponds to the analytical one.

        Use quadratures.TwoPoints1D.
        """
        # --- Prepare parameters ---
        # Domain dimensions
        x_min = 0.0
        x_max = 10.0
        # Number of points in the domain
        N = 100

        # Load applied to the beam
        def f(x):
            return 1000.0

        # Number of training steps
        n_epochs = 5000
        # Learning rate
        lr = 10.0

        # Generate vertices and connectivity
        x_array = torch.linspace(x_min, x_max, N).unsqueeze(-1)
        nodes = torch.arange(0, N)
        elements = torch.vstack([torch.arange(0, N - 1), torch.arange(1, N)]).T

        # Initialize displacement values
        u_init = 0.5 * torch.ones(N, 1)

        # Generate topology
        topology = Topology(nodes, elements)

        # Shape function
        sf = LinearSegment()
        # Quadrature strategy
        quad = TwoPoints1D()
        # Mapping from/to reference/physical coordinates
        mapping = IsoparametricMapping1D(sf)

        # Prepare Field layout and fill it with actual fields
        field_layout = FieldLayout()

        # Displacement
        u = field_layout.add(
            TrainableField(
                name="displacement",
                topology=topology,
                init_values=u_init,
                constraint=Dirichlet(
                    nodes=[0, N - 1], values_imposed=torch.zeros(2, 1)
                ),
            )
        )

        # Positions
        x = field_layout.add(Field(name="positions", topology=topology, values=x_array))

        # Generate mesh
        mesh = Mesh(topology=topology, nodes_positions=x)

        # Define interpolator
        interpolator = Interpolator(mesh, quad, mapping, [FieldInterpolator(sf, u)])

        # Define physics to solve
        physics = ElasticEnergy(field=u) - LoadPotential(field=u, f=f)

        # The loss to use is purely based on physics
        physics_loss = PhysicsLoss(physics=physics, field_layout=field_layout)

        # Define FEM model
        model = FEMModel(
            mesh=mesh,
            field_layout=field_layout,
            interpolator=interpolator,
            loss=physics_loss,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for i in range(n_epochs):
            loss = model()
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        # Evaluate at quadrature points
        result = field_layout["displacement"]

        # Compute analytical solution
        solution = AnalyticalSolution(f=f, x_min=x_min, x_max=x_max)
        u_sol_train = solution.eval(result.x)

        # Check values
        assert result.u.detach().numpy() == pytest.approx(
            u_sol_train.detach().numpy(), rel=self.relative_tolerance
        )

        # Generate test points and interpolate
        # This also tests the boundary condition
        x_test = torch.linspace(x_min, x_max, 30)
        pwi = PointWiseInterpolator(mesh, sf, u, mapping)
        u_test = pwi.at_position(x_test).squeeze()

        # Compute analytical solution
        u_sol_test = solution.eval(x_test)

        # Check values
        # Note: absolute tolerance because there are 0 at boundaries
        assert u_test.detach().numpy() == pytest.approx(
            u_sol_test, abs=self.relative_tolerance * max(abs(u_sol_test))
        )
