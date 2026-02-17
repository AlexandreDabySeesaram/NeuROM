from abc import ABC, abstractmethod
import pytest
import torch
import torch.nn as nn

# Import library modules
from neurom.quadratures import MidPoint1D, TwoPoints1D
from neurom.shape_functions import LinearSegment
from neurom.geometry import IsoparametricMapping1D
from neurom.mesh import Mesh
from neurom.field import Field
from neurom.integrator import Integrator
from neurom.evaluator import ElementEvaluator1D
from neurom.fem_model import FEMModel

torch.set_default_dtype(torch.float32)


class PoissonPhysics:
    """
    Represents potential energy of 1D beam under uniform load.
    .. math::
        \frac{1}{2}u''(x) + f u(x)
    """

    def __init__(self, f):
        self.f = f

    def integrand(self, x, u):
        du_dx = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        return 0.5 * du_dx**2 + self.f * u


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
        return 0.5 * self.f * (x - self.x_min) * (x - self.x_max)


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
        """
        # --- Prepare parameters ---
        # Domain dimensions
        x_min = 0.0
        x_max = 10.0
        # Number of points in the domain
        N = 100
        # Load applied to the beam
        f = 1000.0
        # Number of training steps
        n_epochs = 5000
        # Learning rate
        lr = 10.0

        # Generate vertices and connectivity
        nodes = torch.linspace(x_min, x_max, N)[:, None]
        elements = torch.vstack([torch.arange(0, N - 1), torch.arange(1, N)]).T

        # Generate mesh
        mesh = Mesh(nodes, elements)

        # Shape function
        sf = LinearSegment()
        # Quadrature strategy
        quad = MidPoint1D()
        # Mapping from/to reference/physical coordinates
        mapping = IsoparametricMapping1D(sf)
        # Field
        field = Field(mesh, dirichlet_nodes=[0, N - 1])
        # Evaluator
        evaluator = ElementEvaluator1D(mesh, field, sf, quad, mapping)
        # What physics we cnosider
        physics = PoissonPhysics(f)
        # How to integrate the physics on a domain
        integrator = Integrator()

        # Define FEM model - main orchestrator
        model = FEMModel(
            mesh=mesh,
            field=field,
            evaluator=evaluator,
            physics=physics,
            integrator=integrator,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for i in range(n_epochs):
            loss = model()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate at quadrature points
        x_q, u_q, _ = model.evaluator.evaluate()

        # Compute analytical solution
        solution = AnalyticalSolution(f=f, x_min=x_min, x_max=x_max)
        u_sol_train = solution.eval(x_q)

        # Check values
        assert u_q.detach().numpy() == pytest.approx(
            u_sol_train.detach().numpy(), rel=self.relative_tolerance
        )

        # Generate test points and evaluate
        # This also tests the boundary condition
        x_test = torch.linspace(x_min, x_max, 30)
        u_test = model.evaluator.evaluate_at(x_test).squeeze()

        # Compute analytical solution
        u_sol_test = solution.eval(x_test)

        # Check values
        # Note: absolute tolerance because there are 0 at boundaries
        assert u_test.detach().numpy() == pytest.approx(
            u_sol_test, abs=self.relative_tolerance * max(abs(u_sol_test))
        )
