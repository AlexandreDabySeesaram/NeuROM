import torch
import torch.nn as nn

from neurom.quadratures.reference_coordinates import reference_coordinates
from neurom.quadratures.quadrature_rule import QuadratureRule
from neurom.interpolation.quadrature_positions import QuadraturePositions
from neurom.meshes.mesh import Mesh


class QuadratureContext(nn.Module):
    """Provides context of interpolation at quadrature points

    Provides physical positions and positions on reference element.
    The mapping back from physical to reference space is necessary because in order to later compute autograd on function u(x), there has to be a direct link between ``x`` and ``u``.

    Args:
        mesh (Mesh): The mesh on which to interpolate.
        quad (QuadratureRule): The QuadratureRule to use for setting the points.
        mapping (object): The mapping to use to translate positions from physical to reference space.

    Attributes:
        mesh (Mesh): The mesh on which to interpolate.
        mapping (object): The mapping to use to translate positions from physical to reference space.
        xi_ref (torch.Tensor): The positions in reference space duplicated over the number of elements in the mesh.
    """

    def __init__(self, mesh: Mesh, quad: QuadratureRule, mapping):
        super().__init__()
        self._mesh = mesh
        self._mapping = mapping
        self._quad = quad
        self._xi_ref = reference_coordinates(self._mesh.topology.n_elements, self._quad)
        self._xi_ref.requires_grad_(True)
        self._setup()

    def _setup(self):
        """Compute all geometry-dependent quantities."""
        self._compute_measure()
        self._compute_quad_pos()

    def _compute_quad_pos(self) -> QuadraturePositions:
        """Interpolate the positions on the mesh

        Interpolates the positions from reference to physical positions and then back from physcial to reference.

        Returns:
            A dataclass QuadraturePositions which contains the positions.
        """
        # map to physical space for all quadrature points
        # Tensor of shape (N_e, N_q, dim)
        x_phys = self._mapping.map(
            self._xi_ref, self._mesh.nodes_positions.at_elements()
        )

        # back‑to‑reference (needed for autograd‑safe field interpolation)
        xi_back = self._mapping.inverse_map(
            x_phys, self._mesh.nodes_positions.at_elements()
        )

        self._quad_pos = QuadraturePositions(
            xi_ref=self._xi_ref, x_phys=x_phys, xi_back=xi_back
        )

    def _compute_measure(self) -> torch.Tensor:
        """Helper method to compute the 'measure'

        Returns:
            The product of the determinant of the jacobian from physical to reference coordinates mapping times the quadrature weights. Tensor of shape (N_e, N_q).
        """
        # Compute weighted measure
        w = self._quad.weights()
        dx = self._mapping.det_jacobian(self._mesh.nodes_positions.at_elements())
        m = torch.abs(dx) * w
        n_e = dx.shape[0]
        n_q = w.shape[0]

        self._measure = m.reshape(n_e, n_q, 1)

    @property
    def measure(self):
        return self._measure

    @property
    def interpolate(self):
        return self._quad_pos

    def update(self):
        self._interpolate()
        self._measure()
