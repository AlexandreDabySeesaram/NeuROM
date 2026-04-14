import torch
import torch.nn as nn

from neurom.quadratures.reference_coordinates import reference_coordinates
from neurom.quadratures.quadrature_rule import QuadratureRule
from neurom.interpolation.quadrature_positions import QuadraturePositions
from neurom.meshes.mesh import Mesh

from neurom.samplings import QuadratureSampling


class QuadratureContext(nn.Module):
    """Provides context of interpolation at quadrature points

    Provides physical positions and positions on reference element.
    The mapping back from physical to reference space is necessary because in order to later compute autograd on function u(x), there has to be a direct link between ``x`` and ``u``.

    Args:
        mesh (Mesh): The mesh on which to interpolate.
        mapping (object): The mapping to use to translate positions from physical to reference space.
        quad (QuadratureRule): The QuadratureRule to use for setting the points.

    Attributes:
        _mesh (Mesh): The mesh on which to interpolate.
        _mapping (object): The mapping to use to translate positions from physical to reference space.
        _quad (QuadratureRule): The QuadratureRule to use for setting the points.
        _xi_ref (torch.Tensor): The positions in reference space duplicated over the number of elements in the mesh.
        _quad_pos (QuadraturePositions): The quadrature positions at physical, reference, and back from physical to reference.
        _measure (torch.Tensor): The measure for the all elements and quadrature points, weight * |det_J|. Tensor of shape (N_e,N_q,1).
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
        """Compute all geometry-dependent quantities.

        Re-compute measure and quadrature positions.
        """
        self._compute_measure()
        self._compute_quad_pos()

    def _compute_quad_pos(self) -> None:
        """Interpolate the positions on the mesh

        Interpolates the positions from reference to physical positions and then back from physcial to reference.
        Set self._quad_pos with interpolated positions.
        """
        # map to physical space for all quadrature points
        # Tensor of shape (N_e, N_q, dim)
        x_phys = self._mapping.map(self._xi_ref)

        # back‑to‑reference (needed for autograd‑safe field interpolation)
        xi_back = self._mapping.inverse_map(x_phys)

        self._quad_pos = QuadraturePositions(
            xi_ref=QuadratureSampling(self._xi_ref),
            x_phys=QuadratureSampling(x_phys),
            xi_back=QuadratureSampling(xi_back),
        )

    def _compute_measure(self) -> torch.Tensor:
        """Helper method to compute the 'measure'

        Computes the product of the determinant of the jacobian from physical to reference coordinates mapping times the quadrature weights. Tensor of shape (N_e, N_q, 1).
        Set self._measure with computed measure.
        """
        # Compute weighted measure
        w = self._quad.weights()
        dx = self._mapping.det_jacobian
        m = torch.abs(dx) * w
        n_e = dx.shape[0]
        n_q = w.shape[0]

        self._measure = QuadratureSampling(m.reshape(n_e, n_q, 1))

    @property
    def measure(self) -> torch.Tensor:
        """Compute measure

        Returns:
            (torch.Tensor) of shape (N_e,N_q,1) representing weight * |det_J|.
        """
        return self._measure

    @property
    def interpolate(self) -> QuadraturePositions:
        """Interpolate positions at quadrature points

        Returns:
            (QuadraturePositions) with all positions at physical, reference and back from physical to reference.
        """
        return self._quad_pos

    def update(self):
        self._mapping.update()
        self._setup()
