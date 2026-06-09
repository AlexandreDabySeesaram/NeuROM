"""Per-element quadrature context: reference positions, mapping and integration measure."""

import torch
import torch.nn as nn

from neurom.quadratures.reference_coordinates import reference_coordinates
from neurom.quadratures.quadrature_rule import QuadratureRule
from neurom.interpolation.quadrature_positions import QuadraturePositions
from neurom.meshes.mesh import Mesh

from neurom.samplings import QuadratureSampling


class QuadratureContext(nn.Module):
    """Provides geometric context for interpolation at quadrature points.

    Computes and caches physical positions, reference-element positions, and
    integration measures for all quadrature points on a mesh.  The round-trip
    mapping (physical space -> reference space) is required so that autograd
    can trace the dependency of ``u(x)`` back through ``x``.

    Args:
        mesh (Mesh): The mesh on which to interpolate.
        quad (QuadratureRule): The quadrature rule that defines the reference
            positions and weights.
        mapping: The geometric mapping that provides ``map``,
            ``inverse_map``, and ``det_jacobian`` for translating between
            physical and reference coordinates.

    Attributes:
        _mesh (Mesh): The mesh on which to interpolate.
        _mapping: The geometric mapping used for physical-to-reference
            coordinate translation.
        _quad (QuadratureRule): The quadrature rule defining reference
            positions and weights.
        _xi_ref (torch.Tensor): Reference-space quadrature coordinates
            broadcast over all elements, tensor of shape
            ``(N_e, N_q, dim)``.
        _quad_pos (QuadraturePositions): Cached quadrature positions at
            physical, original reference, and back-mapped reference
            coordinates.
        _measure (QuadratureSampling): Integration measure for every element
            and quadrature point — the product of the quadrature weight and
            the absolute Jacobian determinant ``|det(J)|``, tensor of shape
            ``(N_e, N_q, 1)``.
    """

    def __init__(self, mesh: Mesh, quad: QuadratureRule, mapping):
        super().__init__()
        self._mesh = mesh
        self._mapping = mapping
        self._quad = quad
        self._xi_ref = reference_coordinates(
            self._mesh.connectivity.n_elements, self._quad
        )
        self._xi_ref.requires_grad_(True)
        self._setup()

    def _setup(self):
        """Compute all geometry-dependent quantities.

        Recomputes the integration measure and the quadrature positions.
        Called during construction and again by ``update()`` whenever the
        mesh geometry changes.
        """
        self._compute_measure()
        self._compute_quad_pos()

    def _compute_quad_pos(self) -> None:
        """Compute and cache the quadrature positions on the mesh.

        Maps ``self._xi_ref`` to physical space via ``self._mapping.map``,
        then applies ``self._mapping.inverse_map`` to obtain back-mapped
        reference coordinates. Stores the result in ``self._quad_pos`` as a
        ``QuadraturePositions`` instance.
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

    def _compute_measure(self) -> None:
        """Compute and cache the integration measure for all quadrature points.

        The measure is the product of the quadrature weight and the absolute
        value of the Jacobian determinant ``|det(J)|``.  Stores the result in
        ``self._measure`` as a ``QuadratureSampling`` of shape
        ``(N_e, N_q, 1)``.
        """
        # Compute weighted measure
        w = self._quad.weights()
        dx = self._mapping.det_jacobian
        m = torch.abs(dx) * w
        n_e = dx.shape[0]
        n_q = w.shape[0]

        self._measure = QuadratureSampling(m.reshape(n_e, n_q, 1))

    @property
    def measure(self) -> QuadratureSampling:
        """Return the cached integration measure.

        Returns:
            QuadratureSampling: Integration measure for every element and
            quadrature point — the product of the quadrature weight and
            ``|det(J)|``, tensor of shape ``(N_e, N_q, 1)``.
        """
        return self._measure

    @property
    def interpolate(self) -> QuadraturePositions:
        """Return the cached quadrature positions.

        Returns:
            QuadraturePositions: Positions at all quadrature points in
            reference space (``xi_ref``), physical space (``x_phys``), and
            back-mapped reference space (``xi_back``).
        """
        return self._quad_pos

    def update(self):
        """Refresh the mapping and recompute all geometry-dependent quantities.

        Calls ``self._mapping.update()`` to propagate any changes to the mesh
        node positions into the mapping, then recomputes the measure and
        quadrature positions via ``_setup()``.  Call this at the start of each
        forward pass when mesh nodes are trainable.
        """
        self._mapping.update()
        self._setup()
