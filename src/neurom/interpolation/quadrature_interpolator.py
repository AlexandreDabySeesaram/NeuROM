import torch
import torch.nn as nn

from neurom.quadratures.reference_coordinates import reference_coordinates
from neurom.quadratures.quadrature_rule import QuadratureRule
from neurom.interpolation.quadrature_positions import QuadraturePositions
from neurom.meshes.mesh import Mesh


class QuadratureInterpolator(nn.Module):
    """A interpolator of the positions

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
        self.mesh = mesh
        self.mapping = mapping
        self.xi_ref = reference_coordinates(self.mesh.topology.n_elements, quad)
        self.xi_ref.requires_grad_(True)

    def interpolate(self) -> QuadraturePositions:
        """Interpolate the positions on the mesh

        Interpolates the positions from reference to physical positions and then back from physcial to reference.

        Returns:
            A dataclass QuadraturePositions which contains the positions.
        """
        # map to physical space
        x_phys = self.mapping.map(self.xi_ref, self.mesh.nodes_positions.at_elements())

        # back‑to‑reference (needed for autograd‑safe field interpolation)
        xi_back = self.mapping.inverse_map(
            x_phys, self.mesh.nodes_positions.at_elements()
        )

        return QuadraturePositions(xi_ref=self.xi_ref, x_phys=x_phys, xi_back=xi_back)
