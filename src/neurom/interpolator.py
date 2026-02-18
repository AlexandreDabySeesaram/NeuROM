import torch
import torch.nn as nn

from neurom.geometry import barycentric_to_reference


class Interpolator(nn.Module):
    def __init__(self, mesh, field, sf, quad, mapping):
        super().__init__()
        self.mesh = mesh
        self.field = field
        self.sf = sf
        self.quad = quad
        self.mapping = mapping

    def get_quadrature_points(self):
        """
        Helper methods to prepare quadrature points.
        """
        # (N_q, N_nodes)
        x_q_barycentric = self.quad.points()

        # (N_q, dim )
        xi = barycentric_to_reference(
            x_lambda=x_q_barycentric, element=self.quad.reference_element
        )

        # (N_e, N_q, dim)
        xi_g = xi.unsqueeze(0).expand(self.mesh.topology.n_elements, -1, -1)

        return xi_g

    def interpolate_at_reference(self, xi, field_at_elements):
        """
        Interpolate field on reference coordinates.

        Args:
            xi: The reference coordinate (N_e, N_q, dim)
            field: The field to interpolate (N_e, N_nodes, dim)

        Returns:
            The interpolated field, tensor: (N_e, N_q, dim)
        """
        # (N_e, N_q, N_nodes)
        N = self.sf.N(xi)
        # Sum along N_nodes index
        # Product of tensor (N_e, N_nodes, dim) x (N_e, N_q, N_nodes)
        # This gives tensor (N_e, N_q, dim)
        return torch.einsum("en...,eqn...->eq...", field_at_elements, N)

    def measure(self):
        # Compute weighted measure
        w = self.quad.weights()
        dx = self.mapping.element_size(self.mesh.nodes_positions.at_elements())
        measure = dx * w
        return measure

    def interpolate(self):
        # (N_e, N_q, dim)
        xi_g = self.get_quadrature_points()

        # (N_e, N_q, dim)
        x_g = self.interpolate_at_reference(
            xi_g, self.mesh.nodes_positions.at_elements()
        )

        # Mark it for later autograd
        x_g.requires_grad_(True)

        # (N_e, N_q, dim)
        xi_q = self.mapping.inverse_map(x_g, self.mesh.nodes_positions.at_elements())

        # Gather nodal values per element
        # (N_e, N_q, dim)
        u_q = self.interpolate_at_reference(xi_q, self.field.at_elements())

        # Compute weighted measure
        measure = self.measure()

        return x_g, u_q, measure

    def interpolate_at(self, x):
        x = x.unsqueeze(1).unsqueeze(2)
        element_ids = self.mesh.elements_at(x)

        # Get connectivity for those elements
        element_nodes_ids = self.mesh.topology.conn[element_ids, :]

        # (N_e, N_q, dim)
        x_nodes = self.mesh.nodes_positions.at_elements()[element_ids]

        xi = self.mapping.inverse_map(x, x_nodes)
        u = self.interpolate_at_reference(
            xi, self.field.full_values()[element_nodes_ids]
        )

        return u.detach()
