import torch
import torch.nn as nn

import neurom.elements as elements


class ElementEvaluator1D(nn.Module):
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
        xi = elements.barycentric_to_reference(
            x_lambda=x_q_barycentric, element=self.quad.reference_element
        )

        # (N_e, N_q, dim)
        xi_g = xi.unsqueeze(0).expand(self.mesh.n_elements, -1, -1)

        return xi_g

    def interpolate_at(self, xi, field_at_elements):
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
        return torch.einsum("en...,eqn...->eq...", field_at_elements, N)

    def measure(self):
        # Compute weighted measure
        w = self.quad.weights()
        dx = self.mapping.element_size(x_el_nodes)
        measure = dx * w
        return measure

    def interpolate(self, field):
        # (N_e, N_q, dim)
        xi_g = self.get_quadrature_points()

        # (N_e, N_q, dim)
        field_at_elements = self.field_at_elements(field)

        interpolate_at(xi_g, field_at_elements)
        # (N_e, N_q, dim)
        x_g = self.mapping.map(xi_g, x_el_nodes)
        # Required for autograd
        x_g.requires_grad_(True)

        # (N_e, N_q, dim)
        xi_q = self.mapping.inverse_map(x_g, x_el_nodes)

        # (N_e, N_q, N_nodes)
        N = self.sf.N(xi_q)

        # Gather nodal values per element
        # (N_e, N_nodes, dim)
        element_values = self.field.full_values()[self.mesh.conn]
        element_values = element_values.to(N.dtype)

        # Interpolate field
        # Product of tensor (N_e, N_nodes, dim) x (N_e, N_q, N_nodes) over N_nodes
        # This gives tensor (N_e, N_q, dim)
        u_q = torch.einsum("en...,eqn...->eq...", element_values, N)
        return x_g, u_q, measure

    def evaluate(self):
        # (N_q, N_nodes)
        x_q_barycentric = self.quad.points()
        # (N_q, dim )
        xi = elements.barycentric_to_reference(
            x_lambda=x_q_barycentric, element=self.quad.reference_element
        )
        # (N_e, N_q, dim)
        xi_g = xi.unsqueeze(0).expand(self.mesh.n_elements, -1, -1)
        # (N_e, N_nodes)
        x_el_nodes = self.mesh.element_nodes_positions
        # (N_e, N_q, dim)
        x_g = self.mapping.map(xi_g, x_el_nodes)
        # Required for autograd
        x_g.requires_grad_(True)

        # (N_e, N_q, dim)
        xi_q = self.mapping.inverse_map(x_g, x_el_nodes)

        # (N_e, N_q, N_nodes)
        N = self.sf.N(xi_q)

        # Compute weighted measure
        w = self.quad.weights()
        dx = self.mapping.element_size(x_el_nodes)
        measure = dx * w

        # Gather nodal values per element
        # (N_e, N_nodes, dim)
        element_values = self.field.full_values()[self.mesh.conn]
        element_values = element_values.to(N.dtype)

        # Interpolate field
        # Product of tensor (N_e, N_nodes, dim) x (N_e, N_q, N_nodes) over N_nodes
        # This gives tensor (N_e, N_q, dim)
        u_q = torch.einsum("en...,eqn...->eq...", element_values, N)
        return x_g, u_q, measure

    def evaluate_at(self, x):
        device = self.mesh.nodes_positions.device
        x = x.unsqueeze(1).unsqueeze(2)
        # List elements to which `x` belongs to.
        ids = []
        for x_i in x:
            for e, conn in enumerate(self.mesh.conn):
                x_first = self.mesh.nodes_positions[conn[0]]
                x_second = self.mesh.nodes_positions[conn[1]]
                if x_i >= x_first and x_i <= x_second:
                    ids.append(e)
                    break

        element_ids = torch.tensor(ids, device=device)

        element_nodes_ids = self.mesh.conn[element_ids, :]
        # (N_e, N_q, dim)
        x_nodes = self.mesh.element_nodes_positions[element_ids]

        xi = self.mapping.inverse_map(x, x_nodes)
        N = self.sf.N(xi)
        u_full = self.field.full_values()
        nodes_values = u_full[element_nodes_ids]
        nodes_values = nodes_values.to(N.dtype)
        u_q = torch.einsum("en...,eqn...->eq...", nodes_values, N)
        return u_q.detach()
