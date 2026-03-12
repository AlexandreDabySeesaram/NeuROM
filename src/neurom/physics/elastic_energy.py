import torch

from neurom.physics.term import Term
import neurom.differential as differential


class ElasticEnergy(Term):
    def __init__(self, field):
        self.field_name = field.name
        self.grad = differential.grad_factory(field.dim)

    def integrand(self, fields_layout):
        quad_interp_res = fields_layout[self.field_name]
        x = quad_interp_res.x
        u = quad_interp_res.u
        dx = quad_interp_res.measure

        # Compute du_dx**2
        du_dx = self.grad(x, u)
        inner = torch.einsum("eq...,eq...->eq...", du_dx, du_dx).squeeze()

        return (0.5 * inner) * dx
