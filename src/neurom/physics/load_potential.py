from neurom.physics.term import Term


class LoadPotential(Term):
    def __init__(self, field, f):
        self.field_name = field.name
        self.f = f

    def integrand(self, fields_layout):
        quad_interp_res = fields_layout[self.field_name]
        x = quad_interp_res.x
        u = quad_interp_res.u
        dx = quad_interp_res.measure

        return -(self.f(x) * u).squeeze() * dx
