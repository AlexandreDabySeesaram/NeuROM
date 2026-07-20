import pytest
import torch

from neurom.decompositions import Axis, CPPGD
from neurom.neurom_model import NeuROMModel
from neurom.quadratures import TwoPoints1D
from neurom.shape_functions import LinearSegment
from neurom.geometry import IsoparametricMapping1D
from neurom.meshes import Topology
from neurom.fields import Field
from neurom.constraints import Dirichlet, NoConstraint
from neurom.differential import jacobian_field
from neurom.integrate import integrate
from neurom.field_layout import FieldLayout
from neurom.interpolation import IntegrationDomain

torch.set_default_dtype(torch.float32)


def build_axis(name, coords, constraint, init_values):
    n = coords.shape[0]
    nodes = torch.arange(0, n)
    elements = torch.vstack([torch.arange(0, n - 1), torch.arange(1, n)]).T
    topology = Topology(nodes, elements)
    positions = Field(name=f"{name}_positions", topology=topology, values=coords)
    sf = LinearSegment()
    return Axis(
        name=name,
        nodes_positions=positions,
        sf=sf,
        mapping=IsoparametricMapping1D(sf),
        quad=TwoPoints1D(),
        constraint=constraint,
        init_values=init_values,
    )


def potential_energy(cppgd, field_layout, f_value):
    """External separable parametric energy, read from the filled FieldLayout via directory()."""
    d = cppgd.directory()
    space = [field_layout[name] for name in d["space"]]
    para = [field_layout[name] for name in d["E"]]
    n_modes = len(space)

    dS = [jacobian_field(space[m].x, space[m].u).reshape(space[m].u.shape)
          for m in range(n_modes)]        # each (N_e, N_q, 1)
    
    S = [space[m].u for m in range(n_modes)]
    measure_s = space[0].measure

    g = [para[m].u for m in range(n_modes)]
    E_val = para[0].x                       # coordinate on the E axis == E
    measure_E = para[0].measure

    elastic = 0.0
    for m in range(n_modes):
        for n in range(n_modes):
            Kx = integrate(dS[m] * dS[n] * measure_s)
            AE = integrate(E_val * g[m] * g[n] * measure_E)
            elastic = elastic + Kx * AE
    elastic = 0.5 * elastic

    load = 0.0
    for m in range(n_modes):
        Fx = integrate(f_value * S[m] * measure_s)
        Gm = integrate(g[m] * measure_E)
        load = load + Fx * Gm

    return elastic + load


class Test1dBeamDeflectionPGD:
    relative_tolerance: float = 5e-2

    def test_parametric_beam_matches_analytical(self):
        x_min, x_max = 0.0, 10.0
        E_min, E_max = 100.0, 1000.0
        N_x, N_E = 40, 20
        f_value = 1000.0

        x_coords = torch.linspace(x_min, x_max, N_x).unsqueeze(-1)
        E_coords = torch.linspace(E_min, E_max, N_E).unsqueeze(-1)

        space_axis = build_axis(
            "space",
            x_coords,
            Dirichlet(nodes=[0, N_x - 1], values_imposed=torch.zeros(2, 1)),
            init_values=torch.zeros(N_x, 1),   # space monom starts at 0
        )
        para_axis = build_axis(
            "E",
            E_coords,
            NoConstraint(),
            init_values=torch.ones(N_E, 1),    # E monom starts at 1 (gradient flows)
        )

        cppgd = CPPGD(axes=[space_axis, para_axis], n_modes_max=1, n_modes_ini=1, name="beam")
        field_layout = FieldLayout()
        domain = IntegrationDomain(cppgd.assemblies())
        model = NeuROMModel(
            field_layout, cppgd, domain,
            energy=lambda out: potential_energy(cppgd, out, f_value),
        )

        optimizer = torch.optim.LBFGS(
            [p for p in model.parameters() if p.requires_grad],
            lr=1.0, max_iter=100, line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            out = model()
            loss = model.energy(out)
            loss.backward(retain_graph=True)
            return loss

        for _ in range(30):
            optimizer.step(closure)

        # Compare assembled u(x, E) to the analytical parametric deflection.
        x_test = torch.linspace(x_min, x_max, 15)
        E_test = torch.linspace(E_min, E_max, 5)
        u = model.assemble([x_test, E_test])          # (15, 5)

        xx = x_test.unsqueeze(-1)                      # (15, 1)
        EE = E_test.unsqueeze(0)                       # (1, 5)
        u_analytical = 0.5 * f_value * (xx - x_min) * (xx - x_max) / EE

        scale = float(u_analytical.abs().max())
        assert u.detach().numpy() == pytest.approx(
            u_analytical.numpy(), abs=self.relative_tolerance * scale
        )

        model.eval()
        x_pts = torch.linspace(x_min, x_max, 7)
        E_pts = torch.linspace(E_min, E_max, 7)
        u_pw = model(torch.stack([x_pts, E_pts], dim=1))   # matched pointwise, (7, 1)
        u_pw_analytical = 0.5 * f_value * (x_pts - x_min) * (x_pts - x_max) / E_pts
        scale_pw = float(u_pw_analytical.abs().max())
        assert u_pw.reshape(-1).numpy() == pytest.approx(
            u_pw_analytical.numpy(), abs=self.relative_tolerance * scale_pw
        )

    def test_greedy_enrichment_second_mode_stays_bounded(self):
        """Rank-1 analytical field must still hold after a greedy mode-2 enrichment.

        Adding a second mode should not perturb the already-converged rank-1
        solution: the enrichment should drive the extra mode towards ~0, so the
        rank-2 assembled field must still match the analytical rank-1 solution
        within tolerance.
        """
        x_min, x_max = 0.0, 10.0
        E_min, E_max = 100.0, 1000.0
        N_x, N_E = 40, 20
        f_value = 1000.0

        x_coords = torch.linspace(x_min, x_max, N_x).unsqueeze(-1)
        E_coords = torch.linspace(E_min, E_max, N_E).unsqueeze(-1)

        space_axis = build_axis(
            "space",
            x_coords,
            Dirichlet(nodes=[0, N_x - 1], values_imposed=torch.zeros(2, 1)),
            init_values=torch.zeros(N_x, 1),
        )
        para_axis = build_axis(
            "E",
            E_coords,
            NoConstraint(),
            init_values=torch.ones(N_E, 1),
        )

        cppgd = CPPGD(axes=[space_axis, para_axis], n_modes_max=2, n_modes_ini=1, name="beam")
        field_layout = FieldLayout()
        domain = IntegrationDomain(cppgd.assemblies())
        model = NeuROMModel(
            field_layout, cppgd, domain,
            energy=lambda out: potential_energy(cppgd, out, f_value),
        )

        optimizer = torch.optim.LBFGS(
            [p for p in model.parameters() if p.requires_grad],
            lr=1.0, max_iter=100, line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            out = model()
            loss = model.energy(out)
            loss.backward(retain_graph=True)
            return loss

        for _ in range(30):
            optimizer.step(closure)

        # Greedy-enrich with a second mode: freeze mode 0, activate+zero mode 1.
        cppgd.freeze_mode(0)
        cppgd.add_mode()

        optimizer2 = torch.optim.LBFGS(
            [p for p in model.parameters() if p.requires_grad],
            lr=1.0, max_iter=200, line_search_fn="strong_wolfe",
        )

        def closure2():
            optimizer2.zero_grad()
            out = model()
            loss = model.energy(out)
            loss.backward(retain_graph=True)
            return loss

        for _ in range(150):
            optimizer2.step(closure2)

        # Compare assembled u(x, E) (now rank-2) to the analytical rank-1 field.
        x_test = torch.linspace(x_min, x_max, 15)
        E_test = torch.linspace(E_min, E_max, 5)
        u = model.assemble([x_test, E_test])          # (15, 5)

        xx = x_test.unsqueeze(-1)                      # (15, 1)
        EE = E_test.unsqueeze(0)                       # (1, 5)
        u_analytical = 0.5 * f_value * (xx - x_min) * (xx - x_max) / EE

        scale = float(u_analytical.abs().max())
        assert u.detach().numpy() == pytest.approx(
            u_analytical.numpy(), abs=self.relative_tolerance * scale
        )
