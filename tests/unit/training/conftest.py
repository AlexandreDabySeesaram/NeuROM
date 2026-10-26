"""Shared fixtures of the training tests."""

import pytest
import torch

from neurom.constraints import NoConstraint
from neurom.decompositions import CPPGD, MonomSpec
from neurom.field_layout import FieldLayout
from neurom.interpolation import IntegrationDomain
from neurom.math import integrate
from neurom.neurom_model import NeuROMModel
from neurom.shape_functions import LinearBar


# --- setup --------------------------------------------------------------
# L2 projection of the rank-2 target f(x, y) = x + 2y on [0, 1]^2:
#   J(u) = 1/2 int u^2 - int f u,   minimum -1/2 int f^2 = -4/3 at u = f.
# Not x + y: with identical seeds on both axes, a symmetric target keeps
# every mode symmetric (a_m = b_m), which cannot capture the residual.
# Separated, with u = sum_m a_m(x) b_m(y):
#   J = 1/2 sum_mn (int a_m a_n)(int b_m b_n)
#       - sum_m [(int x a_m)(int b_m) + 2 (int a_m)(int y b_m)].


@pytest.fixture
def make_model(make_factor_space):
    def _make_model():
        specs = [
            MonomSpec(
                space=make_factor_space(name=name, n=6, lo=0.0, hi=1.0),
                sf=LinearBar(),
                constraint=NoConstraint(),
                init_values=0.5 * torch.ones(6, 1),
            )
            for name in ("x", "y")
        ]
        cppgd = CPPGD(monom_specs=specs, n_modes_ini=1)

        def energy(layout):
            directory = cppgd.directory()
            a = [layout[name] for name in directory["x"]]
            b = [layout[name] for name in directory["y"]]

            def inner(f, g, h=None):
                w = f.u.values * g.u.values if g is not None else f.u.values
                if h is not None:
                    w = w * h
                return integrate(w * f.measure.values)

            n = len(a)
            quadratic = sum(
                inner(a[m], a[n_]) * inner(b[m], b[n_])
                for m in range(n)
                for n_ in range(n)
            )
            linear = sum(
                inner(a[m], None, a[m].x.values) * inner(b[m], None)
                + 2 * inner(a[m], None) * inner(b[m], None, b[m].x.values)
                for m in range(n)
            )
            return 0.5 * quadratic - linear

        return NeuROMModel(FieldLayout(), cppgd, IntegrationDomain([]), energy)

    return _make_model
