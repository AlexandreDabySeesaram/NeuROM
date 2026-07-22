"""Five-parametric 1D bar with a tanh-graded Young's modulus -- CP-PGD setup.

Bi-clamped bar under a constant axial load, whose modulus has two zones E1 and E2
separated by a tanh transition centred at ``alpha`` with slope ``n``:

    E(x, E1, E2, alpha, n) = (E2 - E1) / 2 * tanh(n (x - alpha)) + (E2 + E1) / 2

The four material/geometry parameters are treated as extra coordinates, so the
solution is sought in separated form

    u(x, E1, E2, alpha, n) = sum_i X_i(x) lambda_i(E1) mu_i(E2) A_i(alpha) N_i(n)

Because ``E1, E2 > 0`` and ``tanh`` maps into (-1, 1), the modulus stays strictly
between E1 and E2 -- positivity is structural, no clamping needed.

This module only *builds* the problem (axes, decomposition, energy, model). Training
and post-processing come in a later step.
"""

from dataclasses import dataclass

import torch

from neurom.constraints import Dirichlet, NoConstraint
from neurom.decompositions import Axis, CPPGD
from neurom.field_layout import FieldLayout
from neurom.fields import Field
from neurom.geometry import IsoparametricMapping1D
from neurom.interpolation.integration_domain import IntegrationDomain
from neurom.interpolation.quadrature_assembly import QuadratureAssembly
from neurom.meshes import Topology
from neurom.neurom_model import NeuROMModel
from neurom.quadratures import MidPoint1D
from neurom.shape_functions import LinearSegment

torch.set_default_dtype(torch.float32)

# --- domain and parameter intervals ---------------------------------------
X_MIN, X_MAX = 0.0, 10.0
E1_MIN, E1_MAX = 10.0, 100.0
E2_MIN, E2_MAX = 10.0, 100.0
# alpha is kept away from the clamped ends so the transition always sits inside
# the bar; n spans near-uniform (tanh almost linear over the bar) to a sharp
# interface roughly one length unit wide.
ALPHA_MIN, ALPHA_MAX = 2.0, 8.0
N_MIN, N_MAX = 0.5, 5.0

DEFAULT_N_NODES = {"space": 30, "E1": 20, "E2": 20, "alpha": 15, "n": 15}

LOAD_VALUE = 1000.0
# The 2-parameter case was exactly rank-1; a non-separable modulus has no reason
# to be low-rank, hence the larger budget.
N_MODES_MAX = 10

# Axis order is load-bearing: it fixes the column order of CPPGD.evaluate and
# the key order of CPPGD.directory().
AXIS_ORDER = ["space", "E1", "E2", "alpha", "n"]

AXIS_BOUNDS = {
    "space": (X_MIN, X_MAX),
    "E1": (E1_MIN, E1_MAX),
    "E2": (E2_MIN, E2_MAX),
    "alpha": (ALPHA_MIN, ALPHA_MAX),
    "n": (N_MIN, N_MAX),
}


def make_axis(name, lo, hi, n_nodes, constraint, sf, quad, mapping, init_value=0.5):
    """Build one Axis on a uniform 1-D mesh of ``n_nodes`` nodes over [lo, hi].

    Wraps the Topology / Field / Axis boilerplate so the five axes are not five
    copy-pasted blocks. The Axis builds its own Mesh and QuadratureContext.

    Args:
        name (str): Axis name; also the prefix of its nodes-positions field.
        lo, hi (float): Interval bounds.
        n_nodes (int): Number of mesh nodes (so ``n_nodes - 1`` linear elements).
        constraint (Constraint): Dirichlet or NoConstraint for the monoms.
        sf (ShapeFunction), quad (QuadratureRule), mapping: shared discretisation.
        init_value (float): Constant seed for every new monom's nodal values.

    Returns:
        Axis: ready to be handed to CPPGD.
    """
    positions = torch.linspace(lo, hi, n_nodes).unsqueeze(-1)
    nodes = torch.arange(0, n_nodes)
    elements = torch.vstack([torch.arange(0, n_nodes - 1), torch.arange(1, n_nodes)]).T
    topology = Topology(nodes, elements)
    nodes_positions = Field(
        name=f"{name}_positions", topology=topology, values=positions
    )
    return Axis(
        name=name,
        nodes_positions=nodes_positions,
        sf=sf,
        mapping=mapping,
        quad=quad,
        constraint=constraint,
        init_values=init_value * torch.ones(n_nodes, 1),
    )


@dataclass
class Problem:
    """Everything ``build_problem`` assembles, kept together for the caller.

    Attributes:
        model (NeuROMModel): the trainable model (``model()`` fills the layout).
        pgd (CPPGD): the separated representation.
        field_layout (FieldLayout): holds the monom fields and the load.
        domain (IntegrationDomain): interpolates every active field.
        axes (dict[str, Axis]): the five axes, keyed by name.
    """

    model: NeuROMModel
    pgd: CPPGD
    field_layout: FieldLayout
    domain: IntegrationDomain
    axes: dict


def build_problem(loss_fn, *, n_modes_max=N_MODES_MAX, n_modes_ini=1, n_nodes=None):
    """Assemble the five axes, the CP-PGD, the load and the model.

    The energy is *injected*: this function never references ``energy`` directly,
    so the same wiring drives a different (e.g. non-linear PGD) functional
    unchanged.

    Args:
        loss_fn (Callable): ``loss_fn(field_layout, decomposition) -> Tensor``.
        n_modes_max (int): Mode budget of the CP-PGD.
        n_modes_ini (int): Number of initially active (trainable) modes.
        n_nodes (dict[str, int], optional): Per-axis node counts overriding
            ``DEFAULT_N_NODES``; used by the tests to build a tiny problem.

    Returns:
        Problem: the assembled objects.
    """
    sf = LinearSegment()
    quad = MidPoint1D()
    mapping = IsoparametricMapping1D(sf)

    counts = dict(DEFAULT_N_NODES)
    if n_nodes is not None:
        counts.update(n_nodes)

    field_layout = FieldLayout()

    n_x = counts["space"]
    constraints = {
        "space": Dirichlet(nodes=[0, n_x - 1], values_imposed=torch.zeros(2, 1)),
        "E1": NoConstraint(),
        "E2": NoConstraint(),
        "alpha": NoConstraint(),
        "n": NoConstraint(),
    }
    axes = [
        make_axis(
            name,
            *AXIS_BOUNDS[name],
            counts[name],
            constraints[name],
            sf,
            quad,
            mapping,
        )
        for name in AXIS_ORDER
    ]

    pgd = CPPGD(axes=axes, n_modes_max=n_modes_max, name="pgd", n_modes_ini=n_modes_ini)

    # The load is a *field* f(x), interpolated on the SAME quadrature as the
    # space monoms so that inner(f, X) aligns point-for-point. It is the single
    # spatial factor of the separated source f = f_0(x) (x) 1(E1) (x) ... ; the
    # constant parametric factors are carried by the int lambda_i dE1 ... terms
    # of the load part of the energy.
    axis_space = axes[0]
    load_field = field_layout.add(
        Field(
            name="load",
            topology=axis_space.topology,
            values=LOAD_VALUE * torch.ones(n_x, 1),
        )
    )
    assembly_f = QuadratureAssembly(axis_space.context, sf, load_field)

    domain = IntegrationDomain([*pgd.assemblies(), assembly_f])

    model = NeuROMModel(
        field_layout=field_layout,
        decomposition=pgd,
        integration_domain=domain,
        loss=lambda layout: loss_fn(layout, pgd),
    )

    return Problem(
        model=model,
        pgd=pgd,
        field_layout=field_layout,
        domain=domain,
        axes={axis.name: axis for axis in axes},
    )
