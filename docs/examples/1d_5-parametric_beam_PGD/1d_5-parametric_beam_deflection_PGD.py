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
from neurom.differential import jacobian_field
from neurom.field_layout import FieldLayout
from neurom.fields import Field
from neurom.geometry import IsoparametricMapping1D
from neurom.inner import inner
from neurom.integrate import integrate
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


def energy(field_layout, decomposition, load_name="load"):
    """Total potential energy of the tanh-graded bar, in separated form.

    With u = sum_i X_i(x) lambda_i(E1) mu_i(E2) A_i(alpha) N_i(n) and

        E(x, E1, E2, alpha, n) = (E2 + E1)/2 + (E2 - E1)/2 tanh(n (x - alpha)),

    every factor of the elastic term separates into a product of 1-D integrals
    *except* tanh(n (x - alpha)), which couples x, alpha and n. That coupled block
    is integrated by an exact 3-D tensor-product quadrature over those three
    axes' quadrature points (one einsum per mode pair); the (E2 +/- E1)/2
    prefactors stay 1-D moments of the E1 and E2 factors.

    Sign convention follows the 2-parameter example: the returned value is
    ``elastic + load``, the load field carrying its own sign.

    Args:
        field_layout (FieldLayout): filled by ``model()`` (train mode).
        decomposition (CPPGD): supplies the active monom names via ``directory()``.
        load_name (str): name of the load field in the layout.

    Returns:
        torch.Tensor: 0-dim energy.
    """
    directory = decomposition.directory()
    n_modes = len(directory["space"])

    spc = [field_layout[name] for name in directory["space"]]
    e1 = [field_layout[name] for name in directory["E1"]]
    e2 = [field_layout[name] for name in directory["E2"]]
    alp = [field_layout[name] for name in directory["alpha"]]
    slp = [field_layout[name] for name in directory["n"]]

    # Space factors. jacobian_field returns (N_e, N_q, *u_shape, d) -- one extra
    # trailing axis compared to u -- which must be *contracted away* by inner(),
    # not reshaped away: reshaping only appears to work for d = 1 and turns the
    # cross terms into a silent element-wise broadcast (see the 2-parameter
    # example for the full story).
    X = [r.u for r in spc]
    gX = [jacobian_field(x=r.x, u=r.u) for r in spc]
    Jx = [r.measure for r in spc]

    # Parametric factors, with their own coordinate values (needed for the
    # first moments int E1 lambda_i lambda_j dE1 and int E2 mu_i mu_j dE2).
    lam, E1_val, J1 = [r.u for r in e1], [r.x for r in e1], [r.measure for r in e1]
    mu, E2_val, J2 = [r.u for r in e2], [r.x for r in e2], [r.measure for r in e2]
    A, Ja = [r.u for r in alp], [r.measure for r in alp]
    N, Jn = [r.u for r in slp], [r.measure for r in slp]

    # The one non-separable block: tanh(n (x - alpha)) on the tensor product of
    # the space, alpha and n quadrature points, shape (Qx, Qalpha, Qn). It does
    # not depend on the mode pair, so it is built once and reused below.
    # Rebuilt every call (rather than cached at setup) so it stays correct if the
    # meshes ever become trainable (r-adaptivity).
    xq = spc[0].x.reshape(-1)
    aq = alp[0].x.reshape(-1)
    nq = slp[0].x.reshape(-1)
    tanh_grid = torch.tanh(nq[None, None, :] * (xq[:, None, None] - aq[None, :, None]))

    # NB: as in the 2-parameter example, the cross terms (i, j) assume modes i
    # and j share a mesh per axis (the measure and coordinates indexed by i are
    # used for both). Independent per-mode meshes would need a common
    # intersection mesh with a recomputed measure.
    elastic = 0.0
    for i in range(n_modes):
        for j in range(n_modes):
            # Densities at the quadrature points, reused both integrated (the
            # separable part) and raw (contracted against tanh_grid).
            kx_density = inner(gX[i], gX[j]) * Jx[i]
            a_density = A[i] * A[j] * Ja[i]
            n_density = N[i] * N[j] * Jn[i]

            Kx = integrate(kx_density)
            P0 = integrate(a_density)
            Q0 = integrate(n_density)
            L0 = integrate(lam[i] * lam[j] * J1[i])
            L1 = integrate(E1_val[i] * lam[i] * lam[j] * J1[i])
            M0 = integrate(mu[i] * mu[j] * J2[i])
            M1 = integrate(E2_val[i] * mu[i] * mu[j] * J2[i])

            T = torch.einsum(
                "xan,x,a,n->",
                tanh_grid,
                kx_density.reshape(-1),
                a_density.reshape(-1),
                n_density.reshape(-1),
            )

            mean_modulus = 0.5 * (M1 * L0 + L1 * M0)  # (E2 + E1) / 2
            half_contrast = 0.5 * (M1 * L0 - L1 * M0)  # (E2 - E1) / 2
            elastic = elastic + mean_modulus * Kx * P0 * Q0 + half_contrast * T
    elastic = 0.5 * elastic

    # Constant load: separable across every axis, so one 1-D integral per factor.
    load_f = field_layout[load_name].u
    load = 0.0
    for i in range(n_modes):
        Fx = integrate(inner(load_f, X[i]) * Jx[i])
        load = load + (
            Fx
            * integrate(lam[i] * J1[i])
            * integrate(mu[i] * J2[i])
            * integrate(A[i] * Ja[i])
            * integrate(N[i] * Jn[i])
        )

    return elastic + load
