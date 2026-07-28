"""Reference solutions for the 5-parametric bar, by direct FEM at a few parameter points.

The PGD surrogate must be judged against something it did not produce. That
something is a plain, non-reduced FEM solve: one :class:`~neurom.fem_model.FEMModel`
per parameter point, on a fine mesh, minimising the *same* energy

    0.5 int E(x; E1, E2, alpha, n) u'(x)^2 dx + int f u dx,     u(X_MIN) = u(X_MAX) = 0

with the modulus injected analytically at the quadrature points (no interpolation
error on E). Each solve is an ordinary quadratic minimisation solved by LBFGS.

A Latin Hypercube Sample of ``N_LHS`` points in the 4D parameter box (x is
excluded, it is the space grid) is drawn so that the ``overall`` L2 measure is a
genuine global figure over the whole parameter box, at a fraction of the cost
of the full tensor grid it replaces (625 solves at 5 points/axis) while still
space-filling far better than plain random draws. The two ``EXTREME_POINTS``
are appended for the per-point plots. The result is written to
``reference_solution.pt`` so that every decomposition strategy is compared
against *the same* numbers, on the same ``x`` grid, without re-solving.

Run it once::

    python docs/examples/1d_5-parametric_beam_PGD/reference/reference_fem_solution.py

then load it from anywhere::

    from reference_fem_solution import load_reference
    ref = load_reference()          # dict with "x", "params", "u", ...
"""

import importlib.util
from contextlib import contextmanager
from pathlib import Path

import torch
from scipy.stats import qmc

from neurom.constraints import Dirichlet
from neurom.fem_model import FEMModel
from neurom.field_layout import FieldLayout
from neurom.fields import Field, TrainableField
from neurom.geometry import IsoparametricMapping1D
from neurom.interpolation import (
    IntegrationDomain,
    PointWiseInterpolator,
    QuadratureAssembly,
    QuadratureContext,
)
from neurom.meshes import Mesh, Topology
from neurom.physics import ElasticEnergy, LoadPotential
from neurom.physics_loss import PhysicsLoss
from neurom.quadratures import TwoPoints1D
from neurom.shape_functions import LinearSegment

# The reference is solved in double precision. In float32 the LBFGS solution
# stalls around 1e-3 relative -- measured: halving both moduli, which must scale
# u by exactly 2, was off by 3e-3, and the constant-modulus check sat at 2.7e-4.
# In float64 those become 8e-5 and 6e-6, i.e. the reference stops being the
# limiting error when a surrogate is compared against it. The bundle is cast
# back to the caller's dtype on load.
DTYPE = torch.float64

HERE = Path(__file__).resolve().parent
EXAMPLE_PATH = HERE.parent / "1d_5-parametric_beam_deflection_PGD.py"
REFERENCE_PATH = HERE / "reference_solution.pt"

# Reference discretisation: much finer than any PGD space axis (30 nodes), and a
# two-point rule so that the graded modulus is integrated properly inside each
# element -- with a sharp tanh (n = 5) the midpoint rule is visibly off.
N_NODES = 400
QUADRATURE = TwoPoints1D
N_X_SAMPLES = 101
N_PARAM_POINTS = 10
# The reference set is a Latin Hypercube Sample: N_LHS points in the 4D
# parameter box, one per stratum per axis, independently shuffled across axes.
# 300 trades the full grid's 625 solves for roughly half the cost while still
# covering every axis far more evenly than 300 plain random draws would.
N_LHS = 300
SEED = 0

# Deterministic LBFGS budget. The energy is quadratic in the nodal values, so
# this converges to machine-ish precision; it is not a hyperparameter to tune.
N_EPOCHS = 5
MAX_ITER = 200

PARAM_NAMES = ["E1", "E2", "alpha", "n"]

# Two deliberately hard points, appended to the random ones. Both use a sharp
# transition (n near its maximum), so E(x) is close to a step and u(x) is two
# parabola halves joined at a kink -- the regime where a separated
# representation has the most trouble. They are *not* mirror images of each
# other: opposite contrast direction, different contrast ratio (10x vs 5x) and
# a transition on either side of mid-span, so nothing about the second is
# implied by getting the first right.
EXTREME_POINTS = [
    ("soft-stiff-left", {"E1": 10.0, "E2": 100.0, "alpha": 3.0, "n": 5.0}),
    ("stiff-soft-right", {"E1": 100.0, "E2": 20.0, "alpha": 6.5, "n": 4.0}),
]


@contextmanager
def double_precision():
    """Run the enclosed block with ``DTYPE`` as the default dtype, then restore.

    The library builds its tensors with the *default* dtype, and importing the
    example module sets that to float32 -- hence a scoped override rather than a
    module-level ``set_default_dtype``.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(DTYPE)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def load_example(path=EXAMPLE_PATH):
    """Import the PGD example script by path (its filename starts with a digit).

    It is the single source of truth for the geometry, the parameter intervals,
    the load and the modulus law -- this module must solve *that* problem, not a
    copy of it that can drift.
    """
    spec = importlib.util.spec_from_file_location("beam5p", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_parameters(n_points=N_PARAM_POINTS, seed=SEED, example=None):
    """Draw ``n_points`` parameter tuples by Latin Hypercube Sampling.

    Delegates the stratified draw to ``scipy.stats.qmc.LatinHypercube``: each
    axis is cut into ``n_points`` equal strata, every stratum is used exactly
    once per axis, and the stratum order is an independent random permutation
    per axis (no ``scramble``/``optimization`` beyond that default). That gives
    even coverage of each axis on its own -- unlike plain uniform draws, which
    can clump or leave gaps -- without the ``n_points**4`` cost of a full
    tensor grid.

    Seeded, so the reference set is reproducible: regenerating the file gives
    the same parameter points and therefore stays comparable to older runs.

    Args:
        n_points (int): How many parameter points to draw.
        seed (int): RNG seed.
        example (module, optional): the loaded example module (for its bounds).

    Returns:
        torch.Tensor: ``(n_points, 4)``, columns ordered as ``PARAM_NAMES``.
    """
    example = example if example is not None else load_example()
    lows = [example.AXIS_BOUNDS[name][0] for name in PARAM_NAMES]
    highs = [example.AXIS_BOUNDS[name][1] for name in PARAM_NAMES]

    sampler = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=seed)
    unit = sampler.random(n=n_points)
    scaled = qmc.scale(unit, lows, highs)
    return torch.as_tensor(scaled, dtype=torch.get_default_dtype())


def solve_one(params, x_samples, *, example=None, n_nodes=N_NODES, quad=None):
    """Solve the bar at one parameter point and sample the displacement.

    Args:
        params (dict): ``{"E1":..., "E2":..., "alpha":..., "n":...}`` (floats).
        x_samples (torch.Tensor): 1-D positions where the solution is returned.
        example (module, optional): the loaded example module.
        n_nodes (int): nodes of the reference mesh.
        quad (QuadratureRule, optional): defaults to ``TwoPoints1D()``.

    Returns:
        tuple: ``(u_at_samples, energy)`` -- a ``(len(x_samples),)`` tensor and
        the converged energy, both detached.
    """
    example = example if example is not None else load_example()
    with double_precision():
        x_samples = x_samples.to(DTYPE)
        quad = quad if quad is not None else QUADRATURE()

        x_min, x_max = example.X_MIN, example.X_MAX
        positions = torch.linspace(x_min, x_max, n_nodes).unsqueeze(-1)
        nodes = torch.arange(0, n_nodes)
        elements = torch.vstack(
            [torch.arange(0, n_nodes - 1), torch.arange(1, n_nodes)]
        ).T
        topology = Topology(nodes, elements)

        sf = LinearSegment()
        mapping = IsoparametricMapping1D(sf)

        field_layout = FieldLayout()
        u = field_layout.add(
            TrainableField(
                name="displacement",
                topology=topology,
                init_values=torch.zeros(n_nodes, 1),
                constraint=Dirichlet(
                    nodes=[0, n_nodes - 1], values_imposed=torch.zeros(2, 1)
                ),
            )
        )
        x_field = field_layout.add(
            Field(name="positions", topology=topology, values=positions)
        )
        f = field_layout.add(
            Field(
                name="load",
                topology=topology,
                values=example.LOAD_VALUE * torch.ones(n_nodes, 1),
            )
        )

        mesh = Mesh(topology=topology, nodes_positions=x_field)
        ctx = QuadratureContext(mesh, quad, mapping)
        domain = IntegrationDomain(
            [QuadratureAssembly(ctx, sf, u), QuadratureAssembly(ctx, sf, f)]
        )

        # The modulus goes in as a *callable* of the quadrature points, so E(x) is
        # exact there rather than piecewise-linearly interpolated. Sign convention
        # matches the PGD energy: elastic + int f u, i.e. ElasticEnergy - LoadPotential.
        physics = ElasticEnergy(
            field=u, modulus=lambda x: example.modulus(x, **params)
        ) - LoadPotential(field=u, f=f)

        model = FEMModel(
            mesh=mesh,
            field_layout=field_layout,
            integration_domain=domain,
            loss=PhysicsLoss(physics=physics, field_layout=field_layout),
        )

        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=1e-1,
            max_iter=MAX_ITER,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            loss = model()
            loss.backward(retain_graph=True)
            return loss

        for _ in range(N_EPOCHS):
            model()
            optimizer.step(closure)

        energy = model().detach()
        pwi = PointWiseInterpolator(mesh, sf, u, mapping)
        u_samples = pwi.at_position(x_samples.reshape(-1)).reshape(-1).detach()
    return u_samples, energy


def generate(n_lhs=N_LHS, n_x=N_X_SAMPLES, verbose=True):
    """Solve at every parameter point -- the LHS set then ``EXTREME_POINTS``.

    Returns:
        dict: ``x`` ``(P,)``, ``params`` ``(K, 4)``, ``param_names``, ``labels``
        (``K`` strings, ``"lhs-i"`` or the extreme point's name), ``u``
        ``(K, P)``, ``energy`` ``(K,)`` and a ``metadata`` dict describing the
        discretisation and naming the highlighted points.
    """
    example = load_example()
    with double_precision():
        x_samples = torch.linspace(example.X_MIN, example.X_MAX, n_x)
        lhs = sample_parameters(n_lhs, seed=SEED, example=example)
        extreme = torch.tensor(
            [[point[name] for name in PARAM_NAMES] for _, point in EXTREME_POINTS]
        )
        params = torch.cat([lhs, extreme])

    labels = [f"lhs-{i}" for i in range(len(lhs))]
    labels += [name for name, _ in EXTREME_POINTS]
    highlight = {name for name, _ in EXTREME_POINTS}

    solutions, energies = [], []
    for i, (label, row) in enumerate(zip(labels, params)):
        point = dict(zip(PARAM_NAMES, (v.item() for v in row)))
        u_samples, energy = solve_one(point, x_samples, example=example)
        solutions.append(u_samples)
        energies.append(energy)
        # 300 LHS solves would drown the terminal one line each: report a
        # heartbeat every 50 points, plus every highlighted (extreme) point.
        if verbose and (label in highlight or i % 50 == 0 or i == len(labels) - 1):
            values = ", ".join(f"{k}={v:7.3f}" for k, v in point.items())
            print(
                f"[{i + 1:>4}/{len(labels)}] {label:>17}  {values}"
                f" -> energy {energy.item():14.6e}  min u {u_samples.min():12.5e}"
            )

    return {
        "x": x_samples,
        "params": params,
        "param_names": list(PARAM_NAMES),
        "labels": labels,
        "u": torch.stack(solutions),
        "energy": torch.stack(energies),
        "metadata": {
            "n_nodes": N_NODES,
            "quadrature": QUADRATURE.__name__,
            "shape_function": "LinearSegment",
            "load": example.LOAD_VALUE,
            "x_bounds": (example.X_MIN, example.X_MAX),
            "axis_bounds": dict(example.AXIS_BOUNDS),
            "sampling": "lhs",
            "n_lhs": n_lhs,
            "seed": SEED,
            # The points worth looking at one by one; the rest are the LHS
            # set, there to be averaged over by ``overall``, not plotted.
            "highlight": [name for name, _ in EXTREME_POINTS],
        },
    }


def save_reference(reference, path=REFERENCE_PATH):
    """Write the reference bundle to ``path`` (``torch.save``)."""
    torch.save(reference, path)
    return path


def load_reference(path=REFERENCE_PATH):
    """Load the reference bundle written by ``save_reference``.

    The stored tensors are float64 (see ``DTYPE``); they are cast to the caller's
    current default dtype so that comparing them to a float32 surrogate does not
    trip torch's mixed-dtype rules.

    Raises:
        FileNotFoundError: with the command to run if the file is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No reference solution at {path}. Generate it with:\n"
            f"    python {HERE / 'reference_fem_solution.py'}"
        )
    bundle = torch.load(path, weights_only=False)
    dtype = torch.get_default_dtype()
    return {
        key: value.to(dtype)
        if torch.is_tensor(value) and value.is_floating_point()
        else value
        for key, value in bundle.items()
    }


def check_against_analytical(example=None):
    """Sanity check: at ``E1 == E2`` the graded bar has a closed-form solution.

    With a constant modulus the solution is the parabola
    ``0.5 f (x - X_MIN)(x - X_MAX) / E`` (the 2-parametric example's analytical
    solution). Returns the relative L2 error of the FEM reference against it --
    if this is not small, the reference itself is wrong and nothing compared to
    it means anything.
    """
    example = example if example is not None else load_example()
    E = 50.0
    with double_precision():
        x_samples = torch.linspace(example.X_MIN, example.X_MAX, N_X_SAMPLES)
        u_exact = (
            0.5
            * example.LOAD_VALUE
            * (x_samples - example.X_MIN)
            * (x_samples - example.X_MAX)
            / E
        )
    params = {"E1": E, "E2": E, "alpha": 5.0, "n": 2.0}
    u_fem, _ = solve_one(params, x_samples, example=example)
    return (torch.linalg.norm(u_fem - u_exact) / torch.linalg.norm(u_exact)).item()


def main(verbose=True):
    """Validate the solver, generate the reference set, save it."""
    error = check_against_analytical()
    if verbose:
        print(f"constant-modulus check: relative L2 error {error:.3e}")
    if error > 1e-2:
        raise RuntimeError(
            f"FEM reference disagrees with the analytical constant-modulus "
            f"solution (relative L2 error {error:.3e}); not writing a reference."
        )

    reference = generate(verbose=verbose)
    path = save_reference(reference)
    if verbose:
        print(f"\nwrote {reference['u'].shape[0]} solutions to {path}")
    return reference


if __name__ == "__main__":
    main()
