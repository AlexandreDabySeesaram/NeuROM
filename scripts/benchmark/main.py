"""Performance benchmark for the full NeuROM simulation pipeline.

A *simulation* here means the complete pipeline of a 1D beam-deflection problem:
building the connectivity, fields, mesh, mapping, physics, loss and FEM model,
followed by training with an LBFGS optimiser. The benchmark repeats that whole
simulation several times and reports the mean and standard deviation of the
wall-clock execution time (split into a *setup* phase and a *training* phase),
so performance regressions can be detected and tracked over time.

The problem is self-contained (no external mesh file is required) and scales
with ``--nodes``, mirroring ``tests/integration/test_1d_beam_deflection.py``.

Example:
    uv run python scripts/benchmark/benchmark.py --repeats 20 --nodes 200 --epochs 3
    uv run python scripts/benchmark/benchmark.py --quadrature two-points --json out.json
"""

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass

import torch

from neurom.quadratures import MidPoint1D, TwoPoints1D
from neurom.shape_functions import LinearBar
from neurom.geometry import IsoparametricMapping1D
from neurom.meshes import Mesh, Connectivity
from neurom.fields import Field, TrainableField
from neurom.constraints import Dirichlet
from neurom.field_layout import FieldLayout
from neurom.interpolation import (
    QuadratureContext,
    QuadratureAssembly,
    IntegrationDomain,
)
from neurom.physics import ElasticEnergy, LoadPotential
from neurom.physics_loss import PhysicsLoss
from neurom.fem_model import FEMModel

# Available 1D quadrature rules selectable from the command line.
QUADRATURES = {
    "midpoint": MidPoint1D,
    "two-points": TwoPoints1D,
}


@dataclass
class PhaseTimings:
    """Wall-clock duration in seconds of each phase of a single simulation.

    Attributes:
        setup (float): Seconds spent building the model (connectivity, fields,
            mesh, mapping, physics, loss, FEM model and optimiser).
        train (float): Seconds spent in the training loop.
    """

    setup: float
    train: float

    @property
    def total(self) -> float:
        """Total simulation time.

        Returns:
            float: Sum of the setup and training durations, in seconds.
        """
        return self.setup + self.train


def _sync(device: torch.device) -> None:
    """Block until all queued device work has finished.

    Required for accurate timing on CUDA, where kernels are launched
    asynchronously. A no-op on CPU.

    Args:
        device (torch.device): The device the benchmark runs on.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()


def build_model(
    n_nodes: int,
    quad_cls,
    device: torch.device,
):
    """Build the full FEM model and optimiser for the 1D beam problem.

    Constructs the connectivity, displacement and load fields, mesh, isoparametric
    mapping, elastic-energy / load-potential physics, physics loss, FEM model and
    LBFGS optimiser for a clamped 1D beam under a uniform load.

    Args:
        n_nodes (int): Number of mesh nodes along the beam. Controls problem size.
        quad_cls (type): Quadrature-rule class to instantiate (see ``QUADRATURES``).
        device (torch.device): Device on which to allocate the tensors.

    Returns:
        tuple: A pair ``(model, optimizer)`` where ``model`` is a
        :class:`~neurom.fem_model.FEMModel` and ``optimizer`` is a configured
        ``torch.optim.LBFGS`` instance.
    """
    x_min, x_max = 0.0, 10.0

    x_array = torch.linspace(x_min, x_max, n_nodes, device=device).unsqueeze(-1)
    nodes = torch.arange(0, n_nodes, device=device)
    elements = torch.vstack(
        [torch.arange(0, n_nodes - 1), torch.arange(1, n_nodes)]
    ).T.to(device)

    u_init = 0.5 * torch.ones(n_nodes, 1, device=device)
    load = 1000.0 * torch.ones(n_nodes, 1, device=device)

    connectivity = Connectivity(nodes, elements)
    sf = LinearBar()
    quad = quad_cls()

    field_layout = FieldLayout()
    u = field_layout.add(
        TrainableField(
            name="displacement",
            connectivity=connectivity,
            init_values=u_init,
            constraint=Dirichlet(
                nodes=[0, n_nodes - 1], values_imposed=torch.zeros(2, 1, device=device)
            ),
        )
    )
    x = field_layout.add(
        Field(name="positions", connectivity=connectivity, values=x_array)
    )
    f = field_layout.add(Field(name="load", connectivity=connectivity, values=load))

    mesh = Mesh(connectivity=connectivity, nodes_positions=x)
    mapping = IsoparametricMapping1D(sf, mesh)

    physics = ElasticEnergy(field=u) - LoadPotential(field=u, f=f)
    physics_loss = PhysicsLoss(physics=physics, field_layout=field_layout)

    ctx = QuadratureContext(mesh, quad, mapping)
    assembly_u = QuadratureAssembly(ctx, sf, u)
    assembly_f = QuadratureAssembly(ctx, sf, f)
    domain = IntegrationDomain([assembly_u, assembly_f])

    model = FEMModel(
        mesh=mesh,
        field_layout=field_layout,
        integration_domain=domain,
        loss=physics_loss,
    )
    model.to(device)

    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=1e-1, max_iter=50, line_search_fn="strong_wolfe"
    )
    return model, optimizer


def train(model: FEMModel, optimizer, n_epochs: int) -> None:
    """Run the training loop for a fixed number of epochs.

    Args:
        model (FEMModel): The FEM model to train.
        optimizer: The optimiser driving the training (an LBFGS instance).
        n_epochs (int): Number of optimiser steps to perform.
    """

    def closure():
        optimizer.zero_grad()
        loss = model()
        loss.backward(retain_graph=True)
        return loss

    for _ in range(n_epochs):
        optimizer.step(closure)


def run_simulation(
    n_nodes: int,
    n_epochs: int,
    quad_cls,
    device: torch.device,
    seed: int,
) -> PhaseTimings:
    """Run one complete simulation and time its setup and training phases.

    The random seed is reset before each run so that every repetition performs
    exactly the same amount of work, making the timing comparison meaningful.

    Args:
        n_nodes (int): Number of mesh nodes along the beam.
        n_epochs (int): Number of training epochs.
        quad_cls (type): Quadrature-rule class to instantiate.
        device (torch.device): Device on which to run the simulation.
        seed (int): Seed used to initialise the global RNG before the run.

    Returns:
        PhaseTimings: Setup and training durations of this run, in seconds.
    """
    torch.manual_seed(seed)

    t0 = time.perf_counter()
    model, optimizer = build_model(n_nodes, quad_cls, device)
    _sync(device)
    setup = time.perf_counter() - t0

    t1 = time.perf_counter()
    train(model, optimizer, n_epochs)
    _sync(device)
    train_time = time.perf_counter() - t1

    return PhaseTimings(setup=setup, train=train_time)


def summarize(name: str, samples: list[float]) -> dict:
    """Compute summary statistics for a list of timing samples.

    Args:
        name (str): Label of the phase being summarised (e.g. ``"setup"``).
        samples (list[float]): Per-run durations in seconds.

    Returns:
        dict: A mapping with keys ``name``, ``mean``, ``std``, ``min``, ``max``
        and ``median``, all durations expressed in seconds. ``std`` is the
        sample standard deviation, or ``0.0`` when fewer than two samples exist.
    """
    return {
        "name": name,
        "mean": statistics.mean(samples),
        "std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min": min(samples),
        "max": max(samples),
        "median": statistics.median(samples),
    }


def print_report(stats: list[dict], config: dict) -> None:
    """Print a formatted benchmark report to stdout.

    Args:
        stats (list[dict]): Per-phase summaries as returned by :func:`summarize`.
        config (dict): The benchmark configuration (printed as a header).
    """
    print("\nNeuROM pipeline benchmark")
    print(
        "  device={device} dtype={dtype} nodes={nodes} epochs={epochs} "
        "quadrature={quadrature}".format(**config)
    )
    print("  repeats={repeats} (after {warmup} warmup run(s))\n".format(**config))

    header = (
        f"{'phase':<8}{'mean [ms]':>12}{'std [ms]':>12}{'min [ms]':>12}{'max [ms]':>12}"
    )
    print(header)
    print("-" * len(header))
    for s in stats:
        print(
            f"{s['name']:<8}"
            f"{s['mean'] * 1e3:>12.3f}"
            f"{s['std'] * 1e3:>12.3f}"
            f"{s['min'] * 1e3:>12.3f}"
            f"{s['max'] * 1e3:>12.3f}"
        )
    print()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the benchmark.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-r", "--repeats", type=int, default=10, help="Number of measured runs."
    )
    parser.add_argument(
        "-w",
        "--warmup",
        type=int,
        default=2,
        help="Number of unmeasured warmup runs (excluded from statistics).",
    )
    parser.add_argument(
        "-n", "--nodes", type=int, default=100, help="Number of mesh nodes."
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=3, help="Number of training epochs."
    )
    parser.add_argument(
        "-q",
        "--quadrature",
        choices=sorted(QUADRATURES),
        default="midpoint",
        help="Quadrature rule to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device, e.g. 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Base RNG seed (constant across runs)."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Override torch CPU thread count for reproducible timing.",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional path to write the results (config + stats) as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the benchmark from command-line arguments and report the results."""
    args = parse_args()

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    device = torch.device(args.device)
    quad_cls = QUADRATURES[args.quadrature]

    config = {
        "device": str(device),
        "dtype": str(torch.get_default_dtype()),
        "nodes": args.nodes,
        "epochs": args.epochs,
        "quadrature": args.quadrature,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "threads": torch.get_num_threads(),
    }

    # Warmup runs absorb one-time costs (lazy init, vmap tracing) and are
    # deliberately excluded from the reported statistics.
    for _ in range(args.warmup):
        run_simulation(args.nodes, args.epochs, quad_cls, device, args.seed)

    timings: list[PhaseTimings] = []
    for i in range(args.repeats):
        timings.append(
            run_simulation(args.nodes, args.epochs, quad_cls, device, args.seed)
        )
        print(f"  run {i + 1}/{args.repeats} done", end="\r")

    stats = [
        summarize("setup", [t.setup for t in timings]),
        summarize("train", [t.train for t in timings]),
        summarize("total", [t.total for t in timings]),
    ]

    print_report(stats, config)

    if args.json is not None:
        payload = {
            "config": config,
            "stats": stats,
            "runs": [asdict(t) | {"total": t.total} for t in timings],
        }
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Results written to {args.json}")


if __name__ == "__main__":
    main()
