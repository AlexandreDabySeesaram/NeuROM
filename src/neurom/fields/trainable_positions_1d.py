import torch
import torch.nn as nn
import torch.nn.functional as F

from neurom.fields.field_base import FieldBase
from neurom.meshes.topology import Topology


def inv_softplus(y: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse of ``softplus``.

    ``softplus(x) = log(1 + exp(x))``; this returns ``x`` such that
    ``softplus(x) = y`` for ``y > 0``. Equal to ``log(exp(y) - 1)`` but stable
    for large ``y``.
    """
    return torch.log(-torch.expm1(-y)) + y


class TrainablePositions1D(FieldBase):
    """Trainable 1D node positions via a monotone reparametrization.

    **1D ONLY.** Node coordinates are recovered from free real parameters
    ``c`` (one per interval) as::

        delta   = softplus(c)                  # (n-1,)  strictly positive widths
        x_tilde = cat([0], cumsum(delta))      # (n,)    strictly increasing, x_tilde[0] = 0
        x       = a + (b - a) * x_tilde / x_tilde[-1]   # (n,)  x[0] = a, x[-1] = b

    ``softplus`` keeps every interval width positive so nodes never cross; the
    affine normalization pins the endpoints to ``[a, b]``. Both properties hold
    for *any* value of ``c`` — they are structural, not enforced by the
    optimizer. This relies on a total order on a line and does not generalize to
    2D/3D; the general ``Topology`` + free-position path stays the route for
    higher dimensions.

    Args:
        name (str): Field name.
        topology (Topology): Topology whose ``n_nodes`` must equal the number of
            ``initial_positions``.
        initial_positions (torch.Tensor): Shape ``(n, 1)`` or ``(n,)``, strictly
            increasing. ``a`` and ``b`` are taken from its first/last entries and
            the raw parameters are initialized so ``full_values()`` reproduces it.
    """

    def __init__(self, name: str, topology: Topology, initial_positions: torch.Tensor):
        super().__init__(name=name, topology=topology)

        pos = initial_positions.reshape(-1).to(torch.get_default_dtype())
        n = pos.shape[0]
        if n != topology.n_nodes:
            raise ValueError(
                f"initial_positions has {n} entries but topology has "
                f"{topology.n_nodes} nodes."
            )
        if not bool((pos[1:] > pos[:-1]).all()):
            raise ValueError("initial_positions must be strictly increasing.")

        self.register_buffer("a", pos[0].detach().clone())
        self.register_buffer("b", pos[-1].detach().clone())

        # Normalize increments (forward map re-normalizes, so scale is free) and
        # invert softplus so full_values() reproduces the input mesh.
        differences = pos[1:] - pos[:-1]              # (n-1,)
        increments = differences / (self.b - self.a)  # (n-1,), sum to 1
        self.coordinates = nn.Parameter(inv_softplus(increments))

    @property
    def dim(self) -> int:
        return 1

    def full_values(self) -> torch.Tensor:
        """Reparametrized coordinates, shape ``(n, 1)`` (global view)."""
        delta = F.softplus(self.coordinates)                 # (n-1,), > 0
        x_tilde = torch.cumsum(delta, dim=0)                 # (n-1,)
        zero = torch.zeros(1, dtype=x_tilde.dtype, device=x_tilde.device)
        x_tilde = torch.cat([zero, x_tilde])                 # (n,), x_tilde[0] = 0
        x = self.a + (self.b - self.a) * (x_tilde / x_tilde[-1])
        return x.reshape(-1, 1)

    def at_elements(self) -> torch.Tensor:
        """Element-local view: ``full_values()[topology.connectivity]``."""
        return self.full_values()[self.topology.connectivity]
