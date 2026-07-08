from dataclasses import dataclass

import torch
import torch.nn as nn

from neurom.fields.field import Field
from neurom.fields.trainable_field import TrainableField
from neurom.shape_functions.shape_function import ShapeFunction
from neurom.quadratures.quadrature_rule import QuadratureRule
from neurom.constraints.constraint import Constraint
from neurom.meshes.topology import Topology
from neurom.meshes.mesh import Mesh
from neurom.interpolation.quadrature_context import QuadratureContext
from neurom.interpolation.quadrature_assembly import QuadratureAssembly


@dataclass
class Axis:
    """Descriptor of one factor (coordinate direction) of a CP-PGD decomposition.

    Groups everything needed to build and interpolate the monoms living on this
    axis. The axis mesh topology is derived from ``nodes_positions`` so that the
    ``Mesh`` identity check and the monoms' ``TrainableField`` share the exact
    same ``Topology`` object.

    Attributes:
        name (str): Axis name, used as key in the separated interpolation output.
        nodes_positions (Field): Coordinates of the axis mesh nodes.
        sf (ShapeFunction): Shape function used for interpolation on this axis.
        mapping: Reference/physical mapping (e.g. IsoparametricMapping1D).
        quad (QuadratureRule): Quadrature rule for integration on this axis.
        constraint (Constraint): Constraint (boundary conditions) on this axis.
        init_values (torch.Tensor): Initial nodal values for each new monom,
            shape (n_nodes, dim).
    """

    name: str
    nodes_positions: Field
    sf: ShapeFunction
    mapping: object
    quad: QuadratureRule
    constraint: Constraint
    init_values: torch.Tensor

    @property
    def topology(self) -> Topology:
        return self.nodes_positions.topology


class CPPGD(nn.Module):
    """Canonical-polyadic PGD separated-representation model.

    Represents ``u({x_k}) = sum_m prod_k w_m^k(x_k)`` over ``l`` axes. Holds the
    monoms ``w_m^k`` as ``TrainableField`` on each axis, manages greedy mode
    enrichment, and exposes an assembled full-tensor view and a per-monom
    separated view. It computes no energy and owns no training loop.

    Args:
        axes (list[Axis]): The ordered axes of the decomposition.
        n_modes_max (int): Maximum number of modes.
        n_modes_ini (int): Number of initially active (trainable) modes.
    """

    def __init__(self, axes, n_modes_max, n_modes_ini=1):
        super().__init__()
        self.axes = list(axes)
        self.n_modes_max = n_modes_max
        self.register_buffer(
            "n_modes_truncated", torch.tensor(min(n_modes_ini, n_modes_max))
        )

        # One Mesh + QuadratureContext per axis, shared across modes.
        self._meshes = nn.ModuleList(
            [Mesh(a.topology, a.nodes_positions) for a in self.axes]
        )
        self._contexts = nn.ModuleList(
            [
                QuadratureContext(mesh, a.quad, a.mapping)
                for mesh, a in zip(self._meshes, self.axes)
            ]
        )

        # Grid of monoms: modes x axes of TrainableField.
        self.monoms = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        TrainableField(
                            name=f"{a.name}_mode{m}",
                            topology=a.topology,
                            init_values=a.init_values,
                            constraint=a.constraint,
                        )
                        for a in self.axes
                    ]
                )
                for m in range(self.n_modes_max)
            ]
        )

        # Freeze everything, then unfreeze the initially active modes.
        self.freeze_all()
        for m in range(int(self.n_modes_truncated)):
            self.unfreeze_mode(m)

    def freeze_all(self):
        """Freeze the monoms of every mode."""
        for m in range(self.n_modes_max):
            self.freeze_mode(m)

    def freeze_mode(self, m):
        """Freeze the monoms of mode ``m``."""
        for field in self.monoms[m]:
            field.values_reduced.requires_grad_(False)

    def unfreeze_mode(self, m):
        """Unfreeze the monoms of mode ``m``."""
        for field in self.monoms[m]:
            field.values_reduced.requires_grad_(True)

    def interpolate_separated(self):
        """Interpolate each active monom at its axis's quadrature points.

        Returns:
            dict[str, list[QuadratureAssemblyResult]]: axis name -> list indexed
            by mode; entry ``m`` is the interpolation of the single monom
            ``w_m^axis``. Enables writing separable energies monom by monom.
        """
        result = {}
        for k, axis in enumerate(self.axes):
            ctx = self._contexts[k]
            per_mode = []
            for m in range(int(self.n_modes_truncated)):
                assembly = QuadratureAssembly(ctx, axis.sf, self.monoms[m][k])
                per_mode.append(assembly.interpolate())
            result[axis.name] = per_mode
        return result
