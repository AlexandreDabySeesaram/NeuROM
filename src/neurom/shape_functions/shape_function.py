"""Abstract base class for element shape functions."""

from abc import ABC, abstractmethod
import torch

from neurom.math.jacobian import jacobian
from neurom.reference_cell import ReferenceCell


class ShapeFunction(ABC):
    r"""Abstract base class for finite-element shape functions.

    Subclasses implement the ``N`` method for a specific element type and polynomial order.
    A ``ShapeFunction`` is a plain immutable object (not an ``nn.Module``): :math:`N(\xi)` is
    computed directly from the input :math:`\xi`, so it carries no dtype/device state and a single
    instance is shared across meshes.

    The DOF layout (which DOF is a value, a derivative, etc., and where it lives)
    is owned by the :class:`~neurom.elements.FiniteElement`, not the shape function;
    a ``ShapeFunction`` only evaluates the reference basis :math:`N(\xi)`.

    Attributes:
        reference_cell (ReferenceCell): The reference element on which
            the shape function is defined.
    """

    def __init__(self, reference_cell: ReferenceCell):
        """Initialise the shape function with its reference element.

        Args:
            reference_cell (ReferenceCell): The reference element on
                which the shape function is defined.
        """
        self.reference_cell = reference_cell

    @abstractmethod
    def N(self, xi: torch.Tensor) -> torch.Tensor:
        """Evaluate shape functions at reference coordinates.

        Args:
            xi (torch.Tensor): Reference coordinates, tensor of shape
                ``(N_e, N_q, dim_ref)``.

        Returns:
            torch.Tensor: Shape-function values of shape
            ``(N_e, N_q, N_nodes)``.
        """
        pass

    def grad_N(self, xi: torch.Tensor) -> torch.Tensor:
        r"""Reference gradient of the basis: :math:`\partial N_a/\partial\xi_j`.

        Default implementation differentiates :meth:`N` with autograd (so it holds for any basis,
        including high-order ones nonlinear in :math:`\xi`); a subclass with a known closed form may
        override it. ``xi`` is detached and re-marked ``requires_grad`` internally -- the reference
        gradient depends only on :math:`\xi`, never on node positions -- so callers need not manage
        the graph.

        Args:
            xi (torch.Tensor): Reference coordinates, shape ``(N_e, N_q, dim_ref)``.

        Returns:
            torch.Tensor: :math:`\partial N/\partial\xi`, shape ``(N_e, N_q, n_ref, dim_ref)``.
        """
        xi = xi.detach().requires_grad_(True)
        return jacobian(xi, self.N(xi))  # (N_e, N_q, n_ref, dim_ref)
