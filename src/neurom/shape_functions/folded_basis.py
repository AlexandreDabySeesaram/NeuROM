r"""A coded reference basis with the duality correction :math:`C` folded into its tabulation.

A finite element may be declared with a *non-nodal* coded basis :math:`\hat\Phi` (hierarchical,
Bernstein, monomials). The element then computes :math:`C = V^{-1}` once at construction
(:math:`V_{ab} = \hat\ell_a(\hat\Phi_b)`, the reference functionals applied to the coded basis) and
presents the *nodal* basis

.. math::

    \hat H_a(\xi) = \sum_k C_{ka}\,\hat\Phi_k(\xi)

to everything downstream (assembly, interpolation, the DOF transformation). So :math:`C` is a
construction-time step folded into the tabulation, never a per-cell or per-forward factor -- the
runtime DOF transformation stays :math:`M_e = G_e\,T_e`. Derivatives fold automatically: the fold is
a differentiable matmul that autograd traces through, so no separate derivative tabulation is needed.
"""

import torch

from neurom.shape_functions.shape_function import ShapeFunction


class FoldedBasis(ShapeFunction):
    r"""A coded basis :math:`\hat\Phi` presented as the nodal basis :math:`\hat H = \hat\Phi\,C`.

    Args:
        coded (ShapeFunction): The coded (non-nodal) reference basis :math:`\hat\Phi`.
        C (torch.Tensor): The duality correction :math:`C = V^{-1}`, shape ``(n_ref, n_dofs)``. Held
            in its construction dtype (float64) and cast to the inputs at tabulation, so the folded
            basis is dtype/device-agnostic like every other reference tensor.

    Attributes:
        coded (ShapeFunction): The wrapped coded basis.
        C (torch.Tensor): The duality correction.
    """

    def __init__(self, coded: ShapeFunction, C: torch.Tensor) -> None:
        super().__init__(coded.reference_cell)
        self.coded = coded
        self.C = C

    def N(self, xi: torch.Tensor) -> torch.Tensor:
        r"""Folded basis values :math:`\hat H(\xi) = \hat\Phi(\xi)\,C`.

        Args:
            xi (torch.Tensor): Reference coordinates, shape ``(N_e, N_q, dim_ref)``.

        Returns:
            torch.Tensor: Folded (nodal) basis values, shape ``(N_e, N_q, n_dofs)``.
        """
        phi = self.coded.N(xi)  # (N_e, N_q, n_ref)
        return phi @ self.C.to(dtype=phi.dtype, device=phi.device)  # (N_e, N_q, n_dofs)
