r"""Reference-to-physical push-forward of basis *values* (the map-type strategy, B2).

The DOF-side transformation :math:`M_e` (:mod:`neurom.dof_transformations.transformation`) turns
stored physical DOFs into reference coefficients; this module applies the *value-side* factor
:math:`A_e` that turns a reference basis value into a physical one, **after** the reference
contraction :math:`\sum_a \hat q_a \hat H_a(\xi)`:

.. math::

    v_h(x) = A_e\Big(\sum_a \hat q_a \hat H_a(\xi)\Big), \qquad \xi = \phi_e^{-1}(x),

with :math:`A_e` selected by the element's :class:`~neurom.elements.MapType`:

* ``IDENTITY`` -- :math:`A_e = I` (scalar / Lagrange / Hermite);
* ``COVARIANT_PIOLA`` -- :math:`A_e = J_e^{-\top}` (preserves tangential components);
* ``CONTRAVARIANT_PIOLA`` -- :math:`A_e = \tfrac{1}{\det J_e} J_e`, with the **signed**
  :math:`\det J_e` (preserves normal fluxes w.r.t. :math:`R\,\mathbf t`).

The convention is the code's :math:`\phi_e : \hat K \to K`, :math:`J_e = \partial x/\partial\xi`
(deal.II / DOLFINx / UFL). Derivatives need no extra code on affine cells: autograd through
:math:`\xi(x)` gives, for the contravariant map, :math:`\operatorname{div} v = \frac{1}{\det J_e}
\widehat{\operatorname{div}}\,\hat v` once the values are mapped.
"""

import torch

from neurom.apply import apply
from neurom.elements.finite_element import MapType
from neurom.samplings import ElementSampling


def reference_to_physical(
    values: torch.Tensor,
    J: torch.Tensor,
    det_J: torch.Tensor,
    map_type: MapType,
) -> torch.Tensor:
    r"""Push a reference basis value forward to physical space: :math:`v = A_e\,\hat v`.

    Batch-rank-agnostic: every leading axis of ``values`` / ``J`` / ``det_J`` passes through; only
    the trailing component / matrix axes are touched.

    Args:
        values (torch.Tensor): Reference value components, shape ``(*batch, gdim)``.
        J (torch.Tensor): Per-point Jacobian :math:`J_e`, shape ``(*batch, gdim, dim_ref)``
            (square for the simplices in scope, ``gdim == dim_ref``).
        det_J (torch.Tensor): Signed :math:`\det J_e`, shape ``(*batch, 1)``.
        map_type (MapType): Which push-forward to apply.

    Returns:
        torch.Tensor: Physical value components, shape ``(*batch, gdim)``.

    Raises:
        ValueError: If ``map_type`` is not a recognised :class:`~neurom.elements.MapType`.
    """
    if map_type is MapType.IDENTITY:
        return values
    if map_type is MapType.COVARIANT_PIOLA:
        # A_e = J^{-T}: v_i = sum_j (J^{-1})_{ji} \hat v_j.
        J_inv = torch.linalg.inv(J)
        return torch.einsum("...ji,...j->...i", J_inv, values)
    if map_type is MapType.CONTRAVARIANT_PIOLA:
        # A_e = J / det J (signed): v_i = (1/det J) sum_j J_{ij} \hat v_j.
        return torch.einsum("...ij,...j->...i", J, values) / det_J
    raise ValueError(f"unknown map_type {map_type!r}")


def reference_to_physical_at(values, mapping, points, map_type, element_ids=None):
    r"""Map a value :class:`~neurom.samplings.Sampling` to physical space using ``mapping``'s geometry.

    Pulls :math:`J_e` (``jacobian_at``) and the signed :math:`\det J_e` (``det_jacobian``) from
    ``mapping`` as :class:`~neurom.samplings.Sampling` operands and combines them with ``values``
    through :func:`~neurom.apply.apply`, so the per-element :math:`\det J_e` (an
    :class:`~neurom.samplings.ElementSampling`) prefix-broadcasts over ``values``' trailing
    quadrature/point axis with no manual reshaping. A no-op (returns ``values`` untouched, no
    geometry read) for :attr:`MapType.IDENTITY`, so the identity path is bit-identical.

    Args:
        values (Sampling): Reference value components, ``f_shape == (gdim,)``, leading element axis.
        mapping (AffineSimplexMapping): Supplies ``jacobian_at`` and ``det_jacobian``.
        points (Sampling): The evaluation points (their batch drives the broadcast); passed to
            ``mapping.jacobian_at``.
        map_type (MapType): Which push-forward to apply.
        element_ids (torch.Tensor | None): Element subset for a point query, or ``None`` for all.

    Returns:
        Sampling: Physical value components, same subtype/batch as ``values``.
    """
    if map_type is MapType.IDENTITY:
        return values
    J = mapping.jacobian_at(points, element_ids)  # Sampling, f_shape (gdim, dim_ref)
    det = mapping.det_jacobian  # ElementSampling, f_shape (1,), signed
    if element_ids is not None:
        det = ElementSampling(det.values[element_ids])
    # apply prefix-broadcasts the per-element det (and J) over values' trailing point/quad axis.
    return apply(lambda v, j, d: reference_to_physical(v, j, d, map_type), values, J, det)
