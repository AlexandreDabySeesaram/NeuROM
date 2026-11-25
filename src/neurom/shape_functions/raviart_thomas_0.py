r"""Lowest-order Raviart--Thomas (RT0) vector basis on the reference triangle (H(div)).

Three **vector-valued** basis functions, one per edge, spanning
:math:`RT_0 = [P_0]^2 \oplus \mathbf x\,P_0 = \{(a_1 + c\,\xi_0,\ a_2 + c\,\xi_1)\}`. Used only through a
``CONTRAVARIANT_PIOLA`` element (:data:`neurom.elements.RT0_TRIANGLE`): the physical field value is
the pushed-forward :math:`\tfrac{1}{\det J_e} J_e \hat H`.
"""

import torch

from neurom.reference_cell import TRIANGLE
from neurom.shape_functions.shape_function import ShapeFunction


class RaviartThomas0Triangle(ShapeFunction):
    r"""RT0 vector basis on the 2-D reference triangle: 3 edge functions, each a 2-vector."""

    def __init__(self) -> None:
        """Initialise on the reference triangle."""
        super().__init__(TRIANGLE)

    def N(self, xi: torch.Tensor) -> torch.Tensor:
        r"""Evaluate the three RT0 basis functions.

        On the reference triangle with vertices :math:`\hat v_0=(0,0),\ \hat v_1=(1,0),\
        \hat v_2=(0,1)` and reference coordinates :math:`\boldsymbol\xi=(\xi_0,\xi_1)`, the
        lowest-order Raviart--Thomas basis is

        .. math::

            \hat{\mathbf H}_0(\boldsymbol\xi)=\begin{pmatrix}\xi_0\\ \xi_1-1\end{pmatrix},\qquad
            \hat{\mathbf H}_1(\boldsymbol\xi)=\begin{pmatrix}\xi_0\\ \xi_1\end{pmatrix},\qquad
            \hat{\mathbf H}_2(\boldsymbol\xi)=\begin{pmatrix}\xi_0-1\\ \xi_1\end{pmatrix}.

        Each is the classical form :math:`\hat{\mathbf H}_i=\boldsymbol\xi-\hat v_{\mathrm{opp}(i)}`
        -- :math:`\boldsymbol\xi` minus the vertex *opposite* edge :math:`i`; with the cyclic edge
        convention :math:`e_0=(0,1),\,e_1=(1,2),\,e_2=(2,0)` the opposite vertices are
        :math:`\hat v_2,\hat v_0,\hat v_1`. On the reference triangle :math:`2|\hat K|=1`, so the usual
        :math:`1/(2|\hat K|)` area factor is unity and no scaling appears.

        They are dual to the edge-flux DOFs
        :math:`\ell_i(\mathbf v)=\int_{\hat E_i}\mathbf v\cdot\hat{\mathbf n}_i\,\mathrm d\hat s`
        taken with the **outward** normal :math:`\hat{\mathbf n}_i=R\,\hat{\mathbf t}_i`, :math:`R` the
        :math:`-90^\circ` rotation (plan 3.1), giving :math:`\ell_i(\hat{\mathbf H}_j)=\delta_{ij}`
        (flux-nodal, :math:`V=I`). Each has constant divergence
        :math:`\widehat{\operatorname{div}}\hat{\mathbf H}_i=2`.

        Source: the standard RT0 element of Raviart & Thomas (1977); see Brezzi & Fortin (1991),
        Boffi--Brezzi--Fortin (2013), Ern & Guermond (2004).

        Args:
            xi (torch.Tensor): Reference coordinates, shape ``(*batch, 2)``.

        Returns:
            torch.Tensor: Vector basis values, shape ``(*batch, 3, 2)`` -- the basis-function axis
            (``n_ref = 3``, ordered by the cyclic edges ``e01, e12, e20``) then the ``gdim = 2``
            component axis.
        """
        x = xi[..., 0]
        y = xi[..., 1]
        H0 = torch.stack([x, y - 1.0], dim=-1)
        H1 = torch.stack([x, y], dim=-1)
        H2 = torch.stack([x - 1.0, y], dim=-1)
        return torch.stack([H0, H1, H2], dim=-2)  # (*batch, n_ref=3, gdim=2)

    @property
    def n_ref(self) -> int:
        """Three basis functions (the vector axis would otherwise be mistaken for ``n_ref``)."""
        return 3
