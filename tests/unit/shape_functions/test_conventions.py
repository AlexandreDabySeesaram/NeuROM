"""Convention-locking tests for reference elements and shape functions.

These tests pin the node-ordering convention shared by reference elements,
barycentric coordinates and shape functions: barycentric coordinate
``lambda_i`` must be associated with node ``i`` of the reference simplex, i.e.

    N(barycentric_to_reference(lambda)) == lambda

for arbitrary (non-symmetric) barycentric points. Any new element type must
satisfy this invariant.
"""

import pytest
import torch

from neurom.geometry import barycentric_to_reference
from neurom.shape_functions import LinearBar, LinearTriangle

torch.set_default_dtype(torch.float32)

relative_tolerance = 1e-6


@pytest.mark.parametrize(
    "sf,barycentric",
    [
        # Non-symmetric barycentric points so a node permutation cannot hide
        (LinearBar(), torch.tensor([[0.7, 0.3], [0.1, 0.9]])),
        (
            LinearTriangle(),
            torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [1.0, 0.0, 0.0]]),
        ),
    ],
)
def test_shape_functions_match_barycentric_node_ordering(sf, barycentric):
    """
    Test that shape function node ordering matches the reference simplex.

    For each barycentric point, mapping to reference coordinates and
    evaluating the shape functions must give back the barycentric weights.
    """
    xi = barycentric_to_reference(barycentric, sf.reference_element)  # (N_q, dim)

    # Shape functions expect (N_e, N_q, dim)
    N = sf.N(xi.unsqueeze(0))  # (1, N_q, N_nodes)

    assert N.squeeze(0).detach().numpy() == pytest.approx(
        barycentric.numpy(), rel=relative_tolerance
    )


@pytest.mark.parametrize("sf", [LinearBar(), LinearTriangle()])
def test_shape_functions_partition_of_unity(sf):
    """
    Test that the shape functions sum to one anywhere in the element.
    """
    barycentric = torch.tensor(
        [[0.25, 0.75], [0.5, 0.5]]
        if sf.reference_element.simplex.shape[0] == 2
        else [[0.25, 0.5, 0.25], [0.1, 0.2, 0.7]]
    )
    xi = barycentric_to_reference(barycentric, sf.reference_element)
    N = sf.N(xi.unsqueeze(0))

    assert N.sum(dim=-1).detach().numpy() == pytest.approx(
        torch.ones(1, N.shape[1]).numpy(), rel=relative_tolerance
    )
