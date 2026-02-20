import pytest
import torch
import torch.nn as nn

# Import library modules
from neurom.constraints import Dirichlet

torch.set_default_dtype(torch.float32)


@pytest.fixture
def dirichlet():
    """
    Prepare what is needed to define a Dirichlet constraint:
    * nodes: [0, 2, 4]
    * Values: [3., 7., 6.]
    """
    name = "test"
    nodes = torch.tensor([0, 2, 4], dtype=int)
    values_imposed = torch.tensor([3.0, 7.0, 6.0])
    bc = Dirichlet(nodes=nodes, values_imposed=values_imposed)

    return bc


class TestDirichlet:
    """
    Test Dirichlet class

    Attributes:
        relative_tolerance (float): The relative tolerance used to compare floats.
    """

    relative_tolerance: float = 1e-9

    def test_construction(self, dirichlet):
        """
        Test construction of a Dirichlet constraint.
        """
        assert "nodes" in dirichlet._buffers
        assert isinstance(getattr(dirichlet, "nodes"), torch.Tensor)

        expected_nodes = torch.tensor([0, 2, 4], dtype=int)
        assert dirichlet.nodes == pytest.approx(
            expected_nodes, rel=self.relative_tolerance
        )

        assert "values_imposed" in dirichlet._buffers
        assert isinstance(getattr(dirichlet, "values_imposed"), torch.Tensor)

        expected_values = torch.tensor([3.0, 7.0, 6.0])
        assert dirichlet.values_imposed == pytest.approx(
            expected_values, rel=self.relative_tolerance
        )

    def test_get_dofs_free(self, dirichlet):
        """
        Test the dofs_free mask generated is correct.

        It should be set to true for all nodes but the ones owned by the Dirichlet constraint.
        """
        n_nodes = 6
        dofs_free = dirichlet.get_dofs_free(n_nodes)
        assert dofs_free.dtype is torch.bool

        # Check values
        dofs_free_expected = torch.tensor(
            [False, True, False, True, False, True], dtype=torch.bool
        )
        assert torch.equal(dofs_free, dofs_free_expected)

    def test_expand(self, dirichlet):
        """
        Test the expand() method that expand the free values of a field with the imposed ones of the constraint.

        For Dirichlet, it should it should contain the original values for all but the Dirichlet nodes.
        """
        reduced_values = torch.tensor([5, 6.25, 7.5, 8.75, 10.0, 11.25])
        n_nodes = reduced_values.shape[0]
        dofs_free = torch.tensor(
            [False, True, False, True, False, True], dtype=torch.bool
        )

        values = dirichlet.expand(reduced_values, dofs_free)

        # Check values
        # Expected values are Dirichlet values for nodes 0, 2, 4 and original values otherwise.
        expected_values = torch.tensor([3.0, 6.25, 7.0, 8.75, 6.0, 11.25])
        assert values == pytest.approx(expected_values, rel=self.relative_tolerance)
