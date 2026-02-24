import pytest
import torch
import torch.nn as nn

# Import library modules
from neurom.constraints import NoConstraint

torch.set_default_dtype(torch.float32)


class TestNoConstraint:
    """
    Test NoConstraint class

    Attributes:
        relative_tolerance (float): The relative tolerance used to compare floats.
    """

    relative_tolerance: float = 1e-9

    def test_get_dofs_free(self):
        """
        Test the dofs_free mask generated is correct.

        It should be set to true for all ids.
        """
        reduced_values = torch.tensor([5, 6.25, 7.5, 8.75, 10.0])
        n_nodes = reduced_values.shape[0]
        constraint = NoConstraint()

        dofs_free = constraint.get_dofs_free(n_nodes)
        assert dofs_free.dtype is torch.bool

        # Check values
        dofs_free_expected = torch.ones(n_nodes, dtype=torch.bool)
        assert torch.equal(dofs_free, dofs_free_expected)

    def test_expand(self):
        """
        Test the expand() method that expand the free values of a field with the imposed ones of the constraint.

        For NoConstraint, it should return the same exact field values.
        """
        reduced_values = torch.tensor([5, 6.25, 7.5, 8.75, 10.0])
        n_nodes = reduced_values.shape[0]
        dofs_free = torch.ones(n_nodes, dtype=torch.bool)

        constraint = NoConstraint()
        values = constraint.expand(reduced_values, dofs_free)

        # Check values
        assert values == pytest.approx(reduced_values, rel=self.relative_tolerance)
