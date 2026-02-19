import pytest
import torch
import torch.nn as nn

# Import library modules
from neurom.constraints import NoConstraint, Dirichlet
from neurom.field import Field
from neurom.meshes import Topology

torch.set_default_dtype(torch.float32)


@pytest.fixture
def setup():
    """
    Prepare what is needed to define a Field:
    * name = "test"
    * Simple topology: 3 elements with 4 nodes.
    * initial values, [3., 7., 6., -5.]
    """
    name = "test"
    N = 4
    nodes = torch.arange(0, N)
    elements = torch.vstack([torch.arange(0, N - 1), torch.arange(1, N)]).T
    topology = Topology(nodes, elements)
    init_values = torch.tensor([3.0, 7.0, 6.0, -5.0])

    return ("test", topology, init_values)


class TestField:
    """
    Test Field class

    Attributes:
        relative_tolerance (float): The relative tolerance used to compare floats.
    """

    relative_tolerance: float = 1e-9

    def test_construct_non_trainable_field_with_no_constraint(self, setup):
        """
        Test construction of a non-trainable field with NoConstraint constraint
        """
        # Create field
        name = setup[0]
        topology = setup[1]
        init_values = setup[2]
        field = Field(
            name="test",
            topology=topology,
            init_values=init_values,
            constraint=NoConstraint(),
            trainable=False,
        )

        assert field.name == name
        assert isinstance(field.constraint, NoConstraint)

        assert "values_reduced" in field._buffers
        assert isinstance(getattr(field, "values_reduced"), torch.Tensor)
        assert field.values_reduced == pytest.approx(
            init_values, rel=self.relative_tolerance
        )

        dofs_free_expected = torch.ones(topology.n_nodes, dtype=torch.bool)
        assert field.dofs_free == pytest.approx(
            dofs_free_expected, rel=self.relative_tolerance
        )

    def test_construct_trainable_field_with_no_constraint(self, setup):
        """
        Test construction of a trainable field with NoConstraint constraint
        """
        # Create field
        name = setup[0]
        topology = setup[1]
        init_values = setup[2]
        field = Field(
            name="test",
            topology=topology,
            init_values=init_values,
            constraint=NoConstraint(),
            trainable=True,
        )

        assert field.name == name
        assert isinstance(field.constraint, NoConstraint)

        assert isinstance(field.values_reduced, nn.Parameter)
        assert field.values_reduced.detach() == pytest.approx(
            init_values, rel=self.relative_tolerance
        )

        dofs_free_expected = torch.ones(topology.n_nodes, dtype=torch.bool)
        assert field.dofs_free == pytest.approx(
            dofs_free_expected, rel=self.relative_tolerance
        )

    # def test_construct_non_trainable_field_with_dirichlet_constraint(self, setup):
    #    """
    #    Test construction of a non-trainable field with DirichleDirichlett constraint
    #    """
    #    # Create field
    #    name = setup[0]
    #    topology = setup[1]
    #    init_values = setup[2]
    #    field = Field(
    #        name="test",
    #        topology=topology,
    #        init_values=init_values,
    #        constraint=Dirichlet(),
    #        trainable=False,
    #    )

    #    assert field.name == name
    #    assert isinstance(field.constraint, Dirichlet)

    #    assert "values_reduced" in field._buffers
    #    assert isinstance(getattr(field, "values_reduced"), torch.Tensor)
    #    assert field.values_reduced == pytest.approx(
    #        init_values, rel=self.relative_tolerance
    #    )

    #    dofs_free_expected = torch.ones(topology.n_nodes, dtype=torch.bool)
    #    assert field.dofs_free == pytest.approx(
    #        dofs_free_expected, rel=self.relative_tolerance
    #    )

    # def test_construct_trainable_field_with_dirichlet_constraint(self, setup):
    #    """
    #    Test construction of a trainable field with Dirichlet constraint
    #    """
    #    # Create field
    #    name = setup[0]
    #    topology = setup[1]
    #    init_values = setup[2]
    #    field = Field(
    #        name="test",
    #        topology=topology,
    #        init_values=init_values,
    #        constraint=NoConstraint(),
    #        trainable=True,
    #    )

    #    assert field.name == name
    #    assert isinstance(field.constraint, NoConstraint)

    #    assert isinstance(field.values_reduced, nn.Parameter)
    #    assert field.values_reduced.detach() == pytest.approx(
    #        init_values, rel=self.relative_tolerance
    #    )

    #    dofs_free_expected = torch.ones(topology.n_nodes, dtype=torch.bool)
    #    assert field.dofs_free == pytest.approx(
    #        dofs_free_expected, rel=self.relative_tolerance
    #    )
