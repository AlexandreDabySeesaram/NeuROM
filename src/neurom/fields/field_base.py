from abc import ABC, abstractmethod
import torch
import torch.nn as nn

from neurom.meshes.topology import Topology


class FieldBase(nn.Module, ABC):
    def __init__(
        self,
        name: str,
        topology: Topology,
    ):
        super().__init__()

        self.name = name
        self.topology = topology

    @abstractmethod
    def full_values(self):
        """Get the full values

        Expand the reduced values over free dofs with the constrained ones.
        """
        pass

    @abstractmethod
    def at_elements(self):
        """Get the full values at elements

        Get the full values but per element following the topology connectivity.
        """
        pass
