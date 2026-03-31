from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Import library modules
from neurom.quadratures import MidPoint1D, TwoPoints1D
from neurom.shape_functions import LinearSegment
from neurom.geometry import IsoparametricMapping1D
from neurom.meshes import Topology, Mesh
from neurom.constraints import Dirichlet
from neurom.fields import Field, TrainableField
from neurom.field_layout import FieldLayout
from neurom.interpolation import PointWiseInterpolator, Interpolator, FieldInterpolator

from neurom.physics import ElasticEnergy, LoadPotential
from neurom.physics_loss import PhysicsLoss
from neurom.fem_model import FEMModel

from generate_mesh import generate_plate_with_hole

torch.set_default_dtype(torch.float32)


def main():
    generate_plate_with_hole("mesh", 1.0, 1.0, 0.01)


if __name__ == "__main__":
    main()
