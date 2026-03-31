import argparse
from pathlib import Path
import meshio
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

from read_mesh import read_mesh
from write_mesh import write_mesh

torch.set_default_dtype(torch.float32)


def main():
    parser = argparse.ArgumentParser(description="2d case simulation.")
    parser.add_argument(
        "-i",
        "--input-mesh",
        type=Path,
        default="./plate_with_hole.xdmf",
        help="Mesh in xdmf format.",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default="./",
        help="Output directory where the files will be written.",
    )

    # Get arguments
    args = parser.parse_args()
    fname = args.input_mesh
    output_dir = args.output_dir

    # Read mesh
    topology, data = read_mesh(fname)

    # Get positions
    points = data["x"]
    N = points.shape[0]

    # Get gmsh:dim_tags
    dim_tags = data["point_data"]["gmsh:dim_tags"]
    bc_mask = torch.isin(dim_tags[:, 0], torch.tensor([0, 1]))
    breakpoint()
    nodes_bc = topology.nodes[bc_mask]
    n_mask = nodes_bc.shape[0]

    # Initialize displacement value
    u_init = 0.5 * torch.ones(N, 1)

    # Define constant load
    load_value = 1000.0
    load = load_value * torch.ones(N, 1)

    # Define shape function to use
    sf = LinearSegment()
    # Define quadrature method
    quad = TwoPoints1D()
    # Define mapping (for positions only)
    mapping = IsoparametricMapping1D(sf)

    # Prepare Field layout and fill it with actual fields
    field_layout = FieldLayout()

    # Displacement
    u = field_layout.add(
        TrainableField(
            name="displacement",
            topology=topology,
            init_values=u_init,
            constraint=Dirichlet(nodes=nodes_bc, values_imposed=torch.zeros(n_mask, 1)),
        )
    )

    # Positions
    x = field_layout.add(Field(name="positions", topology=topology, values=points))

    # Load
    f = field_layout.add(Field(name="load", topology=topology, values=load))

    # Generate mesh
    mesh = Mesh(topology=topology, nodes_positions=x)

    # Write mesh
    fname = output_dir / "result.xdmf"
    write_mesh(fname, mesh, field_layout)


if __name__ == "__main__":
    main()
