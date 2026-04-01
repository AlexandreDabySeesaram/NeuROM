import argparse
from pathlib import Path
import meshio
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Import library modules
from neurom.quadratures import MidPoint2D
from neurom.shape_functions import LinearTriangle
from neurom.geometry import IsoparametricMapping2D
from neurom.meshes import Topology, Mesh
from neurom.constraints import Dirichlet
from neurom.fields import Field, TrainableField
from neurom.field_layout import FieldLayout
from neurom.interpolation import Interpolator, FieldInterpolator
from neurom.physics import SolidElasticEnergy
from neurom.physics.tensors import *
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

    lame_lambda = 1.25
    lame_mu = 1.0

    # Read mesh
    topology, data = read_mesh(fname)

    # Get positions - restrict to 2D
    points = data["x"][:, 0:2].to(torch.float32)
    N = points.shape[0]

    # Set boundary conditions
    # Tags: 3 == top, 1 == bottom
    dim_tags = data["point_data"]["gmsh:dim_tags"]
    mask_top = torch.logical_and(dim_tags[:, 1] == 3, dim_tags[:, 0] == 1)
    mask_bottom = torch.logical_and(dim_tags[:, 1] == 1, dim_tags[:, 0] == 1)
    nodes_top = topology.nodes[mask_top]
    nodes_bottom = topology.nodes[mask_bottom]
    u_top = torch.tensor([0.0, -1.0]).expand(nodes_top.shape[0], 2)
    u_bottom = torch.tensor([0.0, 0.0]).expand(nodes_bottom.shape[0], 2)
    nodes_u_bc = torch.cat([nodes_top, nodes_bottom])
    u_bc = torch.cat([u_top, u_bottom])

    # Initialize displacement value
    u_init = 0.1 * torch.ones(N, 2)

    # Define shape function to use
    sf = LinearTriangle()
    # Define quadrature method
    quad = MidPoint2D()
    # Define mapping (for positions only)
    mapping = IsoparametricMapping2D(sf)

    # Prepare Field layout and fill it with actual fields
    field_layout = FieldLayout()

    # Displacement
    u = field_layout.add(
        TrainableField(
            name="displacement",
            topology=topology,
            init_values=u_init,
            constraint=Dirichlet(nodes=nodes_u_bc, values_imposed=u_bc),
        )
    )

    # Positions
    x = field_layout.add(Field(name="positions", topology=topology, values=points))

    # Generate mesh
    mesh = Mesh(topology=topology, nodes_positions=x)

    # Write init mesh with all fields
    fname = output_dir / "init.xdmf"
    write_mesh(fname, mesh, field_layout)

    # Define interpolator
    interpolator = Interpolator(
        mesh,
        quad,
        mapping,
        [FieldInterpolator(sf, u)],
    )

    def linear_elastic_stress_capture(strain):
        return linear_elastic_stress(strain, lame_lambda, lame_mu)

    # Define physics to solve
    physics = SolidElasticEnergy(
        field=u,
        strain=green_lagrange_strain,
        constitutive_law=linear_elastic_stress_capture,
    )

    # Potential energy part of the loss
    physics_loss = PhysicsLoss(physics=physics, field_layout=field_layout)

    # Define FEM model
    model = FEMModel(
        mesh=mesh,
        field_layout=field_layout,
        interpolator=interpolator,
        loss=physics_loss,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1)
    loss_history = []

    print("* Training")
    n_epochs = 2000
    for i in range(n_epochs):
        loss = model()

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        loss_history.append(loss.item())
        print(f"{i=} loss={loss.item():.3e}", end="\r")

    x = field_layout.add(Field(name="cauchy_stress", topology=topology, values=points))

    # Write final mesh with all fields
    fname = output_dir / "result.xdmf"
    write_mesh(fname, mesh, field_layout)


if __name__ == "__main__":
    main()
