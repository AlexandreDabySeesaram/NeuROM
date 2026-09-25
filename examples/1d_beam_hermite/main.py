import torch


# Import library modules
from neurom.quadratures import TwoPoints1D
from neurom.shape_functions import HermiteBeam, LinearBar
from neurom.meshes import Connectivity
from neurom.geometry import IsoparametricMapping1D
from neurom.meshes import Mesh
from neurom.constraints import Dirichlet
from neurom.fields import Field, TrainableField
from neurom.field_layout import FieldLayout
from neurom.interpolation import (
    QuadratureContext,
    QuadratureAssembly,
    IntegrationDomain,
)
from neurom.physics import SolidElasticEnergy, MembraneStretchEnergy
from neurom.physics.tensors import (
    green_lagrange_strain,
    linear_elastic_stress_point,
)
from neurom.physics_loss import PhysicsLoss
from neurom.fem_model import FEMModel
from neurom.math import jacobian, second_derivative


def main():

    # Mesh, shape functions, quadrature, mapping, boundary conditions. 
    ##################################################################
    
    # Prepare Field layout and fill it with actual fields
    field_layout = FieldLayout()

    ## Mesh

    # Dimensions
    x_min = 0.0
    x_max = 1.0
    N = 5

    # Generate vertices and connectivity
    x_array = torch.linspace(x_min, x_max, N).unsqueeze(-1)

    nodes = torch.arange(0, N)
    elements = torch.vstack([torch.arange(0, N - 1), torch.arange(1, N)]).T # [e, e+1] for e in range(N-1)

    connectivity_geom = Connectivity(nodes, elements)

    # Positions
    x = field_layout.add(
        Field(name="positions", connectivity=connectivity_geom, values=x_array)
    )

    # Generate mesh
    mesh = Mesh(connectivity=connectivity_geom, nodes_positions=x)

     # Define mapping (for positions only)
    sf_geom = LinearBar() 
    mapping_geom = IsoparametricMapping1D(sf_geom, mesh) 


    ## Field
    # Initialize displacement values
    u_init = 0.5 * torch.ones(2*N, 1) # 2 DOFs per node (w and its slope in xi, i.e. (h/2)*w')

    nodes = torch.arange(0, 2* N)
    elements = torch.vstack([
    torch.arange(0, 2*N - 2, 2),   # 0, 2, ..., 2N-4
    torch.arange(1, 2*N - 1, 2),   # 1, 3, ..., 2N-3
    torch.arange(2, 2*N,     2),   # 2, 4, ..., 2N-2
    torch.arange(3, 2*N + 1, 2),   # 3, 5, ..., 2N-1
    ]).T

    connectivity_field = Connectivity(nodes, elements)

    # Boundary conditions
    nodes_u_bc = [0, 1, 2*N -2, 2*N - 1]
    u_bc = torch.zeros(4, 1)


    # Displacement
    u = field_layout.add(
        TrainableField(
            name="displacement",
            connectivity=connectivity_field,
            init_values=u_init,                         #used to initialize the values # self.dim = init_values.shape[1]
            constraint=Dirichlet(nodes=nodes_u_bc, values_imposed=u_bc),
        )
    )

    # sf and mapping interpolation ?
    sf_field = HermiteBeam()

    # Quadrature strategy: two Gauss points per element.
    quad = TwoPoints1D()

    # Define interpolation at quadrature
    ctx = QuadratureContext(mesh, quad, mapping_geom)
    assembly_u = QuadratureAssembly(ctx, sf_field, u)
    domain = IntegrationDomain([assembly_u])


    # Energy.
    #########################################################

    # Bending
    bending_energy = SolidElasticEnergy(u, strain=second_derivative, stress_point=lambda kappa: kappa)

    # Axial load (non-dimensional P, first clamped-clamped critical load is 4*pi^2)
    P = 0.5 * 4 * torch.pi**2

    # Axial load potential: -P/2 * int w'^2
    axial_loading_energy = SolidElasticEnergy(u, strain=jacobian, stress_point=lambda eps: -P * eps)

    # Membrane stretching: 1/8 * (int w'^2)^2

    # Stretch: integrand w'^2 dx (SolidElasticEnergy has a built-in 1/2, hence 2 * eps)
    stretch = SolidElasticEnergy(u, strain=jacobian, stress_point=lambda eps: 2 * eps)

    membrane_stretching_energy = MembraneStretchEnergy(stretch)

    # Sum the pieces
    energy = bending_energy + axial_loading_energy + membrane_stretching_energy
    #Define loss
    loss = PhysicsLoss(energy, field_layout)


    # FEM Model.
    #########################################################

    # Define FEM model
    model = FEMModel(
        mesh=mesh,
        field_layout=field_layout,
        integration_domain=domain,
        loss=loss,
    )



    # Optimizer.
    #########################################################
    
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-1, #line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        loss = model()
        if loss.requires_grad:
            loss.backward()
        return loss

    loss_history = []

    print("* Training")
    n_epochs = 1000
    for i in range(n_epochs):
        print(i)
        loss = optimizer.step(closure)
        loss_history.append(loss.item())
        print(f"i = {i} loss={loss.item():.3e}",)


if __name__ == "__main__":
    main()