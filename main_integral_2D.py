#%% Libraries import
# import HiDeNN library
from HiDeNN_PDE import MeshNN_2D
# Import pre-processing functions
import Bin.Pre_processing as pre
# Import torch librairies
import torch

from Bin.PDE_Library import Strain, Stress, InternalEnergy_2D
from Bin.Training import Training_2D_Integral

#Import post processing libraries
import Post.Plots as Pplot
import time
import os
import numpy as numpy

Lame_coeff = False

if Lame_coeff:
    lmbda = 1.25                                                # First Lamé's coefficient
    mu = 1.0                                                    # Second Lamé's coefficient
    E = mu*(3*lmbda+2*mu)/(lmbda+mu)                            # Young's modulus
    nu = lmbda/(2*(lmbda+mu))                                   # Poisson's ratio
else:
    E = 175                                                     # Young's modulus (GPa)
    nu = 0.3                                                    # Poisson's ratio
    lmbda = (E*nu)/((1+nu)*(1-2*nu))                            # First Lamé's coefficient
    mu = E/(2*(1+nu))                                           # Second Lamé's coefficient


order = 1                                                       # Order of the FE interpolation
dimension = 2                                                   # Dimension of the problem

MaxElemSize = 5

Domain_mesh = pre.Mesh('Rectangle',MaxElemSize, order, dimension)    # Create the mesh object
Volume_element = 100                               # Volume element correspond to the 1D elem in 1D


DirichletDictionryList = [  {"Entity": 111, "Value": 0, "Normal": 1, "Relation": False, "Constitutive": False},
                            {"Entity": 111, "Value": -1, "Normal": 0, "Relation": False, "Constitutive": False},
                            {"Entity": 113, "Value": 1, "Normal": 1, "Relation": False, "Constitutive": False},
                            {"Entity": 113, "Value": 1, "Normal": 0, "Relation": False, "Constitutive": False}
                            ]

Excluded_elements = []
Domain_mesh.AddBCs(Volume_element, Excluded_elements, DirichletDictionryList)           # Include Boundary physical domains infos (BCs+volume)
Domain_mesh.MeshGeo()                                # Mesh the .geo file if .msh does not exist
Domain_mesh.ReadMesh()                               # Parse the .msh file
Domain_mesh.ExportMeshVtk()



Model_2D = MeshNN_2D(Domain_mesh, 2)                # Create the associated model (with 2 components)
Model_2D.UnFreeze_Values()
Model_2D.Freeze_Mesh()


# Get plotcoordinates 
L = 10                                              # Length of the Beam

n_visu = 10                                         # Sample in smallest direction for post-processing
TrailCoord_1d_x = torch.tensor([i for i in torch.linspace(0,L,n_visu)],dtype=torch.float64, requires_grad=True)
TrailCoord_1d_y = torch.tensor([i for i in torch.linspace(0,5*L,5*n_visu)],dtype=torch.float64,  requires_grad=True)
PlotCoordinates = torch.cartesian_prod(TrailCoord_1d_x,TrailCoord_1d_y)
IDs_plot = torch.tensor(Domain_mesh.GetCellIds(PlotCoordinates),dtype=torch.int)
Model_2D.eval()
u_predicted = Model_2D(PlotCoordinates, IDs_plot)

Pplot.Plot2Dresults(u_predicted, PlotCoordinates , "_u_init_integral_fine")

Model_2D.train()
List_elems = torch.range(0,Domain_mesh.NElem-1,dtype=torch.int)
u_predicted,xg,detJ = Model_2D(PlotCoordinates, List_elems)

Model_2D.eval()
u_predicted_eval = Model_2D(xg, List_elems)
Pplot.Plot2Dresults(u_predicted_eval, xg , "_u_init_integral_Gauss")

We = torch.sum(InternalEnergy_2D(u_predicted_eval,xg,lmbda, mu)*detJ)
print(We)
learning_rate = 0.001                              # optimizer learning rate


#%% Training 
optimizer = torch.optim.Adam(Model_2D.parameters(), lr=learning_rate)
n_epochs = 10000
Loss_vect, Duration, U_interm = Training_2D_Integral(Model_2D, optimizer, n_epochs,List_elems,lmbda, mu)

#%% Evaluate trained model
u_predicted = Model_2D(PlotCoordinates, IDs_plot)
Pplot.Plot2Dresults(u_predicted, PlotCoordinates , "_u_final_fine")

#%% Get nodal values from the trained model
u_x = [u for u in Model_2D.nodal_values_x]
u_y = [u for u in Model_2D.nodal_values_y]

#%% Compute the strain 
eps =  Strain(Model_2D(xg, List_elems),xg)

#%% Export the results to vtk 
import meshio
#%% Read mesh
meshBeam = meshio.read('geometries/'+Domain_mesh.name_mesh)
u = torch.stack([torch.cat(u_x),torch.cat(u_y)],dim=1)
sol = meshio.Mesh(meshBeam.points, {"triangle":meshBeam.cells_dict["triangle"]},
point_data={"U":u.data}, 
cell_data={"eps": [eps.data]}, )
sol.write(
    "Results/sol_u.vtk", 
)

#%% Export intermediate convergence steps
for timestep in range(len(U_interm)):
    sol = meshio.Mesh(meshBeam.points, {"triangle":meshBeam.cells_dict["triangle"]},
    point_data={"U":U_interm[timestep]})

    sol.write(
        f"Results/Video/TimeSeries/sol_u_{timestep}.vtk",  
    )