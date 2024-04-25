#%% Libraries import
# import HiDeNN library
from HiDeNN_PDE import MeshNN_2D
# Import pre-processing functions
import Bin.Pre_processing as pre
# Import torch librairies
import torch
import torch.nn as nn

from Bin.PDE_Library import Strain, Stress, InternalEnergy_2D
from Bin.Training import Training_2D_Integral

#Import post processing libraries
import Post.Plots as Pplot
import time
import os
import numpy as numpy

# Name = 'Rectangle'
Name = 'Square'
# Name = 'Hole'
Name = 'Square_small'
Name = 'Hole_3'


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

Domain_mesh = pre.Mesh(Name,MaxElemSize, order, dimension)    # Create the mesh object
Volume_element = 100                               # Volume element correspond to the 1D elem in 1D


DirichletDictionryList = [  {"Entity": 111, "Value": 0, "Normal": 1, "Relation": False, "Constitutive": False},
                            {"Entity": 111, "Value": 0, "Normal": 0, "Relation": False, "Constitutive": False},
                            {"Entity": 113, "Value": 1, "Normal": 1, "Relation": False, "Constitutive": False},
                            {"Entity": 113, "Value": 0, "Normal": 0, "Relation": False, "Constitutive": False}
                            ]

Borders = [111,112,113,114,115]
Domain_mesh.AddBorders(Borders)
Excluded_elements = []
Domain_mesh.AddBCs(Volume_element, Excluded_elements, DirichletDictionryList)           # Include Boundary physical domains infos (BCs+volume)
Domain_mesh.MeshGeo()                                # Mesh the .geo file if .msh does not exist
Domain_mesh.ReadMesh()                               # Parse the .msh file
Domain_mesh.ExportMeshVtk()



Model_2D = MeshNN_2D(Domain_mesh, 2)                # Create the associated model (with 2 components)
Model_2D.UnFreeze_Values()
# Model_2D.Freeze_Mesh()
Model_2D.UnFreeze_Mesh()


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

detJ = torch.abs(detJ)

Model_2D.eval()
u_predicted_eval = Model_2D(xg, List_elems)
Pplot.Plot2Dresults(u_predicted_eval, xg , "_u_init_integral_Gauss")

We = torch.sum(InternalEnergy_2D(u_predicted_eval,xg,lmbda, mu)*detJ)
print(We)
learning_rate = 0.001                              # optimizer learning rate


#%% Training 
# Fisrt stage
max_refinment = 1
n_refinement = 0
stagnation = False
Loss_tot = []
Duration_tot = 0
U_interm_tot = []
X_interm_tot = []
Connectivity_tot = []
Cell_dict = []
d_eps_max_vect = []
eps_max_vect = []
import numpy as np
Nnodes_max = 1000
coeff_refinement = np.power((Nnodes_max/Domain_mesh.NNodes),1/max_refinment)
dmax_threshold = 1e-7
import meshio
while n_refinement < max_refinment and not stagnation:
    print(f"Refinement level: {n_refinement}")
    n_refinement +=1
    optimizer = torch.optim.Adam(Model_2D.parameters(), lr=learning_rate)
    n_epochs = 500
    if n_refinement>4:
        n_epochs = 1000
    Loss_vect, Duration, U_interm, X_interm, Connectivity_interm = Training_2D_Integral(Model_2D, optimizer, n_epochs,List_elems,lmbda, mu)
    # Save current convergence state
    Loss_tot += Loss_vect
    Duration_tot += Duration
    U_interm_tot += U_interm
    X_interm_tot += X_interm
    Connectivity_tot += Connectivity_interm
    meshBeam = meshio.read('geometries/'+Domain_mesh.name_mesh)
    Cell_dict+=[meshBeam.cells_dict["triangle"] for _ in range(len(X_interm))]
    # Cmpute max strain
    _,xg,detJ = Model_2D(PlotCoordinates, List_elems)
    Model_2D.eval()
    eps =  Strain(Model_2D(xg, List_elems),xg)
    max_eps = torch.max(eps)
    if n_refinement >1:
        d_eps_max = 2*torch.abs(max_eps-max_eps_old)/(max_eps_old+max_eps_old)
        d_eps_max_vect.append(d_eps_max.data)
        eps_max_vect.append(max_eps.data)
        max_eps_old = max_eps
        if d_eps_max<dmax_threshold:
            stagnation = True
    else:
        max_eps_old = max_eps
    if n_refinement < max_refinment and not stagnation:
        # Refine mesh
        MaxElemSize = MaxElemSize/coeff_refinement
        Domain_mesh_2 = pre.Mesh(Name,MaxElemSize, order, dimension)    # Create the mesh object
        Domain_mesh_2.AddBCs(Volume_element, Excluded_elements, DirichletDictionryList)           # Include Boundary physical domains infos (BCs+volume)
        Domain_mesh_2.AddBorders(Borders)
        Domain_mesh_2.MeshGeo()                                # Mesh the .geo file if .msh does not exist
        Domain_mesh_2.ReadMesh()                               # Parse the .msh file
        Domain_mesh_2.ExportMeshVtk()
        List_elems = torch.range(0,Domain_mesh_2.NElem-1,dtype=torch.int)
        # Initialise finer model
        Model_2D_2 = MeshNN_2D(Domain_mesh_2, 2)                # Create the associated model (with 2 components)
        
        newcoordinates = [coord for coord in Model_2D_2.coordinates]
        newcoordinates = torch.cat(newcoordinates,dim=0)
        # Update Domain_mesh vtk mesh to get correct cell Ids
        Domain_mesh.Nodes = [[i+1,X_interm[-1][i,0].item(),X_interm[-1][i,1].item(),0] for i in range(len(Domain_mesh.Nodes))]
        Domain_mesh.ExportMeshVtk(flag_update = True)
        IDs_newcoord = torch.tensor(Domain_mesh.GetCellIds(newcoordinates),dtype=torch.int)
        NewNodalValues = Model_2D(newcoordinates,IDs_newcoord) 
        if -1 in IDs_newcoord:
            print(IDs_newcoord)
            index_neg = (IDs_newcoord == -1).nonzero(as_tuple=False)
            oldcoordinates = [coord for coord in Model_2D.coordinates]
            oldcoordinates = torch.cat(oldcoordinates,dim=0)
            for ind_neg in index_neg:
                not_found_coordinates = newcoordinates[ind_neg]
                dist_vect = not_found_coordinates - oldcoordinates
                dist = torch.norm(dist_vect, dim=1)
                closest_old_nodal_value = dist.topk(1, largest=False)[1]
                NewNodalValues[0][ind_neg] = Model_2D.nodal_values_x[closest_old_nodal_value].type(torch.float64)
                NewNodalValues[1][ind_neg] = Model_2D.nodal_values_y[closest_old_nodal_value].type(torch.float64)
                # IDs_newcoord[ind_neg] = 0

        new_nodal_values_x = nn.ParameterList([nn.Parameter((torch.tensor([i[0]]))) for i in NewNodalValues.t()])
        new_nodal_values_y = nn.ParameterList([nn.Parameter(torch.tensor([i[1]])) for i in NewNodalValues.t()])
        new_nodal_values = [new_nodal_values_x,new_nodal_values_y]
        Model_2D_2.nodal_values_x = new_nodal_values_x
        Model_2D_2.nodal_values_y = new_nodal_values_y
        Model_2D_2.nodal_values = new_nodal_values
        Domain_mesh = Domain_mesh_2
        Model_2D = Model_2D_2
        Model_2D.UnFreeze_Values()
        # Model_2D.Freeze_Mesh()
        Model_2D.UnFreeze_Mesh()
        Model_2D.train()
    else:
        Model_2D.train()





#%% Export the results to vtk 
import meshio
#%% Read mesh
#%% Get nodal values from the trained model
u_x = [u for u in Model_2D.nodal_values_x]
u_y = [u for u in Model_2D.nodal_values_y]

#%% Compute the strain 
# List_elems = torch.range(0,Model_2D.NElem-1,dtype=torch.int)
# _,xg,detJ = Model_2D(PlotCoordinates, List_elems)
# Model_2D.eval()
# eps =  Strain(Model_2D(xg, List_elems),xg)
# sigma =  torch.stack(Stress(eps[:,0], eps[:,1], eps[:,2], lmbda, mu),dim=1)

X_interm_tot = [torch.cat([x_i,torch.zeros(x_i.shape[0],1)],dim=1) for x_i in X_interm_tot]

# meshBeam = meshio.read('geometries/'+Domain_mesh.name_mesh)
# # u = torch.stack([torch.cat(u_x),torch.cat(u_y)],dim=1)
u = torch.stack([torch.cat(u_x),torch.cat(u_y),torch.zeros(torch.cat(u_x).shape[0])],dim=1)

# sol = meshio.Mesh(X_interm_tot[-1].data, {"triangle":Connectivity_tot[-1].data},
# point_data={"U":u.data}, 
# cell_data={"eps": [eps.data], "sigma": [sigma.data]}, )
# sol.write(
#     "Results/Paraview/sol_u_end_training_shear_"+Name+".vtk", 
# )

#%% Export intermediate convergence steps
meshBeam = meshio.read('geometries/'+Domain_mesh.name_mesh)

U_interm_tot = [torch.cat([u,torch.zeros(u.shape[0],1)],dim=1) for u in U_interm_tot]

for timestep in range(len(U_interm_tot)):
    sol = meshio.Mesh(X_interm_tot[timestep].data, {"triangle":Connectivity_tot[timestep].data},
    point_data={"U":U_interm_tot[timestep]})

    sol.write(
        f"Results/Paraview/TimeSeries/sol_u_multiscale_autom_spliting2_"+Name+f"_{timestep}.vtk",  
    )

# %%
