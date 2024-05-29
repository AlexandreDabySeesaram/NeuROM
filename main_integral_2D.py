#%% Libraries import
from HiDeNN_PDE import MeshNN_2D
import Bin.Pre_processing as pre
import torch
import torch.nn as nn
import numpy as np
from Bin.PDE_Library import Strain, Stress, InternalEnergy_2D, VonMises
from Bin.Training import Training_2D_Integral, Training_2D_Integral_LBFGS
import Post.Plots as Pplot
import time
import os
import matplotlib.pyplot as plt


#%% Choose geometry
# Name = 'Rectangle'
Name = 'Square'
# Name = 'Hole'
# Name = 'Square_small'
# Name = 'Hole_3'
# Name = 'L_shape'
# Name = 'Square_Holes_3'

# Initialise meterial
Mat = pre.Material( flag_lame = True,                         # If True should input lmbda and mu instead
                    #coef1 = 5*1e-3,                                    # Young Modulus
                    # coef2 = 0.3                                    # Poisson's ratio
                    coef1 = 1.25,
                    coef2 = 1.0
                    )


# Create mesh
order = 1    
n_integration_points = 1
                                                   # Order of the FE interpolation
dimension = 2                                                   # Dimension of the problem
MaxElemSize = 0.125                                             # Maximum element size of the mesh
Domain_mesh = pre.Mesh(Name,MaxElemSize, order, dimension)      # Create the mesh object
Volume_element = 100                                            # Volume element

DirichletDictionryList = [  {"Entity": 111, "Value": 0, "Normal": 1, "Relation": False, "Constitutive": False},
                            {"Entity": 111, "Value": 0, "Normal": 0, "Relation": False, "Constitutive": False},
                            {"Entity": 113, "Value": 1, "Normal": 1, "Relation": False, "Constitutive": False},
                            {"Entity": 113, "Value": 0, "Normal": 0, "Relation": False, "Constitutive": False}
                            ]

Borders = [111,112,113,114,115]                                 # List of the structure's boundaries
Domain_mesh.AddBorders(Borders)

Excluded_elements = []                                          # Element to exclude from the boundaries
Domain_mesh.AddBCs( Volume_element, 
                    Excluded_elements, 
                    DirichletDictionryList)                     # Include Boundary physical domains infos (BCs+volume)
Domain_mesh.MeshGeo()                                           # Mesh the .geo file if .msh does not exist
Domain_mesh.ReadMesh()                                          # Parse the .msh file
Domain_mesh.ExportMeshVtk()

# Create model
n_component = 2                                                 # Number of components of the interpolated field
Model_2D = MeshNN_2D(Domain_mesh, n_component, n_integration_points)                  # Create the associated model (with 2 components)
Model_2D.UnFreeze_FEM()
# Model_2D.UnFreeze_Mesh()
Model_2D.Freeze_Mesh()


# Get plotcoordinates 
L = 10                                                          # Length of the Beam
n_visu = 10                                                     # Sample in smallest direction for post-processing
TrailCoord_1d_x = torch.tensor([i for i in torch.linspace(0,L,n_visu)],dtype=torch.float64, requires_grad=True)
TrailCoord_1d_y = torch.tensor([i for i in torch.linspace(0,5*L,5*n_visu)],dtype=torch.float64,  requires_grad=True)
EvalCoordinates = torch.cartesian_prod(TrailCoord_1d_x,TrailCoord_1d_y)


List_elems = torch.arange(0,Domain_mesh.NElem,dtype=torch.int)


learning_rate = 0.001                                           # optimizer learning rate

Model_2D.RefinementParameters(  MaxGeneration = 2, 
                                Jacobian_threshold = 0.5)
                                
Model_2D.TrainingParameters(    Stagnation_threshold = 1e-8, 
                                Max_epochs = 12000, 
                                learning_rate = 0.001)

#%% Training 
# Fisrt stage
max_refinment = 1
n_refinement = 0
stagnation = False
Loss_tot = []
Duration_tot = 0
U_interm_tot = []
Gen_interm_tot = []
X_interm_tot = []
Connectivity_tot = []
d_eps_max_vect = []
eps_max_vect = []
detJ_tot = []
Nnodes_max = 1000
coeff_refinement = np.power((Nnodes_max/Domain_mesh.NNodes),1/max_refinment)
dmax_threshold = 1e-7
import meshio

while n_refinement < max_refinment and not stagnation:
    print(f"Refinement level: {n_refinement}")
    n_refinement +=1
    optimizer = torch.optim.Adam(Model_2D.parameters(), lr=Model_2D.learning_rate)
    n_epochs = 3000
    if n_refinement>4:
        n_epochs = 1000
    # Loss_vect, Duration = Training_2D_Integral(Model_2D, optimizer, n_epochs,List_elems,Mat)

    optimizer = torch.optim.LBFGS(Model_2D.parameters(),line_search_fn="strong_wolfe")
    Loss_vect, Duration = Training_2D_Integral_LBFGS(Model_2D, optimizer, n_epochs,List_elems,Mat)


    # Save current convergence state
    Loss_tot += Loss_vect
    Duration_tot += Duration
    U_interm_tot += Model_2D.U_interm
    Gen_interm_tot += Model_2D.G_interm
    detJ_tot += Model_2D.Jacobian_interm
    X_interm_tot += Model_2D.X_interm
    Connectivity_tot += Model_2D.Connectivity_interm
    meshBeam = meshio.read('geometries/'+Domain_mesh.name_mesh)
    # Cmpute max strain
    _,xg,detJ = Model_2D()
    Model_2D.eval()
    List_elems = torch.arange(0,Model_2D.NElem,dtype=torch.int)
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
        # MaxElemSize = MaxElemSize/coeff_refinement
        MaxElemSize = MaxElemSize/4

        Domain_mesh_2 = pre.Mesh(Name,MaxElemSize, order, dimension)    # Create the mesh object
        Domain_mesh_2.AddBCs(Volume_element, Excluded_elements, DirichletDictionryList)           # Include Boundary physical domains infos (BCs+volume)
        Domain_mesh_2.AddBorders(Borders)
        Domain_mesh_2.MeshGeo()                                # Mesh the .geo file if .msh does not exist
        Domain_mesh_2.ReadMesh()                               # Parse the .msh file
        Domain_mesh_2.ExportMeshVtk()
        List_elems = torch.arange(0,Domain_mesh_2.NElem,dtype=torch.int)
        # Initialise finer model
        Model_2D_2 = MeshNN_2D(Domain_mesh_2, 2, n_integration_points)                # Create the associated model (with 2 components)
        
        newcoordinates = [coord for coord in Model_2D_2.coordinates]
        newcoordinates = torch.cat(newcoordinates,dim=0)
        # Update Domain_mesh vtk mesh to get correct cell Ids
        Domain_mesh.Nodes = [[i+1,Model_2D.coordinates[i][0][0].item(),Model_2D.coordinates[i][0][1].item(),0] for i in range(len(Model_2D.coordinates))]
        Domain_mesh.Connectivity = Model_2D.connectivity
        # HERE THE CELLs must be updated due to h adaptivity
        Domain_mesh.ExportMeshVtk(flag_update = True)
        IDs_newcoord = torch.tensor(Domain_mesh.GetCellIds(newcoordinates),dtype=torch.int)
        NewNodalValues = Model_2D(newcoordinates,IDs_newcoord) 
        if -1 in IDs_newcoord:
            # print(IDs_newcoord)
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

        new_nodal_values_x = nn.ParameterList([nn.Parameter((torch.tensor([i[0]]))) for i in NewNodalValues.t()])
        new_nodal_values_y = nn.ParameterList([nn.Parameter(torch.tensor([i[1]])) for i in NewNodalValues.t()])
        new_nodal_values = [new_nodal_values_x,new_nodal_values_y]
        Model_2D_2.nodal_values_x = new_nodal_values_x
        Model_2D_2.nodal_values_y = new_nodal_values_y
        Model_2D_2.nodal_values = new_nodal_values
        Domain_mesh = Domain_mesh_2
        Model_2D = Model_2D_2
        Model_2D.UnFreeze_FEM()
        Model_2D.Freeze_Mesh()
        # Model_2D.UnFreeze_Mesh()
        Model_2D.train()
        Model_2D.RefinementParameters(  MaxGeneration = 3, 
                                Jacobian_threshold = 0.2)
        Model_2D.TrainingParameters(    Stagnation_threshold = 1e-7, 
                                        Max_epochs = 500, 
                                        learning_rate = 0.001)
    else:
        Model_2D.train()



#%% Export the results to vtk 
import meshio
#%% Read mesh
#%% Get nodal values from the trained model
u_x = [u for u in Model_2D.nodal_values_x]
u_y = [u for u in Model_2D.nodal_values_y]

#%% Compute the strain 
List_elems = torch.arange(0,Model_2D.NElem,dtype=torch.int)
_,xg,detJ = Model_2D(EvalCoordinates, List_elems)
Model_2D.eval()
eps =  Strain(Model_2D(xg, List_elems),xg)
sigma =  torch.stack(Stress(eps[:,0], eps[:,1], eps[:,2], Mat.lmbda, Mat.mu),dim=1)
sigma_VM = VonMises(sigma)
X_interm_tot = [torch.cat([x_i,torch.zeros(x_i.shape[0],1)],dim=1) for x_i in X_interm_tot]
u = torch.stack([torch.cat(u_x),torch.cat(u_y),torch.zeros(torch.cat(u_x).shape[0])],dim=1)


Coord_converged = np.array([[Model_2D.coordinates[i][0][0].item(),Model_2D.coordinates[i][0][1].item(),0] for i in range(len(Model_2D.coordinates))])
Connect_converged = Model_2D.connectivity
sol = meshio.Mesh(Coord_converged, {"triangle":(Connect_converged-1)},
point_data={"U":u.data}, 
cell_data={"eps": [eps.data], "sigma": [sigma.data],  "sigma_vm": [sigma_VM.data]}, )
sol.write(
    "Results/Paraview/sol_u_end_training_gravity_NoBCs_fixed_"+Name+".vtk", 
)

# #%% Export intermediate convergence steps
# meshBeam = meshio.read('geometries/'+Domain_mesh.name_mesh)

# U_interm_tot = [torch.cat([u,torch.zeros(u.shape[0],1)],dim=1) for u in U_interm_tot]

# for timestep in range(len(U_interm_tot)):
#     sol = meshio.Mesh(X_interm_tot[timestep].data, {"triangle":Connectivity_tot[timestep].data},
#     point_data={"U":U_interm_tot[timestep]}, 
#     cell_data={"Gen": [Gen_interm_tot[timestep]], "detJ": [detJ_tot[timestep].data]}, )

#     sol.write(
#         f"Results/Paraview/TimeSeries/solution_multiscale_gravity_NoBCs_fixed_"+Name+f"_{timestep}.vtk",  
#     )

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


coord = torch.tensor(np.load("../2D_example/eval_coordinates.npy"), dtype=torch.float64, requires_grad=True)
List_elems = torch.tensor(Domain_mesh.GetCellIds(coord),dtype=torch.int)

u = Model_2D(coord, List_elems)
eps =  Strain(Model_2D(coord, List_elems),coord)
sigma =  torch.stack(Stress(eps[:,0], eps[:,1], eps[:,2], Mat.lmbda, Mat.mu),dim=1)
sigma_VM = VonMises(sigma)

np.save("../2D_example/NN_solution/"+str(MaxElemSize)+"_u.npy", np.array(u.detach()))
np.save("../2D_example/NN_solution/"+str(MaxElemSize)+"_sigma.npy", np.array(sigma.detach()))
np.save("../2D_example/NN_solution/"+str(MaxElemSize)+"_sigma_VM.npy", np.array(sigma_VM.detach()))

