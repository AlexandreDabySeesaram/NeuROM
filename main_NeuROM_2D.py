#%% Libraries import
from HiDeNN_PDE import MeshNN_2D, NeuROM
import Bin.Pre_processing as pre
import torch
import torch.nn as nn
import numpy as np
from Bin.PDE_Library import Strain, Stress, InternalEnergy_2D, VonMises
from Bin.Training import Training_2D_Integral,Training_2D_NeuROM
import Post.Plots as Pplot
import time
import os
import matplotlib.pyplot as plt
from importlib import reload  # Python 3.4+

#%% Build model

Name = 'Hole'
Name = 'Hole_3'

# Initialise meterial
Mat = pre.Material( flag_lame = False,                          # If True should input lmbda and mu instead
                    coef1 = 1,                             # Young Modulus
                    coef2 = 0.3                                 # Poisson's ratio
                    )

# Create mesh
order = 1                                                       # Order of the FE interpolation
dimension = 2                                                   # Dimension of the problem
MaxElemSize = 0.5                                                 # Maximum element size of the mesh
Domain_mesh = pre.Mesh(Name,MaxElemSize, order, dimension)      # Create the mesh object
Volume_element = 100                                            # Volume element

DirichletDictionryList = [  {"Entity": 111, "Value": 0, "Normal": 1, "Relation": False, "Constitutive": False},
                            {"Entity": 111, "Value": 0, "Normal": 0, "Relation": False, "Constitutive": False},
                            {"Entity": 113, "Value": 0, "Normal": 1, "Relation": False, "Constitutive": False},
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


# Parametric domain
#%% Application of NeuROM
n_modes_max = 100
n_modes_ini = 2

mu_min = 100
mu_max = 200
N_mu = 30

# Para Young
Eu_min = 1e-3
Eu_max = 10e-3
N_E = 10


# ParameterHypercube = torch.tensor([[Eu_min,Eu_max,N_E],[Eu_min,Eu_max,N_A]])
ParameterHypercube = torch.tensor([[Eu_min,Eu_max,N_E]])

n_modes = 100

BeamROM = NeuROM(Domain_mesh, ParameterHypercube, n_modes_ini,n_modes_max)
BeamROM.train()
BeamROM.TrainingParameters(    Stagnation_threshold = 1e-7, 
                                Max_epochs = 10000, 
                                learning_rate = 0.001)
u_predicted,xg,detJ = BeamROM.Space_modes[0]()
optimizer = torch.optim.Adam([p for p in BeamROM.parameters() if p.requires_grad], lr=BeamROM.learning_rate)
Param_trial = torch.linspace(Eu_min,Eu_max,50, 
                                    dtype=torch.float32, requires_grad=True)
Param_trial = Param_trial[:,None] # Add axis so that dimensions match

Para_coord_list = nn.ParameterList((Param_trial,Param_trial))

#%% Check init

#%% Post 
u_k,xg_k,detJ_k = BeamROM.Space_modes[0]()

Pplot.Plot2Dresults(u_k, xg_k, '2D_ROM_FirstMode_before')

u_k2,xg_k2,detJ_k2 = BeamROM.Space_modes[1]()

Pplot.Plot2Dresults(u_k2, xg_k2, '2D_ROM_SecondMode_before')

#%% Train 

Loss_vect, Duration = Training_2D_NeuROM(BeamROM, Para_coord_list, optimizer, BeamROM.Max_epochs,Mat)

#%% Post 
u_k,xg_k,detJ_k = BeamROM.Space_modes[0]()
Pplot.Plot2Dresults(u_k, xg_k, '2D_ROM_FirstMode')
u_k2,xg_k2,detJ_k2 = BeamROM.Space_modes[1]()
Pplot.Plot2Dresults(u_k2, xg_k2, '2D_ROM_SecondMode')

Pplot.PlotParaModes(BeamROM,Para_coord_list,'name_model')


# %% Pyvista plot

import pyvista as pv
import vtk
import meshio

#%% Solo
filename = 'Geometries/'+Domain_mesh.name_mesh
mesh = pv.read(filename)

Nodes = np.stack(Domain_mesh.Nodes)
parameter = 5e-3

Param_trial = torch.tensor([parameter],dtype=torch.float32)
Param_trial = Param_trial[:,None] # Add axis so that dimensions match
Para_coord_list = nn.ParameterList((Param_trial,Param_trial))
BeamROM.eval()
u_sol = BeamROM(torch.tensor(Nodes[:,1:]),Para_coord_list)
u = torch.stack([(u_sol[0,:,0]),(u_sol[1,:,0]),torch.zeros(u_sol[0,:,0].shape[0])],dim=1)


# Plot the mesh
scalar_field_name = 'Uy'
mesh.point_data['U'] = u.data
mesh.point_data['Ux'] = u[:,0].data
mesh.point_data['Uy'] = u[:,1].data
mesh.point_data['Uz'] = u[:,2].data

plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars=scalar_field_name, cmap='viridis', scalar_bar_args={'title': 'Displacement', 'vertical': True})
plotter.show()

#%% Parametric



# Define the parameter to adjust and its initial value
parameter = 5e-3

Param_trial = torch.tensor([parameter],dtype=torch.float32)
Param_trial = Param_trial[:,None] # Add axis so that dimensions match
Para_coord_list = nn.ParameterList((Param_trial,Param_trial))

BeamROM.eval()
u_sol = BeamROM(torch.tensor(Nodes[:,1:]),Para_coord_list)
u = torch.stack([(u_sol[0,:,0]),(u_sol[1,:,0]),torch.zeros(u_sol[0,:,0].shape[0])],dim=1)
mesh.point_data['U'] = u.data
mesh.point_data['Ux'] = u[:,0].data
mesh.point_data['Uy'] = u[:,1].data
mesh.point_data['Uz'] = u[:,2].data
plotter = pv.Plotter()

# Function to update the solution based on the parameter
def update_solution(value):
    # plotter.clear()
    parameter = value
    Param_trial = torch.tensor([parameter],dtype=torch.float32)
    Param_trial = Param_trial[:,None] # Add axis so that dimensions match
    Para_coord_list = nn.ParameterList((Param_trial,Param_trial))
    u_sol = BeamROM(torch.tensor(Nodes[:,1:]),Para_coord_list)
    u = torch.stack([(u_sol[0,:,0]),(u_sol[1,:,0]),torch.zeros(u_sol[0,:,0].shape[0])],dim=1)
    mesh = pv.read(filename)
    u[:,2]+=200*value
    mesh.point_data['U'] = u.data
    mesh.point_data['Ux'] = u[:,0].data
    mesh.point_data['Uy'] = u[:,1].data
    mesh.point_data['Uz'] = u[:,2].data
    plotter.add_mesh(mesh.warp_by_vector(vectors="U",factor=20.0,inplace=True), scalars=scalar_field_name, cmap='viridis', scalar_bar_args={'title': 'Displacement', 'vertical': False}, show_edges=True)
    return

plotter.add_slider_widget(update_solution, [1e-3, 10e-3], title='Stifness')
plotter.show()



# %%
