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

#%% Build model

Name = 'Hole'
Name = 'Hole_3'

# Initialise meterial
Mat = pre.Material( flag_lame = False,                          # If True should input lmbda and mu instead
                    coef1 = 5*1e-3,                             # Young Modulus
                    coef2 = 0.3                                 # Poisson's ratio
                    )

# Create mesh
order = 1                                                       # Order of the FE interpolation
dimension = 2                                                   # Dimension of the problem
MaxElemSize = 5                                                 # Maximum element size of the mesh
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


# Parametric domain
#%% Application of NeuROM
n_modes = 100

mu_min = 100
mu_max = 200
N_mu = 10

# Para Young
Eu_min = 100
Eu_max = 200
N_E = 10


# ParameterHypercube = torch.tensor([[Eu_min,Eu_max,N_E],[Eu_min,Eu_max,N_A]])
ParameterHypercube = torch.tensor([[Eu_min,Eu_max,N_E]])

n_modes = 100

BeamROM = NeuROM(Domain_mesh, n_modes, ParameterHypercube)
BeamROM.train()
BeamROM.TrainingParameters(    Stagnation_threshold = 1e-7, 
                                Max_epochs = 1000, 
                                learning_rate = 0.001)
u_predicted,xg,detJ = BeamROM.Space_modes[0]()
optimizer = torch.optim.Adam([p for p in BeamROM.parameters() if p.requires_grad], lr=BeamROM.learning_rate)
Param_trial = torch.linspace(mu_min,mu_max,50, 
                                    dtype=torch.float32, requires_grad=True)
Param_trial = Param_trial[:,None] # Add axis so that dimensions match

Para_coord_list = nn.ParameterList((Param_trial,Param_trial))

Loss_vect, Duration = Training_2D_NeuROM(BeamROM, Para_coord_list, optimizer, BeamROM.Max_epochs,Mat)
