#%% Libraries import
from HiDeNN_PDE import MeshNN_2D
import src.Pre_processing as pre
import torch
import torch.nn as nn
import numpy as numpy

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

order = 1    
n_integration_points = 1
                                                   # Order of the FE interpolation
dimension = 2                                                   # Dimension of the problem
MaxElemSize = 0.075                                                 # Maximum element size of the mesh
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

cell_ids = torch.arange(0,Model_2D.NElem-1)


node_ids = Domain_mesh.Connectivity[cell_ids,:]

node1_coord =  torch.cat([Model_2D.coordinates[int(row)-1] for row in node_ids[:,0]])
node2_coord =  torch.cat([Model_2D.coordinates[int(row)-1] for row in node_ids[:,1]])
node3_coord =  torch.cat([Model_2D.coordinates[int(row)-1] for row in node_ids[:,2]])

coord = (node1_coord + node2_coord + node3_coord)/3

numpy.save("../2D_example/eval_coordinates.npy", numpy.array(coord))
