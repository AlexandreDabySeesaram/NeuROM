#%% Libraries import
# import HiDeNN library
from HiDeNN_PDE import MeshNN_2D, NeuROM
# Import pre-processing functions
import Bin.Pre_processing as pre
# Import torch librairies
import torch
import torch.nn as nn
# Import mechanical functions
from Bin.PDE_Library import RHS, PotentialEnergyVectorised, \
     Derivative, AnalyticGradientSolution, AnalyticSolution, AnalyticBiParametricSolution
# Import Training funcitons

#Import post processing libraries
import Post.Plots as Pplot
import time
import os
import torch._dynamo as dynamo
import numpy as numpy
mps_device = torch.device("mps")


from Bin.Training import LBFGS_Stage2_2D, GradDescend_Stage1_2D
from Bin.PDE_Library import Mixed_2D_loss

class Dataset(torch.utils.data.Dataset):

    def __init__(self, X):
        self.X = X
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):    
        x = self.X[index]

        return x

def GetMaxElemSize(L, np, order):
    if order ==1:
        MaxElemSize = L/(np-1)                         # Compute element size
    elif order ==2:
        n_elem = 0.5*(np-1)
        MaxElemSize = L/n_elem                         # Compute element size

    return MaxElemSize




L = 10                                              # Length of the Beam
np = 12                                             # Number of Nodes in the Mesh
#A = 1                                               # Section of the beam
#E = 175                                             # Young's Modulus (should be 175)

loss = [[],[]]

lmbda = 1.25
mu = 1.0

# Definition of the space discretisation
order_u = 2                                          # Order of the shape functions
order_du = 1
dimension = 2

MaxElemSize = GetMaxElemSize(L, np, order_du)

if dimension ==1:
    Domain_mesh_u = pre.Mesh('Beam',MaxElemSize, order_u, dimension)    # Create the mesh object
    Domain_mesh_du = pre.Mesh('Beam',MaxElemSize, order_du, dimension)    # Create the mesh object
if dimension ==2:
    # Domain_mesh_u = pre.Mesh('Rectangle',MaxElemSize, order_u, dimension)    # Create the mesh object
    # Domain_mesh_du = pre.Mesh('Rectangle',MaxElemSize, order_du, dimension)    # Create the mesh object
    Domain_mesh_u = pre.Mesh('Square',MaxElemSize, order_u, dimension)    # Create the mesh object
    Domain_mesh_du = pre.Mesh('Square',MaxElemSize, order_du, dimension)    # Create the mesh object

Volume_element = 100                               # Volume element correspond to the 1D elem in 1D
Excluded_elements_u = []

####################################################################
# User defines all boundary conditions

DirichletDictionryList = [  {"Entity": 111, "Value": 0, "Normal": 0, "Relation": False , "Constitutive": False},
                            {"Entity": 111, "Value": 0, "Normal": 1, "Relation": False, "Constitutive": False},
                            ]


'''
DirichletDictionryList = [  {"Entity": 111, "Value": 0, "Normal": 0, "Relation": False , "Constitutive": False},
                            {"Entity": 111, "Value": 0, "Normal": 1, "Relation": False, "Constitutive": False},
                            {"Entity": 112, "Value": 0, "Normal": 0, "Relation": False , "Constitutive": False},
                            {"Entity": 112, "Value": 0, "Normal": 1, "Relation": False, "Constitutive": False},
                            {"Entity": 113, "Value": 0, "Normal": 0, "Relation": False , "Constitutive": False},
                            {"Entity": 113, "Value": 0, "Normal": 1, "Relation": False, "Constitutive": False},
                            {"Entity": 114, "Value": 0, "Normal": 0, "Relation": False , "Constitutive": False},
                            {"Entity": 114, "Value": 0, "Normal": 1, "Relation": False, "Constitutive": False},
                        ]
'''

Domain_mesh_u.AddBCs(Volume_element, Excluded_elements_u, DirichletDictionryList)           # Include Boundary physical domains infos (BCs+volume)
Domain_mesh_u.MeshGeo()                                # Mesh the .geo file if .msh does not exist
Domain_mesh_u.ReadMesh()                               # Parse the .msh file
Domain_mesh_u.ExportMeshVtk()

####################################################################
# Normal: 0     component: x, du_dx
# Normal: 1     component: y, dv_dy
# Normal: 2     component: x, du_dy = dv_dx

# "Relation": Neumann BC with general n 
#   If "Relation": True and value = [f1,f2]   ---> interpreted as stress.n = [f1,f2]
#   If "Relation": True and value = [c]   ---> interpreted as stress.n = c.n

# "Constitutive": On boundaries where displacement is prescribed by Dirichlet BC.
#                 Stress = compted stress(displacement) 


DirichletDictionryList = [  {"Entity": 112, "Value": 0.0, "Normal": 0, "Relation": False, "Constitutive": False},
                            {"Entity": 112, "Value": 0.0, "Normal": 2, "Relation": False, "Constitutive": False},
                            {"Entity": 114, "Value": 0.0, "Normal": 0, "Relation": False, "Constitutive": False},
                            {"Entity": 114, "Value": 0.0, "Normal": 2, "Relation": False, "Constitutive": False},
                            {"Entity": 113, "Value": 0.05, "Normal": 1, "Relation": False, "Constitutive": False},
                            {"Entity": 113, "Value": 0.0, "Normal": 2, "Relation": False, "Constitutive": False},
                            {"Entity": 115, "Value": [0.0], "Normal": 0, "Relation": True, "Constitutive": False},
                            {"Entity": 111, "Value": 0.0, "Normal": 0, "Relation": False, "Constitutive": True}
                        ]


'''
DirichletDictionryList = [  {"Entity": 112, "Value": 0.0, "Normal": 0, "Relation": False, "Constitutive": True},
                            {"Entity": 114, "Value": 0.0, "Normal": 0, "Relation": False, "Constitutive": True},
                            {"Entity": 113, "Value": 0.0, "Normal": 0, "Relation": False, "Constitutive": True},
                            {"Entity": 111, "Value": 0.0, "Normal": 0, "Relation": False, "Constitutive": True},
                            {"Entity": 115, "Value": [1.0], "Normal": 0, "Relation": True, "Constitutive": False}
                        ]
'''

Excluded_elements_du = [200]
Domain_mesh_du.AddBCs(Volume_element, Excluded_elements_du, DirichletDictionryList)           # Include Boundary physical domains infos (BCs+volume)
Domain_mesh_du.MeshGeo()                                # Mesh the .geo file if .msh does not exist
Domain_mesh_du.ReadMesh()                               # Parse the .msh file
Domain_mesh_du.ExportMeshVtk()
Domain_mesh_du.ReadNormalVectors()
####################################################################

print("Model u")
Model_u = MeshNN_2D(Domain_mesh_u, 2)                # Create the associated model
Model_u.UnFreeze_Values()
Model_u.Freeze_Mesh()

print()
print("Model du")
Model_du = MeshNN_2D(Domain_mesh_du, 3)                # Create the associated model
Model_du.UnFreeze_Values()
Model_du.Freeze_Mesh()

####################################################################


def GenerateData(n_train):
    TrailCoord_1d_x = torch.tensor([i for i in torch.linspace(0,L,n_train)],dtype=torch.float64, requires_grad=True)
    TrailCoord_1d_y = torch.tensor([i for i in torch.linspace(0,L,n_train)],dtype=torch.float64,  requires_grad=True)

    PlotCoordinates = torch.cartesian_prod(TrailCoord_1d_x,TrailCoord_1d_y)
    IDs_u = torch.tensor(Domain_mesh_du.GetCellIds(PlotCoordinates),dtype=torch.int)
    IDs_du = torch.tensor(Domain_mesh_du.GetCellIds(PlotCoordinates),dtype=torch.int)

    inside_domain = torch.where(IDs_u > -1)
    #PlotCoordinates = torch.tensor(PlotCoordinates[inside_domain], dtype=torch.float64, requires_grad=True)
    PlotCoordinates = PlotCoordinates[inside_domain]
    PlotCoordinates = PlotCoordinates.clone().detach().requires_grad_(True)

    IDs_u = IDs_u[inside_domain]
    IDs_du = IDs_du[inside_domain]
    return PlotCoordinates, IDs_u, IDs_du

n_train = 50
TrainCoordinates, TrainIDs_u, TrainIDs_du = GenerateData(n_train)

n_train = 70
PlotCoordinates, IDs_u, IDs_du = GenerateData(n_train)




if len(Model_du.constit_BC_node_IDs)>0:
    constit_point_IDs = Model_du.constit_BC_node_IDs[0]
    constit_point_coord = [Model_u.coordinates[i] for i in constit_point_IDs]
    constit_cell_IDs_u = torch.tensor([torch.tensor(Domain_mesh_u.GetCellIds(i),dtype=torch.int)[0] for i in constit_point_coord])
    constit_point_coord = torch.cat(constit_point_coord).clone().detach().requires_grad_(True)
else:
    constit_cell_IDs_u = []
    constit_point_coord = []
####################################################

u_predicted = Model_u(PlotCoordinates, IDs_u) 
du_predicted = Model_du(PlotCoordinates, IDs_du) 

Pplot.Plot2Dresults(u_predicted, PlotCoordinates , "_u_Initial")
#Pplot.Plot2Dresults_Derivative(du_predicted, s11, s22, s12, n_train, 5*n_train, "_Stress_Initial")

optimizer = torch.optim.Adam(list(Model_u.parameters())+list(Model_du.parameters()))

w0 = L
w1 = 1


print("**************** START TRAINING 1st stage ***************\n")
loss = [[],[]]

current_BS = 500
n_epochs = 400
CoordinatesBatchSet = torch.utils.data.DataLoader([[TrainCoordinates[i], TrainIDs_u[i], TrainIDs_du[i]] for i in range((TrainIDs_u.shape)[0])], 
                                                        batch_size=current_BS, shuffle=True)print("Number of training points = ", PlotCoordinates.shape[0])
print("Batch size = ", CoordinatesBatchSet.batch_size)
print("Number of batches per epoch = ", len(CoordinatesBatchSet))
print()

Model_u, Model_du, loss = GradDescend_Stage1_2D(Model_u, Model_du, IDs_u, IDs_du, PlotCoordinates, CoordinatesBatchSet, w0, w1, n_epochs, optimizer, 
                                n_train, loss, constit_point_coord, constit_cell_IDs_u, lmbda, mu)



print("*************** 2nd stage LBFGS ***************\n")
n_epochs = 30
Model_u, Model_du = LBFGS_Stage2_2D(Model_u, Model_du, IDs_u, IDs_du, PlotCoordinates, 
                            TrainCoordinates, TrainIDs_u, TrainIDs_du,
                            w0, w1, n_train, n_epochs, constit_point_coord, constit_cell_IDs_u, lmbda, mu)


num_sol_name = "Rectangle_order_1_2.5_order2_displacement.npy"

Model_u.Update_Middle_Nodes(Domain_mesh_u)

#NumSol_eval(Domain_mesh_u, Domain_mesh_du, Model_u, Model_du, num_sol_name, L)

