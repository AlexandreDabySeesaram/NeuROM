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
np = 5                                             # Number of Nodes in the Mesh
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
    Domain_mesh_u = pre.Mesh('Rectangle',MaxElemSize, order_u, dimension)    # Create the mesh object
    Domain_mesh_du = pre.Mesh('Rectangle',MaxElemSize, order_du, dimension)    # Create the mesh object

Volume_element = 100                               # Volume element correspond to the 1D elem in 1D

####################################################################
# User defines all boundary conditions
DirichletDictionryList = [  {"Entity": 111, "Value": 0, "Normal": 0},
                            {"Entity": 111, "Value": 0, "Normal": 1},
                            {"Entity": 113, "Value": 0, "Normal": 0},
                            {"Entity": 113, "Value": 1, "Normal": 1},]

Domain_mesh_u.AddBCs(Volume_element, DirichletDictionryList)           # Include Boundary physical domains infos (BCs+volume)
Domain_mesh_u.MeshGeo()                                # Mesh the .geo file if .msh does not exist
Domain_mesh_u.ReadMesh()                               # Parse the .msh file
Domain_mesh_u.ExportMeshVtk()

####################################################################
# Normal: 0     component: x, du_dx
# Normal: 1     component: y, dv_dy
# Normal: 2     component: x, du_dy = dv_dx

DirichletDictionryList = [  {"Entity": 112, "Value": 0, "Normal": 0},
                            {"Entity": 112, "Value": 0, "Normal":  2},
                            {"Entity": 114, "Value": 0, "Normal": 0},
                            {"Entity": 114, "Value": 0, "Normal": 2},]

Domain_mesh_du.AddBCs(Volume_element, DirichletDictionryList)           # Include Boundary physical domains infos (BCs+volume)
Domain_mesh_du.MeshGeo()                                # Mesh the .geo file if .msh does not exist
Domain_mesh_du.ReadMesh()                               # Parse the .msh file
Domain_mesh_du.ExportMeshVtk()

####################################################################

n_train = 20

TrailCoord_1d_x = torch.tensor([i for i in torch.linspace(0,L,n_train)],dtype=torch.float64, requires_grad=True)
TrailCoord_1d_y = torch.tensor([i for i in torch.linspace(0,5*L,5*n_train)],dtype=torch.float64,  requires_grad=True)

PlotCoordinates = torch.cartesian_prod(TrailCoord_1d_x,TrailCoord_1d_y)
IDs_u = torch.tensor(Domain_mesh_u.GetCellIds(PlotCoordinates),dtype=torch.int)
IDs_du = torch.tensor(Domain_mesh_du.GetCellIds(PlotCoordinates),dtype=torch.int)



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

#print("Model du param")
#for param in Model_du.parameters():
#    if param.requires_grad == True:
#        print(param)

####################################################


u_predicted = Model_u(PlotCoordinates, IDs_u) 
du_predicted = Model_du(PlotCoordinates, IDs_du) 

Pplot.Plot2Dresults(u_predicted, n_train, 5*n_train, "_u_Initial")
#Pplot.Plot2Dresults_Derivative(du_predicted, s11, s22, s12, n_train, 5*n_train, "_Stress_Initial")

optimizer = torch.optim.Adam(list(Model_u.parameters())+list(Model_du.parameters()), lr=1.0e-2)

w0 = numpy.sqrt(10*50)
w1 = 1


print("**************** START TRAINING 1st stage ***************\n")
loss = [[],[]]



current_BS = 200
n_epochs = 200
CoordinatesBatchSet = torch.utils.data.DataLoader([[PlotCoordinates[i], IDs_u[i], IDs_du[i]] for i in range((IDs_u.shape)[0])], batch_size=current_BS, shuffle=True)
print("Number of training points = ", PlotCoordinates.shape[0])
print("Batch size = ", CoordinatesBatchSet.batch_size)
print("Number of batches per epoch = ", len(CoordinatesBatchSet))
print()

Model_u, Model_du, loss = GradDescend_Stage1_2D(Model_u, Model_du, IDs_u, IDs_du, PlotCoordinates, CoordinatesBatchSet, w0, w1, n_epochs, optimizer, n_train, loss)

print("*************** 2nd stage LBFGS ***************\n")

Model_u, Model_du = LBFGS_Stage2_2D(Model_u, Model_du, IDs_u, IDs_du, PlotCoordinates, w0, w1, n_train)



num_sol_name = "Rectangle_order_1_2.5_order2_displacement.npy"

NumSol_eval(Domain_mesh_u, Domain_mesh_du, Model_u, Model_du, num_sol_name, L)