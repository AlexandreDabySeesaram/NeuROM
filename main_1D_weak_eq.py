#%% Libraries import
# import HiDeNN library
from HiDeNN_PDE import MeshNN_1D, NeuROM
# Import pre-processing functions
import Bin.Pre_processing as pre
# Import torch librairies
import torch
import torch.nn as nn
# Import mechanical functions
from Bin.PDE_Library import RHS, PotentialEnergyVectorised, \
     Derivative, AnalyticGradientSolution, AnalyticSolution, AnalyticBiParametricSolution
# Import Training funcitons
from Bin.Training import Test_GenerateShapeFunctions, Training_InitialStage, \
     Training_FinalStageLBFGS, FilterTrainingData, Training_NeuROM, Training_NeuROM_FinalStageLBFGS, \
     Mixed_Training_InitialStage, Training_FinalStageLBFGS_Mixed, Training_1D_Integral, Training_1D_Integral_LBFGS,\
     Training_1D_WeakEQ_LBFGS, Training_1D_WeakEQ

import matplotlib.pyplot as plt
import matplotlib

#Import post processing libraries
import Post.Plots as Pplot
import time
import os
import torch._dynamo as dynamo
mps_device = torch.device("mps")

import numpy

class Dataset(torch.utils.data.Dataset):

    def __init__(self, X):
        self.X = X
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):    
        x = self.X[index]

        return x


#%% Pre-processing (could be put in config file later)
# Defintition of the structure and meterial
L = 10                                              # Length of the Beam
np = 40                                             # Number of Nodes in the Mesh
A = 1                                               # Section of the beam
E = 175  

alpha = 0.0                                       # Weight for the Mesh regularisation 
order = 1                                          # Order of the shape functions
dimension = 1
n_integr_points = 3

                                           # Young's Modulus (should be 175)
# User defines all boundary conditions 
DirichletDictionryList = [  {"Entity": 1, 
                             "Value": 0.0, 
                             "Normal":0, "Relation": False, "Constitutive": False}, 
                            {"Entity": 2, 
                             "Value": 0.005, 
                             "Normal":0, "Relation": False, "Constitutive": False}]

# Definition of the space discretisation


if order ==1:
    MaxElemSize = L/(np-1)                         # Compute element size
elif order ==2:
    n_elem = 0.5*(np-1)
    MaxElemSize = L/n_elem                         # Compute element size
Excluded = []

if dimension ==1:
    Beam_mesh = pre.Mesh('Beam',MaxElemSize, order, dimension)    # Create the mesh object


Borders = [1,2]
Beam_mesh.AddBorders(Borders)
Volume_element = 100                               # Volume element correspond to the 1D elem in 1D
Beam_mesh.AddBCs(Volume_element,Excluded,
                 DirichletDictionryList)           # Include Boundary physical domains infos (BCs+volume)
Beam_mesh.MeshGeo()                                # Mesh the .geo file if .msh does not exist
Beam_mesh.ReadMesh()                               # Parse the .msh file
Beam_mesh.AssemblyMatrix()                         # Build the assembly weight matrix
Beam_mesh.ExportMeshVtk1D()

#%% Options & Hyperparameters

learning_rate = 1.0e-3                         # optimizer learning rate
n_visu = 5000

#%% Application of the Space HiDeNN
BeamModel = MeshNN_1D(Beam_mesh, n_integr_points)                # Create the associated model
BeamModel.UnFreeze_Values()
BeamModel.Freeze_Mesh()
# BeamModel.UnFreeze_Mesh()

BeamModel.eval()

val_interior = 1.0
val_bc = 0.0

BeamModelTest = MeshNN_1D(Beam_mesh, n_integr_points)                # Create the associated model
BeamModelTest.Freeze_Values()
BeamModelTest.Freeze_Mesh()

BeamModelTest.train()

List_elems = torch.arange(0,Beam_mesh.NElem,dtype=torch.int)


PlotCoordinates = torch.tensor([i for i in torch.linspace(0,L,n_visu)],dtype=torch.float64, requires_grad=True)
IDs_plot = torch.tensor(Beam_mesh.GetCellIds1D(PlotCoordinates),dtype=torch.int)


InitialCoordinates = [BeamModel.coordinates[i].data.item() for i in range(len(BeamModel.coordinates))]
optimizer = torch.optim.Adam(BeamModel.parameters(), lr=learning_rate)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


n_epochs = 20000                             # Maximum number of iterations for the training stage
error, error2, Coord_trajectories = Training_1D_WeakEQ(BeamModel, BeamModelTest, optimizer, n_epochs, PlotCoordinates, IDs_plot, List_elems,A,E)

# error, error2, Coord_trajectories = [],[],[]
n_epochs = 100  
Training_1D_WeakEQ_LBFGS(BeamModel, BeamModelTest, n_epochs, PlotCoordinates, IDs_plot, List_elems,A,E, error, error2, Coord_trajectories)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

new_coord = [coord for coord in BeamModel.coordinates]
new_coord = torch.cat(new_coord,dim=0)

Beam_mesh.Nodes = [[i+1,new_coord[i].item(),0,0] for i in range(len(Beam_mesh.Nodes))]
Beam_mesh.ExportMeshVtk1D(flag_update = True)

IDs_plot = torch.tensor(Beam_mesh.GetCellIds1D(PlotCoordinates),dtype=torch.int)


BeamModel.eval()
BeamModelTest.eval()



u_predicted = BeamModel(PlotCoordinates, IDs_plot)[:,0]

BeamModelTest.SetFixedValues(0,1)
u_predicted_test = BeamModelTest(PlotCoordinates, IDs_plot)[:,0]

for node in range(1,BeamModelTest.NElem-1):
    BeamModelTest.SetFixedValues(node,1)

    u_predicted_test = u_predicted_test + BeamModelTest(PlotCoordinates, IDs_plot)[:,0]

print("u_predicted = ", u_predicted.shape)
print("u_predicted_test = ", u_predicted_test.shape)

analytical_norm = torch.linalg.vector_norm(AnalyticSolution(A,E,PlotCoordinates.data)).data

l2_loss = torch.linalg.vector_norm(AnalyticSolution(A,E,PlotCoordinates.data) - u_predicted).data/analytical_norm
print(f'* Final l2 loss : {numpy.format_float_scientific(l2_loss, precision=4)}')

du_dx = torch.autograd.grad(u_predicted, PlotCoordinates, grad_outputs=torch.ones_like(u_predicted), create_graph=True)[0]
l2_loss_grad = torch.linalg.vector_norm(AnalyticGradientSolution(A,E,PlotCoordinates.data) - du_dx).data/torch.linalg.vector_norm(AnalyticGradientSolution(A,E,PlotCoordinates.data)).data

print(f'* Final l2 loss grad : {numpy.format_float_scientific(l2_loss_grad, precision=4)}')



du_dx = torch.autograd.grad(u_predicted, PlotCoordinates, grad_outputs=torch.ones_like(u_predicted), create_graph=True)[0]
du_test_dx = torch.autograd.grad(u_predicted_test, PlotCoordinates, grad_outputs=torch.ones_like(u_predicted_test), create_graph=True)[0]
# du_test_dx = torch.autograd.grad(u_test, x, grad_outputs=torch.ones_like(u_test), create_graph=True)[0]

prod =  du_test_dx # A*E*du_dx*du_test_dx #+u_predicted_test*RHS(PlotCoordinates)



Coordinates = [BeamModel.coordinates[i].data.item() for i in range(len(BeamModel.coordinates))]

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(9, 7)

param = BeamModel.coordinates[3]
if param.requires_grad == True:
    plt.scatter(InitialCoordinates,[coord*0 for coord in InitialCoordinates], s=6, color="pink", alpha=0.5, label = 'Initial Nodes')
    
plt.plot(Coordinates,[coord*0 for coord in Coordinates],'.k', markersize=2, label = 'Mesh Nodes')
plt.plot(PlotCoordinates.data,AnalyticSolution(A,E,PlotCoordinates.data), label = 'Ground Truth')
plt.plot(PlotCoordinates.data,u_predicted.data,'--', label = 'HiDeNN')
plt.xlabel(r'$\underline{x}$ [m]')
plt.ylabel(r'$\underline{u}\left(\underline{x}\right)$')
plt.legend(loc="upper left")
# plt.title('Displacement')
plt.savefig('Results/Displacement.pdf', transparent=True)  
#plt.show()
plt.clf()


fig = matplotlib.pyplot.gcf()
fig.set_size_inches(9, 7)

param = BeamModel.coordinates[3]
if param.requires_grad == True:
    plt.scatter(InitialCoordinates,[coord*0 for coord in InitialCoordinates], s=6, color="pink", alpha=0.5, label = 'Initial Nodes')
    
plt.plot(Coordinates,[coord*0 for coord in Coordinates],'.k', markersize=2, label = 'Mesh Nodes')
plt.plot(PlotCoordinates.data,AnalyticGradientSolution(A,E,PlotCoordinates.data), label = 'Ground Truth')
plt.plot(PlotCoordinates.data,du_dx.data,'--', label = 'HiDeNN')
plt.xlabel(r'$\underline{x}$ [m]')
plt.ylabel(r'$\underline{u}\left(\underline{x}\right)$')
plt.legend(loc="upper left")
# plt.title('Displacement')
plt.savefig('Results/Gradient.pdf', transparent=True)  
#plt.show()
plt.clf()
