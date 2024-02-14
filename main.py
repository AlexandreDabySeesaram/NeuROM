#%% Libraries import
# import HiDeNN library
from HiDeNN_PDE import MeshNN, NeuROM
# Import pre-processing functions
import Bin.Pre_processing as pre
# Import torch librairies
import torch
import torch.nn as nn
mps_device = torch.device("mps")
# Import mechanical functions
from Bin.PDE_Library import RHS, PotentialEnergyVectorised, \
        Derivative, AnalyticGradientSolution, AnalyticSolution
# Import Training funcitons

from Bin.Training import Test_GenerateShapeFunctions, Training_InitialStage, \
    Training_FinalStageLBFGS, FilterTrainingData, Training_NeuROM
#Import post processing libraries
import Post.Plots as Pplot
import time
#%% Pre-processing (could be put in config file later)
# Geometry of the Mesh
L = 10                                      # Length of the Beam
np = 23                                     # Number of Nodes in the Mesh
A = 1                                       # Section of the beam
E = 175                                     # Young's Modulus (should be 175)
alpha =0.005                                # Weight for the Mesh regularisation 

# User defines all boundary conditions 
DirichletDictionryList = [{"Entity": 1, 
                           "Value": 0, 
                           "normal":1}, 
                            {"Entity": 2, 
                             "Value": 10, 
                             "normal":1}]

MaxElemSize = L/(np-1)                      # Compute element size
Beam_mesh = pre.Mesh('Beam',MaxElemSize)    # Create the mesh object
Volume_element = 100                        # Volume element correspond to the 1D elem in 1D
Beam_mesh.AddBCs(Volume_element,
                 DirichletDictionryList)    # Include Boundary physical domains infos (BCs+volume)
Beam_mesh.MeshGeo()                         # Mesh the .geo file if .msh does not exist
Beam_mesh.ReadMesh()                        # Parse the .msh file
Beam_mesh.AssemblyMatrix()                  # Build the assembly weight matrix


#%% Application of the NN
BeamModel = MeshNN(Beam_mesh,alpha)     # Create the associated model
# Boundary conditions
u_0 = 0                                 #Left BC
u_L = 0                                 #Right BC
BeamModel.SetBCs(u_0,u_L)

# Set the coordinates as trainable
BeamModel.UnFreeze_Mesh()
# Set the coordinates as untrainable
# BeamModel.Freeze_Mesh()
# Set the require output requirements
BoolPlot = False                        # Boolean for plots used for gif
BoolPlotPost = False                    # Boolean for plots used for Post
BoolCompareNorms = True                 # Boolean for comparing energy norm to L2 norm
BoolGPU = False                         # Boolean enabling GPU computations (autograd function is not working currently on mac M2)




#%% Define loss and optimizer
learning_rate = 0.001
n_epochs = 700
optimizer = torch.optim.Adam(BeamModel.parameters(), lr=learning_rate)
MSE = nn.MSELoss()

#%% Debuging cell
n_modes = 1
mu_min = 100
mu_max = 200
N_mu = 10

BCs=[u_0,u_L]
BeamROM = NeuROM(Beam_mesh, BCs, n_modes, mu_min, mu_max,N_mu)

TrialCoordinates = torch.tensor([[i/50] for i in range(2,500)], 
                                dtype=torch.float32, requires_grad=True)

TrialPara = torch.linspace(mu_min,mu_max,50, 
                                dtype=torch.float32, requires_grad=True)
TrialPara = TrialPara[:,None] # Add axis so that dimensions match

u_x = BeamModel(TrialCoordinates)
t_start = time.time()
u_x_para = BeamROM(TrialCoordinates,TrialPara)
t_end = time.time()

print(f'* Evaluation time of {u_x_para.shape[0]*u_x_para.shape[1]} values: {t_end-t_start}s')
optimizer = torch.optim.Adam(BeamROM.parameters(), lr=learning_rate)


#%% Training
BeamROM.Freeze_Mesh()
BeamROM.Freeze_MeshPara()
# BeamROM.Freeze_Space()
BeamROM.Freeze_Para()
import matplotlib.pyplot as plt


# Train Space
BeamROM.UnFreeze_Para()

# Loss_vect =  Training_NeuROM(BeamROM, A, L, TrialCoordinates,TrialPara, optimizer, n_epochs, BoolCompareNorms, MSE)
Loss_vect =  Training_NeuROM(BeamROM, A, L, TrialCoordinates,TrialPara, optimizer, 5000, BoolCompareNorms, MSE)
BeamROM.Freeze_Space()
# BeamROM.UnFreeze_Para()
# # TrialPara = torch.linspace(mu_min,mu_max,200)
# TrialPara = torch.linspace(mu_min,mu_max,200)
# TrialPara = TrialPara[:,None] # Add axis so that dimensions match
# # Train para
# Loss_vect =  Training_NeuROM(BeamROM, A, L, TrialCoordinates,TrialPara, optimizer, n_epochs, BoolCompareNorms, MSE)



#%% Check model
import matplotlib.pyplot as plt

PaperPara = torch.tensor([150])
PaperPara = PaperPara[:,None] # Add axis so that dimensions match
u_150 = BeamROM(TrialCoordinates,PaperPara)
u_analytical_150 = AnalyticSolution(A,PaperPara.item(),TrialCoordinates.data)
plt.plot(TrialCoordinates.data,u_analytical_150, color="#01426A")
plt.plot(TrialCoordinates.data,u_150.data,'--', color="#01426A")

PaperPara = torch.tensor([200])
PaperPara = PaperPara[:,None] # Add axis so that dimensions match
u_200 = BeamROM(TrialCoordinates,PaperPara)
u_analytical_200 = AnalyticSolution(A,PaperPara.item(),TrialCoordinates.data)
plt.plot(TrialCoordinates.data,u_analytical_200, color="#00677F")
plt.plot(TrialCoordinates.data,u_200.data,'--',color="#00677F")

PaperPara = torch.tensor([100])
PaperPara = PaperPara[:,None] # Add axis so that dimensions match
u_100 = BeamROM(TrialCoordinates,PaperPara)
u_analytical_100 = AnalyticSolution(A,PaperPara.item(),TrialCoordinates.data)
plt.plot(TrialCoordinates.data,u_analytical_100,color="#A92021")
plt.plot(TrialCoordinates.data,u_100.data,'--',color="#A92021")
plt.show()
plt.clf()

#%% plot interactive
from ipywidgets import interact, widgets

def interactive_plot(E):
    # Calculate the corresponding function values for each x value
    E = torch.tensor([E])
    E = E[:,None] # Add axis so that dimensions match
    u_E = BeamROM(TrialCoordinates,E)
    
    # Plot the function
    plt.plot(TrialCoordinates.data, u_E.data)
    plt.title('Displacement')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.grid(True)
    plt.ylim((0,0.02))
    plt.show()

# Create an interactive slider
slider = widgets.FloatSlider(value=0, min=100, max=200, step=0.01, description='E')

# Connect the slider to the interactive plot function
interactive_plot_widget = interact(interactive_plot, E=slider)

#%% Training loop
# TrialCoordinates = torch.tensor([[i/50] for i in range(2,500)], 
#                                 dtype=torch.float32, requires_grad=True)

# # If GPU
# if BoolGPU:
#     BeamModel.to(mps_device)
#     TrialCoordinates = torch.tensor([[i/50] for i in range(2,500)], 
#                                 dtype=torch.float32, requires_grad=True).to(mps_device)



# # Test_GenerateShapeFunctions(BeamModel, TrialCoordinates)
# n_elem = 1
# error, error2, InitialCoordinates, Coord_trajectories, BeamModel = Training_InitialStage(BeamModel, A, E, L, n_elem, 
#                                                                                          TrialCoordinates, optimizer, n_epochs, 
#                                                                                          BoolCompareNorms, MSE)

# Training_FinalStageLBFGS(BeamModel, A, E, L, n_elem, InitialCoordinates, 
#                          TrialCoordinates, n_epochs, BoolCompareNorms, 
#                          MSE, error, error2, Coord_trajectories)


#%% Post-processing
if BoolPlotPost:
    # Retrieve coordinates
    Coordinates = Coord_trajectories[-1]
    # Tests on trained data and compare to reference
    Pplot.PlotSolution_Coordinates_Analytical(A,E,InitialCoordinates,Coordinates,
                                            TrialCoordinates,AnalyticSolution,BeamModel,
                                            'Solution_displacement')
    # Plots the gradient & compare to reference
    Pplot.PlotGradSolution_Coordinates_Analytical(A,E,InitialCoordinates,Coordinates,
                                                TrialCoordinates,AnalyticGradientSolution,
                                                BeamModel,Derivative,'Solution_gradients')
    # plots zoomed energy loss
    Pplot.PlotEnergyLoss(error,0,'Loss')

    # plots zoomed energy loss
    Pplot.PlotEnergyLoss(error,2500,'Loss_zoomed')

    # Plots trajectories of the coordinates while training
    Pplot.PlotTrajectories(Coord_trajectories,'Trajectories')

    if BoolCompareNorms:
        Pplot.Plot_Compare_Loss2l2norm(error,error2,'Loss_Comaprison')

# %%
