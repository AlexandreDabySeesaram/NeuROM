#%% Libraries import
# import HiDeNN library
from HiDeNN_PDE import MeshNN
# Import pre-processing functions
import Bin.Pre_processing as pre
# Import torch librairies
import torch
import torch.nn as nn
mps_device = torch.device("mps")
# Import mechanical functions
from Bin.PDE_Library import RHS, PotentialEnergyVectorised, \
        Derivative, AnalyticGradientSolution, AnalyticSolution
#Import post processing libraries
import Post.Plots as Pplot
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
# BeamModel.UnFreeze_Mesh()
# Set the coordinates as untrainable
BeamModel.Freeze_Mesh()
# Set the require output requirements
BoolPlot = False                        # Boolean for plots used for gif
BoolPlotPost = False                    # Boolean for plots used for Post
BoolCompareNorms = True                 # Boolean for comparing energy norm to L2 norm
BoolGPU = False                         # Boolean enabling GPU computations (autograd function is not working currently on mac M2)




#%% Define loss and optimizer
learning_rate = 0.001
n_epochs = 5000
optimizer = torch.optim.Adam(BeamModel.parameters(), lr=learning_rate)
MSE = nn.MSELoss()



#%% Training loop
TrialCoordinates = torch.tensor([[i/50] for i in range(2,500)], 
                                dtype=torch.float32, requires_grad=True)

# If GPU
if BoolGPU:
    BeamModel.to(mps_device)
    TrialCoordinates = torch.tensor([[i/50] for i in range(2,500)], 
                                dtype=torch.float32, requires_grad=True).to(mps_device)


from Bin.Training import Test_GenerateShapeFunctions, Training_InitialStage, \
    Training_FinalStageLBFGS, FilterTrainingData

# Test_GenerateShapeFunctions(BeamModel, TrialCoordinates)
n_elem = 1
error, error2, InitialCoordinates, Coord_trajectories, BeamModel = Training_InitialStage(BeamModel, A, E, L, n_elem, 
                                                                                         TrialCoordinates, optimizer, n_epochs, 
                                                                                         BoolCompareNorms, MSE)

Training_FinalStageLBFGS(BeamModel, A, E, L, n_elem, InitialCoordinates, 
                         TrialCoordinates, n_epochs, BoolCompareNorms, 
                         MSE, error, error2, Coord_trajectories)


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
