#%% Libraries import
# import HiDeNN library
from HiDeNN_PDE import MeshNN, NeuROM
# Import pre-processing functions
import Bin.Pre_processing as pre
# Import torch librairies
import torch
import torch.nn as nn
# Import mechanical functions
from Bin.PDE_Library import RHS, PotentialEnergyVectorised, \
     Derivative, AnalyticGradientSolution, AnalyticSolution
# Import Training funcitons
from Bin.Training import Test_GenerateShapeFunctions, Training_InitialStage, \
     Training_FinalStageLBFGS, FilterTrainingData, Training_NeuROM
#Import post processing libraries
import Post.Plots as Pplot
import time
import os
import torch._dynamo as dynamo
mps_device = torch.device("mps")

#%% Pre-processing (could be put in config file later)
# Defintition of the structure and meterial
L = 10                                              # Length of the Beam
np = 50                                             # Number of Nodes in the Mesh
A = 1                                               # Section of the beam
E = 175                                             # Young's Modulus (should be 175)
# User defines all boundary conditions 
DirichletDictionryList = [  {"Entity": 1, 
                             "Value": 0, 
                             "normal":1}, 
                            {"Entity": 2, 
                             "Value": 0, 
                             "normal":1}]

# Definition of the space discretisation
alpha =0.005                                       # Weight for the Mesh regularisation 
order = 1                                          # Order of the shape functions
if order ==1:
    MaxElemSize = L/(np-1)                         # Compute element size
elif order ==2:
    n_elem = 0.5*(np-1)
    MaxElemSize = L/n_elem                         # Compute element size

Beam_mesh = pre.Mesh('Beam',MaxElemSize, order)    # Create the mesh object
Volume_element = 100                               # Volume element correspond to the 1D elem in 1D
Beam_mesh.AddBCs(Volume_element,
                 DirichletDictionryList)           # Include Boundary physical domains infos (BCs+volume)
Beam_mesh.MeshGeo()                                # Mesh the .geo file if .msh does not exist
Beam_mesh.ReadMesh()                               # Parse the .msh file
Beam_mesh.AssemblyMatrix()                         # Build the assembly weight matrix

#%% Options 
BoolPlot = False                                   # Boolean for plots used for gif
BoolPlotPost = False                               # Boolean for plots used for Post
BoolCompareNorms = True                            # Boolean for comparing energy norm to L2 norm
BoolGPU = False                                    # Boolean enabling GPU computations (autograd function is not working currently on mac M2)
TrainingRequired = False                           # Boolean leading to Loading pre trained model or retraining from scratch
SaveModel = False                                  # Boolean leading to Loading pre trained model or retraining from scratch
ParametricStudy = True                             # Boolean to switch between space model and parametric sturdy

#%% Application of the Space HiDeNN
BeamModel = MeshNN(Beam_mesh,alpha)                # Create the associated model
# Boundary conditions
u_0 = DirichletDictionryList[0]['Value']           #Left BC
u_L = DirichletDictionryList[1]['Value']           #Right BC
BeamModel.SetBCs(u_0,u_L)

# Set the boundary values as trainable
#BeamModel.UnFreeze_BC()
# Set the coordinates as trainable
BeamModel.UnFreeze_Mesh()
# Set the coordinates as untrainable
# BeamModel.Freeze_Mesh()
# Set the require output requirements

#%% Application of NeuROM
n_modes = 1
mu_min = 100
mu_max = 200
N_mu = 10

# Para Young
Eu_min = 100
Eu_max = 200
N_E = 10

# Para Area
A_min = 0.1
A_max = 10
N_A = 10

# ParameterHypercube = torch.tensor([[Eu_min,Eu_max,N_E],[A_min,A_max,N_A]])
ParameterHypercube = torch.tensor([[Eu_min,Eu_max,N_E]])

# Boundary conditions
u_0 = DirichletDictionryList[0]['Value']           #Left BC
u_L = DirichletDictionryList[1]['Value']           #Right BC
BeamROM = NeuROM(Beam_mesh, [u_0,u_L], n_modes, ParameterHypercube)
name_model = 'ROM_1Para_np_'+str(np)+'_order_'+str(order)+'_nmodes_'\
            +str(n_modes)+'_npara_'+str(ParameterHypercube.shape[0])

#%% Define hyperparameters
learning_rate = 0.001
n_epochs = 7000
FilterTrainingData = False

#%% Training 
if not ParametricStudy:
    # Training loop (Non parametric model)
    print("Training loop (Non parametric model)")
    optimizer = torch.optim.SGD(BeamModel.parameters(), lr=learning_rate)
    TrialCoordinates = torch.tensor([[i] for i in torch.linspace(0,L,500)], dtype=torch.float32, requires_grad=True)
    
    # If GPU
    if BoolGPU:
        BeamModel.to(mps_device)
        TrialCoordinates = torch.tensor([[i/50] for i in range(2,502)], 
                                    dtype=torch.float32, requires_grad=True).to(mps_device)

    # Training initial stage
    error, error2, InitialCoordinates, Coord_trajectories, BeamModel = Training_InitialStage(BeamModel, A, E, L, 
                                                                                            TrialCoordinates, optimizer, n_epochs, 
                                                                                            BoolCompareNorms, nn.MSELoss(), FilterTrainingData)

    # Training final stage
    Training_FinalStageLBFGS(BeamModel, A, E, L, InitialCoordinates, 
                            TrialCoordinates, n_epochs, BoolCompareNorms, 
                            nn.MSELoss(), FilterTrainingData,
                            error, error2, Coord_trajectories)

else:
    BeamROM.Freeze_Mesh()
    BeamROM.Freeze_MeshPara()
    TrialCoordinates = torch.tensor([[i/50] for i in range(2,500)], 
                                    dtype=torch.float32, requires_grad=True)
    TrialPara = torch.linspace(mu_min,mu_max,50, 
                                    dtype=torch.float32, requires_grad=True)
    TrialPara = TrialPara[:,None] # Add axis so that dimensions match

    if not TrainingRequired:
        # Load pre trained model
        if os.path.isfile('TrainedModels/'+name_model):
            BeamROM.load_state_dict(torch.load('TrainedModels/'+name_model))
            print('************ LOADING MODEL COMPLETE ***********\n')
        else: 
            TrainingRequired = True
            print('**** WARNING NO PRE TRAINED MODEL WAS FOUND ***\n')

    if TrainingRequired:
        if BoolGPU:
            BeamROM.to(mps_device)
            TrialCoordinates = TrialCoordinates.to(mps_device)
            TrialPara = TrialPara.to(mps_device)
            BeamROM(TrialCoordinates,TrialPara)

        # Train model
        # BeamROM.UnFreeze_Mesh()
        # BeamROM.UnFreeze_Para()
        optimizer = torch.optim.Adam(BeamROM.parameters(), lr=learning_rate)
        Loss_vect =  Training_NeuROM(BeamROM, A, L, TrialCoordinates,TrialPara, optimizer, n_epochs, BoolCompareNorms, nn.MSELoss())

        # Save model
        if SaveModel:
            torch.save(BeamROM.state_dict(), 'TrainedModels/'+name_model)



#%% Post-processing
if BoolPlotPost:
    Pplot.Plot_Parametric_Young(BeamROM,TrialCoordinates,A,AnalyticSolution,name_model)
    Pplot.Plot_Parametric_Young_Interactive(BeamROM,TrialCoordinates,A,AnalyticSolution,name_model)
    Pplot.PlotModes(BeamROM,TrialCoordinates,TrialPara,A,AnalyticSolution,name_model)

