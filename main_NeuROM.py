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
     Derivative, AnalyticGradientSolution, AnalyticSolution, AnalyticBiParametricSolution
# Import Training funcitons
from Bin.Training import Test_GenerateShapeFunctions, Training_InitialStage, \
     Training_FinalStageLBFGS, FilterTrainingData, Training_NeuROM, Training_NeuROM_FinalStageLBFGS, \
     Mixed_Training_InitialStage, Training_FinalStageLBFGS_Mixed
#Import post processing libraries
import Post.Plots as Pplot
import time
import os
import torch._dynamo as dynamo
mps_device = torch.device("mps")
from importlib import reload  # Python 3.4+

#%% Options & Hyperparameters
# Solver
TrainingStrategy = 'Integral'                       # 'Integral' or 'Mixed'
ParametricStudy = True                              # Boolean to switch between plain HiDeNN and NeuROM
BiPara = True                                       # Enable bi-stifness beam in 1D
n_modes_max = 100                                   # Maximum number of modes in the Tensor Decomposition
n_modes_ini = 1                                     # Initial number of modes in the Tensor Decomposition
loss_decrease_c = 1e-5                              # Stagnation criterion for the loss decay rate (mode addition criterion)

# hardware
BoolGPU = False                                     # Boolean enabling GPU computations (autograd function is not working currently on mac M2)
BoolCompile = False                                 # Enable compilation of the model

# Training
TrainingRequired = True                             # Boolean leading to Loading pre trained model or retraining from scratch
n_epochs = 3000                                     # Maximum number of iterations for the training stage
learning_rate = 0.001                               # optimizer learning rate
LoadPreviousModel = True                           # Boolean to enable reusing a previously trained model
BoolFilterTrainingData = True                       # Slightly move training samples if they are on the mesh nodes exactly

# post-process
BoolPlot = False                                    # Boolean for plots used for gif
BoolPlotPost = False                                # Boolean for plots used for Post
BoolCompareNorms = True                             # Boolean for comparing energy norm to L2 norm
SaveModel = False                                   # Boolean leading to Loading pre trained model or retraining from scratch
Visualisatoin_only = False

#%% Pre-processing (could be put in config file later)
# Definition of the space discretisation
dimension = 1                                      # Dimension of the study (1D or 2D)
order = 2                                          # Order of the shape functions

# Choose the geometry
match dimension:
    case 1:
        #1D:
        Name = 'Beam'
    case 2:
        #2D: 
        # Name = 'Rectangle'
        # Name = 'Square'
        Name = 'Hole'
        # Name = 'Hole_3'

# Defintition of the structure 
np = 20                                             # Number of Nodes in the Mesh in 1D
MaxElemSize2D = 1                                   # Maximum element size in the 2D mesh
L = 10                                              # Length of the Beam
A = 1                                               # Section of the beam
E = 5*1e-3                                          # Young's modulus (175 if 1D 5*1e-3 if 2D)

# Initialise meterial
Mat = pre.Material( flag_lame = False,              # If True should input lmbda and mu instead of E and nu
                    coef1 = E,                      # Young Modulus
                    coef2 = 0.3                     # Poisson's ratio
                    )

# defintion of the volume and boundary conditions 
Volume_element = 100                               # Volume element correspond to the 1D elem in 1D

DirichletDictionryList = [  {"Entity": 1, 
                             "Value": 0, 
                             "Normal":1, 
                             "Relation": False,
                              "Constitutive": False}, 
                            {"Entity": 2, 
                             "Value": 0.0, 
                             "Normal":1, 
                             "Relation": False, 
                             "Constitutive": False}
                        ]
#%% Create mesh object
# Definition of the (initial) element size of the mesh
MaxElemSize = pre.ElementSize(
                                dimension=dimension,
                                L = L,
                                order = order,
                                np = np,
                                MaxElemSize2D = MaxElemSize2D
                            )
Excluded = []
Mesh_object = pre.Mesh( Name,                       # Create the mesh object
                        MaxElemSize, 
                        order, 
                        dimension)

Mesh_object.AddBCs(Volume_element,Excluded,
                 DirichletDictionryList)            # Include Boundary physical domains infos (BCs+volume)
Mesh_object.MeshGeo()                               # Mesh the .geo file if .msh does not exist
Mesh_object.ReadMesh()                              # Parse the .msh file
Mesh_object.AssemblyMatrix()                        # Build the assembly weight matrix

#%% Application of the Space HiDeNN
Model_FEM = MeshNN(Mesh_object)                     # Build the model

# Set the coordinates as trainable
Model_FEM.UnFreeze_Mesh()
# Set the coordinates as untrainable
Model_FEM.Freeze_Mesh()

#%% Application of NeuROM
# Parameter space-definition

match dimension:
    case 1:
        # Stiffness
        para_1_min = 100                            # Minimum value for the 1st parameter
        para_1_max = 200                            # Maximum value for the 1st parameter
        N_para_1 = 10                               # Discretisation of the 1D parametric space

        para_2_min = 100                            # Minimum value for the 2nd parameter
        para_2_max = 200                            # Minimum value for the 2nd parameter
        N_para_2 = 10                               # Discretisation of the 1D parametric space
    case 2:
        # Stiffness
        para_1_min = 1e-3                           # Minimum value for the 1st parameter
        para_1_max = 10e-3                          # Maximum value for the 1st parameter
        N_para_1 = 10                               # Discretisation of the 1D parametric space
        # Angle
        para_2_min = 0                              # Minimum value for the 2nd parameter
        para_2_max = 2*torch.pi                     # Maximum value for the 2nd parameter
        N_para_2 = 30                               # Discretisation of the 1D parametric space

if BiPara:
    ParameterHypercube = torch.tensor([ [para_1_min,para_1_max,N_para_1],
                                        [para_2_min,para_2_max,N_para_2]])
else:
    ParameterHypercube = torch.tensor([[para_1_min,para_1_max,N_para_1]])

ROM_model = NeuROM(                               # Build the surrogate (reduced-order) model
                Mesh_object, 
                ParameterHypercube, 
                n_modes_ini,
                n_modes_max
                )

#%% Load coarser model  

PreviousFullModel = 'TrainedModels/1D_Bi_Stiffness_np_10'
if LoadPreviousModel:
    ROM_model.Init_from_previous(PreviousFullModel)

#%% Training 
ROM_model.Freeze_Mesh()                         # Set space mesh cordinates as untrainable
ROM_model.Freeze_MeshPara()                     # Set parameters mesh cordinates as untrainable

match dimension:
    case 1:
        # Nodale coordinates where the model is evaluated during training
        Training_coordinates = torch.tensor([[i/50] for i in range(2,500)], 
                                dtype=torch.float32, requires_grad=True)

Training_para_coordinates_1 = torch.linspace(
                            para_1_min,para_1_max,5*N_para_1, 
                            dtype=torch.float32, requires_grad=True
                            )
Training_para_coordinates_1 = Training_para_coordinates_1[:,None]

Training_para_coordinates_2 = torch.linspace(
                            para_2_min,para_2_max,5*N_para_2, 
                            dtype=torch.float32, requires_grad=True
                            )
Training_para_coordinates_2 = Training_para_coordinates_2[:,None] 

if BiPara:
    Training_para_coordinates_list = nn.ParameterList(
                                        (Training_para_coordinates_1,
                                        Training_para_coordinates_2))
else:
    Training_para_coordinates_list = [Training_para_coordinates_1]

if TrainingRequired:
    ROM_model.train()
    optimizer = torch.optim.Adam([p for p in ROM_model.parameters() if p.requires_grad], lr=learning_rate)
    match dimension:
        case 1:
            Loss_vect, L2_error, training_time, Mode_vect,Loss_decrease_vect = Training_NeuROM(ROM_model, A, L, Training_coordinates,Training_para_coordinates_list, optimizer, n_epochs,BiPara,loss_decrease_c = loss_decrease_c)
            Loss_vect, L2_error = Training_NeuROM_FinalStageLBFGS(ROM_model, A, L, Training_coordinates,Training_para_coordinates_list, optimizer, n_epochs, 10,Loss_vect,L2_error,training_time,BiPara)
        case 2:
            Loss_vect, Duration = Training_2D_NeuROM(   ROM_model, 
                                                        Training_para_coordinates_list, 
                                                        optimizer, 
                                                        ROM_model.Max_epochs,
                                                        Mat)
    ROM_model.eval()

    #%% Post-processing

    if BoolPlotPost:
        Pplot.Plot_Parametric_Young(ROM_model,Training_coordinates,A,AnalyticSolution,name_model)
        Pplot.Plot_Parametric_Young_Interactive(ROM_model,Training_coordinates,A,AnalyticSolution,name_model)
        Pplot.PlotModes(ROM_model,Training_coordinates,TrialPara,A,AnalyticSolution,name_model)
        Pplot.Plot_Compare_Loss2l2norm(Loss_vect,L2_error,'2StageTraining')

    if False:
        Space_modes = [ROM_model.Space_modes[l](Training_coordinates) for l in range(ROM_model.n_modes)]
        u_i = torch.cat(Space_modes,dim=1) 
    if Visualisatoin_only:
        Space_modes = [ROM_model.Space_modes[l](Training_coordinates) for l in range(ROM_model.n_modes)]
        u_i = torch.cat(Space_modes,dim=1) 

    # Pplot.Plot_Parametric_Young_Interactive(ROM_model,Training_coordinates,A,AnalyticSolution,name_model)
    # Pplot.AppInteractive(ROM_model,Training_coordinates,A,AnalyticSolution)

    u_eval = ROM_model(Training_coordinates,Training_para_coordinates_list)
    import matplotlib.pyplot as plt
    # Pplot.PlotModesBi(ROM_model,Training_coordinates,Training_para_coordinates_list,A,AnalyticSolution,name_model)
    # ROM_model = torch.load('TrainedModels/FullModel_BiParametric')


    if Visualisatoin_only:
        Training_coordinates = torch.tensor([[i/10] for i in range(2,100)], 
                                        dtype=torch.float32, requires_grad=True)
        TrialPara = torch.linspace(mu_min,mu_max,50, 
                                        dtype=torch.float32, requires_grad=True)
        TrialPara = TrialPara[:,None] # Add axis so that dimensions match

        TrialPara2 = torch.linspace(mu_min,mu_max,50, 
                                        dtype=torch.float32, requires_grad=True)
        TrialPara2 = TrialPara2[:,None] # Add axis so that dimensions match

        if BiPara:
            Training_para_coordinates_list = nn.ParameterList((TrialPara,TrialPara2))
        else:
            Training_para_coordinates_list = [TrialPara]
        ROM_model = torch.load('TrainedModels/FullModel_BiParametric')
        ROM_model = torch.load('TrainedModels/FullModel_BiParametric_np100')

        Pplot.Plot_BiParametric_Young_Interactive(ROM_model,Training_coordinates,A,AnalyticBiParametricSolution,name_model)
        Pplot.Plot_Loss_Modes(Mode_vect,Loss_vect,'Loss_Modes')
        Pplot.Plot_Lossdecay_Modes(Mode_vect,Loss_decrease_vect,'LossDecay_Modes',1e-5)
        Pplot.Plot_NegLoss_Modes(Mode_vect,Loss_vect,'Loss_ModesNeg',True)

# %%
