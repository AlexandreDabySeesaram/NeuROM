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

#%% Import config file
import tomllib
with open("Configuration/config_1D.toml", mode="rb") as f:
    config = tomllib.load(f)
    
#%% Initialise meterial
Mat = pre.Material( flag_lame = False,                                  # If True should input lmbda and mu instead of E and nu
                    coef1     = config["material"]["E"],                # Young Modulus
                    coef2     = config["material"]["nu"]                # Poisson's ratio
                    )


#%% Create mesh object
# Definition of the (initial) element size of the mesh
MaxElemSize = pre.ElementSize(
                                dimension     = config["interpolation"]["dimension"],
                                L             = config["geometry"]["L"],
                                order         = config["interpolation"]["order"],
                                np            = config["interpolation"]["np"],
                                MaxElemSize2D = config["interpolation"]["MaxElemSize2D"]
                            )
Excluded = []
Mesh_object = pre.Mesh( 
                        config["geometry"]["Name"],                       # Create the mesh object
                        MaxElemSize, 
                        config["interpolation"]["order"], 
                        config["interpolation"]["dimension"]
                        )

if int(Mesh_object.dim) != int(Mesh_object.dimension):
    raise ValueError("The dimension of the provided geometry does not match the job dimension")

Mesh_object.AddBCs(                                                     # Include Boundary physical domains infos (BCs+volume)
                    config["geometry"]["Volume_element"],
                    Excluded,
                    config["DirichletDictionryList"]
                    )                   

Mesh_object.MeshGeo()                                                   # Mesh the .geo file if .msh does not exist
Mesh_object.ReadMesh()                                                  # Parse the .msh file
Mesh_object.AssemblyMatrix()                                            # Build the assembly weight matrix

#%% Application of the Space HiDeNN
Model_FEM = MeshNN(Mesh_object)                                         # Build the model

# Set the coordinates as trainable
Model_FEM.UnFreeze_Mesh()
# Set the coordinates as untrainable
Model_FEM.Freeze_Mesh()

#%% Application of NeuROM
# Parameter space-definition

if BiPara:
    ParameterHypercube = torch.tensor([ [   config["parameters"]["para_1_min"],
                                            config["parameters"]["para_1_max"],
                                            config["parameters"]["N_para_1"]],
                                        [   config["parameters"]["para_2_min"],
                                            config["parameters"]["para_2_max"],
                                            config["parameters"]["N_para_2"]]])
else:
    ParameterHypercube = torch.tensor([[    config["parameters"]["para_1_min"],
                                            config["parameters"]["para_1_max"],
                                            config["parameters"]["N_para_1"]]])

ROM_model = NeuROM(                                                     # Build the surrogate (reduced-order) model
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
ROM_model.Freeze_Mesh()                                                 # Set space mesh cordinates as untrainable
ROM_model.Freeze_MeshPara()                                             # Set parameters mesh cordinates as untrainable

match dimension:
    case 1:
        # Nodale coordinates where the model is evaluated during training
        Training_coordinates = torch.tensor([[i/50] for i in range(2,500)], 
                                            dtype=torch.float32, 
                                            requires_grad=True)

Training_para_coordinates_1 = torch.linspace(
                                                config["parameters"]["para_1_min"],
                                                config["parameters"]["para_1_max"],
                                                5*config["parameters"]["N_para_1"], 
                                                dtype=torch.float32, 
                                                requires_grad=True
                                            )

Training_para_coordinates_1 = Training_para_coordinates_1[:,None]

Training_para_coordinates_2 = torch.linspace(
                                                config["parameters"]["para_2_min"],
                                                config["parameters"]["para_2_max"],
                                                5*config["parameters"]["N_para_2"], 
                                                dtype=torch.float32, 
                                                requires_grad=True
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
            Loss_vect, L2_error, training_time, Mode_vect,Loss_decrease_vect = Training_NeuROM( ROM_model,
                                                                                                 A, 
                                                                                                 L, 
                                                                                                 Training_coordinates,
                                                                                                 Training_para_coordinates_list, 
                                                                                                 optimizer, 
                                                                                                 n_epochs,
                                                                                                 BiPara,
                                                                                                 loss_decrease_c = loss_decrease_c)

            Loss_vect, L2_error = Training_NeuROM_FinalStageLBFGS(  ROM_model, 
                                                                    A, 
                                                                    L, 
                                                                    Training_coordinates,
                                                                    Training_para_coordinates_list, 
                                                                    optimizer, 
                                                                    n_epochs, 
                                                                    10,
                                                                    Loss_vect,
                                                                    L2_error,
                                                                    training_time,
                                                                    BiPara)
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
