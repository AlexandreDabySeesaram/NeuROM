#%% Libraries import
# import HiDeNN library
from HiDeNN_PDE import MeshNN, NeuROM, MeshNN_2D
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
     Mixed_Training_InitialStage, Training_FinalStageLBFGS_Mixed, Training_2D_NeuROM
#Import post processing libraries
import Post.Plots as Pplot
import time
import os
import torch._dynamo as dynamo
mps_device = torch.device("mps")
from importlib import reload  # Python 3.4+

#%% Import config file
import tomllib
with open("Configuration/config_2D.toml", mode="rb") as f:
    config = tomllib.load(f)
    
#%% Initialise meterial
Mat = pre.Material(             flag_lame = False,                          # If True should input lmbda and mu instead of E and nu
                                coef1     = config["material"]["E"],        # Young Modulus
                                coef2     = config["material"]["nu"]        # Poisson's ratio
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
                                config["geometry"]["Name"],             # Create the mesh object
                                MaxElemSize, 
                                config["interpolation"]["order"], 
                                config["interpolation"]["dimension"]
                        )

Mesh_object.AddBorders(config["Borders"]["Borders"])
Mesh_object.AddBCs(                                                     # Include Boundary physical domains infos (BCs+volume)
                                config["geometry"]["Volume_element"],
                                Excluded,
                                config["DirichletDictionryList"]
                    )                   

Mesh_object.MeshGeo()                                                   # Mesh the .geo file if .msh does not exist
Mesh_object.ReadMesh()                                                  # Parse the .msh file

if config["interpolation"]["dimension"] ==1:
    Mesh_object.AssemblyMatrix()                                        # Build the assembly weight matrix

if int(Mesh_object.dim) != int(Mesh_object.dimension):
    raise ValueError("The dimension of the provided geometry does not match the job dimension")

#%% Application of the Space HiDeNN
match config["interpolation"]["dimension"]:
    case 1:     
        Model_FEM = MeshNN(Mesh_object)                                 # Build the model
    case 2:
        Model_FEM = MeshNN_2D(Mesh_object, n_components = 2)

# Set the coordinates as trainable
Model_FEM.UnFreeze_Mesh()
# Set the coordinates as untrainable
Model_FEM.Freeze_Mesh()

#%% Application of NeuROM
# Parameter space-definition

if config["solver"]["BiPara"]:
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
                                            config["solver"]["n_modes_ini"],
                                            config["solver"]["n_modes_max"]
                )

#%% Load coarser model  

PreviousFullModel = 'TrainedModels/1D_Bi_Stiffness_np_10'
if config["training"]["LoadPreviousModel"]:
    ROM_model.Init_from_previous(PreviousFullModel)

#%% Training 
ROM_model.Freeze_Mesh()                                                 # Set space mesh cordinates as untrainable
ROM_model.Freeze_MeshPara()                                             # Set parameters mesh cordinates as untrainable

ROM_model.TrainingParameters(   Stagnation_threshold = config["training"]["Stagnation_threshold"], 
                                Max_epochs = config["training"]["n_epochs"], 
                                learning_rate = config["training"]["learning_rate"])
                                
match config["interpolation"]["dimension"]:
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

if config["solver"]["BiPara"]:
    Training_para_coordinates_list = nn.ParameterList(
                                                        (Training_para_coordinates_1,
                                                        Training_para_coordinates_2))
else:
    Training_para_coordinates_list = [Training_para_coordinates_1]

if config["training"]["TrainingRequired"]:
    ROM_model.train()
    optimizer = torch.optim.Adam([p for p in ROM_model.parameters() if p.requires_grad], lr=config["training"]["learning_rate"])
    match config["interpolation"]["dimension"]:
        case 1:
            Loss_vect, L2_error, training_time, Mode_vect,Loss_decrease_vect = Training_NeuROM( ROM_model,
                                                                                                 config["geometry"]["A"], 
                                                                                                 config["geometry"]["L"], 
                                                                                                 Training_coordinates,
                                                                                                 Training_para_coordinates_list, 
                                                                                                 optimizer, 
                                                                                                 config["training"]["n_epochs"],
                                                                                                 config["solver"]["BiPara"],
                                                                                                 loss_decrease_c = config["training"]["loss_decrease_c"])

            Loss_vect, L2_error = Training_NeuROM_FinalStageLBFGS(  ROM_model, 
                                                                    config["geometry"]["A"], 
                                                                    config["geometry"]["L"], 
                                                                    Training_coordinates,
                                                                    Training_para_coordinates_list, 
                                                                    optimizer, 
                                                                    config["training"]["n_epochs"], 
                                                                    10,
                                                                    Loss_vect,
                                                                    L2_error,
                                                                    training_time,
                                                                    config["solver"]["BiPara"])
        case 2:
            Loss_vect, Duration = Training_2D_NeuROM(   ROM_model, 
                                                        Training_para_coordinates_list, 
                                                        optimizer, 
                                                        ROM_model.Max_epochs,
                                                        Mat)
    ROM_model.eval()

    #%% Post-processing
# Pplot.Plot_BiParametric_Young_Interactive(ROM_model,Training_coordinates,config["geometry"]["A"],AnalyticBiParametricSolution,'name_model')