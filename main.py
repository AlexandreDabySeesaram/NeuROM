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
     Mixed_Training_InitialStage, Training_FinalStageLBFGS_Mixed, Training_2D_NeuROM, Training_2D_FEM
#Import post processing libraries
import Post.Plots as Pplot
import time
import os
import torch._dynamo as dynamo
mps_device = torch.device("mps")
from importlib import reload  # Python 3.4+

#%% Import config file
import tomllib

# Add possibility to specify name of config file with argparse
with open("Configuration/config_2D.toml", mode="rb") as f:
    config = tomllib.load(f)
    
#%% Initialise material
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
                                config["geometry"]["Name"],                 # Create the mesh object
                                MaxElemSize, 
                                config["interpolation"]["order"], 
                                config["interpolation"]["dimension"]
                        )

Mesh_object.AddBorders(config["Borders"]["Borders"])
Mesh_object.AddBCs(                                                         # Include Boundary physical domains infos (BCs+volume)
                                config["geometry"]["Volume_element"],
                                Excluded,
                                config["DirichletDictionryList"]
                    )                   

Mesh_object.MeshGeo()                                                       # Mesh the .geo file if .msh does not exist
Mesh_object.ReadMesh()                                                      # Parse the .msh file
Mesh_object.ExportMeshVtk()

if config["interpolation"]["dimension"] ==1:
    Mesh_object.AssemblyMatrix()                                            # Build the assembly weight matrix

if int(Mesh_object.dim) != int(Mesh_object.dimension):
    raise ValueError("The dimension of the provided geometry does not match the job dimension")

#%% Application of the Space HiDeNN
match config["interpolation"]["dimension"]:
    case 1:     
        Model_FEM = MeshNN(Mesh_object)                                     # Build the model
    case 2:
        Model_FEM = MeshNN_2D(Mesh_object, n_components = 2)

# Set the coordinates as trainable
Model_FEM.UnFreeze_Mesh()
# Set the coordinates as untrainable
Model_FEM.Freeze_Mesh()
Model_FEM.UnFreeze_FEM()

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

ROM_model = NeuROM(                                                         # Build the surrogate (reduced-order) model
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
if config["solver"]["ParametricStudy"]: 
    ROM_model.Freeze_Mesh()                                                 # Set space mesh cordinates as untrainable
    ROM_model.Freeze_MeshPara()                                             # Set parameters mesh cordinates as untrainable

    ROM_model.TrainingParameters(   loss_decrease_c = config["training"]["loss_decrease_c"], 
                                    Max_epochs = config["training"]["n_epochs"], 
                                    learning_rate = config["training"]["learning_rate"])
                                    
    if config["training"]["TrainingRequired"]:
        ROM_model.train()
        optimizer = torch.optim.Adam([p for p in ROM_model.parameters() if p.requires_grad], lr=config["training"]["learning_rate"])
        match config["interpolation"]["dimension"]:
            case 1:
                Training_NeuROM(ROM_model,config,optimizer)                 # First stage of training (ADAM)

                Training_NeuROM_FinalStageLBFGS(ROM_model,config)           # Second stage of training (LBFGS)
            case 2:
                Training_2D_NeuROM(ROM_model, config, optimizer, Mat)
        ROM_model.eval()
else:
    Model_FEM.TrainingParameters(   loss_decrease_c = config["training"]["loss_decrease_c"], 
                                    Max_epochs = config["training"]["n_epochs"], 
                                    learning_rate = config["training"]["learning_rate"])
    Model_FEM = Training_2D_FEM(Model_FEM, config, Mat)


#%% Post-processing

print("*************** POST-PROCESSING ***************\n")

if config["solver"]["ParametricStudy"]:
    Pplot.Plot_BiParametric_Young_Interactive(ROM_model,Training_coordinates,config["geometry"]["A"],AnalyticBiParametricSolution,'name_model')
else:
    if config["postprocess"]["exportVTK"]:
        Pplot.ExportFinalResult_VTK(Model_FEM,Mat,config["postprocess"]["Name_export"])
        Pplot.ExportHistoryResult_VTK(Model_FEM,Mat,config["postprocess"]["Name_export"])
       
# %%
