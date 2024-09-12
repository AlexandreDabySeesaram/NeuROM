#%% Libraries import
# import HiDeNN library
from HiDeNN_PDE import MeshNN, NeuROM, MeshNN_2D, MeshNN_1D
# Import pre-processing functions
import src.Pre_processing as pre
# Import torch librairies
import torch
import torch.nn as nn
# Import mechanical functions
from src.PDE_Library import RHS, PotentialEnergyVectorised, \
     Derivative, AnalyticGradientSolution, AnalyticSolution, AnalyticBiParametricSolution
# Import Training funcitons
from src.Training import Test_GenerateShapeFunctions, Training_InitialStage, \
     Training_FinalStageLBFGS, FilterTrainingData, Training_NeuROM, Training_NeuROM_FinalStageLBFGS, \
     Mixed_Training_InitialStage, Training_FinalStageLBFGS_Mixed, Training_2D_NeuROM, Training_2D_FEM, Training_1D_FEM_LBFGS,\
     Training_1D_FEM_Gradient_Descent, Training_1D_Mixed_LBFGS
#Import post processing libraries
import Post.Plots as Pplot
import time
import os
import torch._dynamo as dynamo
mps_device = torch.device("mps")
from importlib import reload  # Python 3.4+
# from src import MyHeaders
import tomllib

#%% Specify default configuratoin file

####################################################
###                                              ###
###                  USER INPUT                  ###
###                                              ###
####################################################

Default_config_file = 'Configuration/config_1D_Mixed.toml'

####################################################
###                                              ###
####################################################
#%% Import config file
# Read script arguments
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-cf',type=str, help = 'path to the desired configuration file', default=Default_config_file, action = 'store')
    
    jupyter = False
    # jupyter = MyHeaders.is_notebook()
    if jupyter:
        args = parser.parse_args('')
    else:
        args = parser.parse_args()
    inputs = vars(args)
    print(f"* Executing job in {args.cf}")

# Add possibility to specify name of config file with argparse
with open(args.cf, mode="rb") as f:
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
                                order         = config["interpolation"]["order_u"],
                                np            = config["interpolation"]["np"],
                                MaxElemSize2D = config["interpolation"]["MaxElemSize2D"]
                            )
Excluded = []
Mesh_object_u = pre.Mesh( 
                                config["geometry"]["Name"],                 # Create the mesh object
                                MaxElemSize, 
                                config["interpolation"]["order_u"], 
                                config["interpolation"]["dimension"]
                        )

Mesh_object_u.AddBorders(config["Borders_u"]["Borders"])
Mesh_object_u.AddBCs(                                                         # Include Boundary physical domains infos (BCs+volume)
                                config["geometry"]["Volume_element"],
                                Excluded,
                                config["DirichletDictionryList_u"]
                    )   
                    
Mesh_object_u.MeshGeo()                                                       # Mesh the .geo file if .msh does not exist
Mesh_object_u.ReadMesh() 
Mesh_object_u.AssemblyMatrix()                                            # Build the assembly weight matrix

Mesh_object_du = pre.Mesh( 
                                config["geometry"]["Name"],                 # Create the mesh object
                                MaxElemSize, 
                                config["interpolation"]["order_du"], 
                                config["interpolation"]["dimension"]
                        )
Mesh_object_du.AddBorders(config["Borders_du"]["Borders"])
Mesh_object_du.AddBCs(                                                         # Include Boundary physical domains infos (BCs+volume)
                                config["geometry"]["Volume_element"],
                                Excluded,
                                config["DirichletDictionryList_du"]
                    ) 

Mesh_object_du.MeshGeo()                                                       # Mesh the .geo file if .msh does not exist
Mesh_object_du.ReadMesh()   
Mesh_object_du.AssemblyMatrix()                                            # Build the assembly weight matrix

if int(Mesh_object_u.dim) != int(Mesh_object_u.dimension):
    raise ValueError("The dimension of the provided geometry does not match the job dimension")

#%% Application of the Space HiDeNN
match config["interpolation"]["dimension"]:
    case 1:
        Model_FEM_u = MeshNN(Mesh_object_u)
        Model_FEM_du = MeshNN(Mesh_object_du)


# Set the coordinates as trainable
Model_FEM_u.UnFreeze_Mesh()
Model_FEM_du.UnFreeze_Mesh()
# Set the coordinates as untrainable
Model_FEM_u.Freeze_Mesh()
Model_FEM_du.Freeze_Mesh()

# Make nodal values trainable (except the BC). Default choice 
Model_FEM_u.UnFreeze_FEM()
Model_FEM_du.UnFreeze_FEM()


if not config["solver"]["FrozenMesh"]:
    Model_FEM_u.UnFreeze_Mesh()    
    Model_FEM_du.UnFreeze_Mesh()    

match config["interpolation"]["dimension"]:
    case 1:
        Model_FEM_u, Model_FEM_du = Training_1D_Mixed_LBFGS(Model_FEM_u, Model_FEM_du, config, Mat)     




#%% Post-processing # # # # # # # # # # # # # # # # # # # # # # 

match config["interpolation"]["dimension"]:

    case 1:
        Pplot.Plot_Eval_1d(Model_FEM_u,config,Mat, Model_FEM_du)


            

                


       
# %%
