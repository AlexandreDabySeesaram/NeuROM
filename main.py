#%% Libraries import
# import HiDeNN library
from HiDeNN_PDE import MeshNN, NeuROM, MeshNN_2D, MeshNN_1D
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
from Bin import MyHeaders
import tomllib
import numpy as np


####################################################
###                                              ###
###             /!\   WARNING   /!\              ###
###      import vtkmodules.util.pickle_support   ###
###         in serialization.py of poytorch      ###
###                                              ###
#################################################### 
#%% Specify default configuratoin file

####################################################
###                                              ###
###                  USER INPUT                  ###
###                                              ###
####################################################

Default_config_file = 'Configuration/config_2D_ROM.toml'
# Default_config_file = 'Configuration/config_1D.toml'

####################################################
###                                              ###
####################################################
#%% Import config file
# Read script arguments
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-cf',type=str, help = 'path to the desired configuration file', default=Default_config_file, action = 'store')
    jupyter = MyHeaders.is_notebook()
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

match config["interpolation"]["dimension"]:
    case 1:
        if config["solver"]["IntegralMethod"] == "Gaussian_quad":
            Mesh_object.ExportMeshVtk1D()
    case 2:
        Mesh_object.ExportMeshVtk()

# if config["interpolation"]["dimension"] ==2:
#     Mesh_object.ExportMeshVtk()

if config["interpolation"]["dimension"] ==1 and config["solver"]["IntegralMethod"] == "Trapezoidal":
    Mesh_object.AssemblyMatrix()                                            # Build the assembly weight matrix

if int(Mesh_object.dim) != int(Mesh_object.dimension):
    raise ValueError("The dimension of the provided geometry does not match the job dimension")

#%% Application of the Space HiDeNN
match config["interpolation"]["dimension"]:
    case 1:
        match config["solver"]["IntegralMethod"]:                           # Build the model
            case "Gaussian_quad":
                Model_FEM = MeshNN_1D(Mesh_object, config["interpolation"]["n_integr_points"])  
            case "Trapezoidal":
                Model_FEM = MeshNN(Mesh_object)
    case 2:
        Model_FEM = MeshNN_2D(Mesh_object, n_components = 2)

# Set the coordinates as trainable
Model_FEM.UnFreeze_Mesh()
# Set the coordinates as untrainable
Model_FEM.Freeze_Mesh()
if not config["solver"]["FrozenMesh"]:
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
                                            config,
                                            config["solver"]["n_modes_ini"],
                                            config["solver"]["n_modes_max"]
                )

#%% Load coarser model  

match config["solver"]["BiPara"]:
    case True:
        PreviousFullModel = 'TrainedModels/1D_Bi_Stiffness_np_10'
    case False:
        match config["solver"]["IntegralMethod"]:
            case "Trapezoidal":
                PreviousFullModel = 'TrainedModels/1D_Mono_Stiffness_np_100'
            case "Gaussian_quad":
                # PreviousFullModel = 'TrainedModels/1D_Mono_Stiffness_Gauss_np_40'
                PreviousFullModel = 'TrainedModels/1D_Mono_Stiffness_Gauss_np_100'


if config["training"]["LoadPreviousModel"]:
    ROM_model.Init_from_previous(PreviousFullModel)

#%% Training 
ROM_model.Freeze_Mesh()                                                     # Set space mesh cordinates as untrainable
ROM_model.Freeze_MeshPara()                                                 # Set parameters mesh cordinates as untrainable

if config["solver"]["ParametricStudy"]: 
    if not config["solver"]["FrozenMesh"]:
        ROM_model.UnFreeze_Mesh()                                             # Set space mesh cordinates as trainable
    if not config["solver"]["FrozenParaMesh"]:
        ROM_model.UnFreeze_MeshPara()                                         # Set parameters mesh cordinates as trainable

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
                # Training_2D_NeuROM(ROM_model, config, optimizer, Mat)
                Training_NeuROM(ROM_model, config, optimizer, Mat)                # First stage of training (ADAM)
        ROM_model.eval()
else:
    Model_FEM.TrainingParameters(   loss_decrease_c = config["training"]["loss_decrease_c"], 
                                    Max_epochs = config["training"]["n_epochs"], 
                                    learning_rate = config["training"]["learning_rate"])
    Model_FEM = Training_2D_FEM(Model_FEM, config, Mat)


#%% Post-processing

print("*************** POST-PROCESSING ***************\n")
if config["solver"]["ParametricStudy"]:
    Training_coordinates = torch.tensor([[i/50] for i in range(2,500)], 
                    dtype=torch.float32, 
                    requires_grad=True)

    if min(ROM_model.training_recap["Loss_vect"]) > 0:                  # Find sign of the converged loss
        sign = "Positive"
    else:
        sign = "Negative"
    if config["solver"]["BiPara"]:                                      # define type of parametric study for saving files
        Study = "_BiPara"
    else: 
        Study = "_MonoPara" 
    val = str(np.format_float_scientific(1e15, precision=2))            # Convergence criterion

    if config["postprocess"]["Plot_loss_mode"]:                         # Plot loss and modes
        Pplot.Plot_PosNegLoss_Modes(ROM_model.training_recap["Mode_vect"],ROM_model.training_recap["Loss_vect"],
                                    'Loss_Modes'+"_"+config["geometry"]["Name"]+Study+"_"+val
                                    , sign = sign,tikz = True)
    if config["postprocess"]["Plot_loss_decay_mode"]:                   # Plot loss rate and modes
        Pplot.Plot_Lossdecay_Modes(ROM_model.training_recap["Mode_vect"],ROM_model.training_recap["Loss_decrease_vect"],
                                    'Loss_rate_Modes'+"_"+config["geometry"]["Name"]+"_"+val,True)
    
    if config["postprocess"]["Interactive_pltot"]:

        match config["interpolation"]["dimension"]:
            case 1:
                if config["solver"]["BiPara"]:
                    Pplot.Plot_BiParametric_Young_Interactive(  ROM_model,
                                                Training_coordinates,
                                                config["geometry"]["A"],
                                                AnalyticBiParametricSolution,
                                                'name_model')
                else:
                    ROM_model.eval()
                    Pplot.Plot_Parametric_Young_Interactive(    ROM_model,
                                                Training_coordinates,
                                                config["geometry"]["A"],
                                                AnalyticSolution,
                                                'name_model')                                
            case 2:
                import pyvista as pv
                import vtk
                import meshio
                filename = 'Geometries/'+Mesh_object.name_mesh
                mesh = pv.read(filename)
                Nodes = np.stack(Mesh_object.Nodes)

                match config["postprocess"]["PyVista_Type"]:
                    case "Frame":
                        parameter = 5e-3
                        E = torch.tensor([parameter],dtype=torch.float32)
                        E = E[:,None] # Add axis so that dimensions match
                        theta = torch.tensor([torch.pi/3],dtype=torch.float32)
                        theta = theta[:,None] # Add axis so that dimensions match

                        Para_coord_list = nn.ParameterList((E,theta))
                        ROM_model.eval()
                        u_sol = ROM_model(torch.tensor(Nodes[:,1:]),Para_coord_list)

                        match ROM_model.n_para:
                            case 1:
                                u = torch.stack([(u_sol[0,:,0]),(u_sol[1,:,0]),torch.zeros(u_sol[0,:,0].shape[0])],dim=1)
                            case 2:
                                u = torch.stack([(u_sol[0,:,0,0]),(u_sol[1,:,0,0]),torch.zeros(u_sol[0,:,0,0].shape[0])],dim=1)
                        # Plot the mesh
                        scalar_field_name = 'Ux'
                        mesh.point_data['U'] = u.data
                        mesh.point_data['Ux'] = u[:,0].data
                        mesh.point_data['Uy'] = u[:,1].data
                        mesh.point_data['Uz'] = u[:,2].data

                        plotter = pv.Plotter()
                        plotter.add_mesh(mesh, scalars=scalar_field_name, cmap='viridis', scalar_bar_args={'title': 'Displacement', 'vertical': True})
                        plotter.show()
                    case "Static":

                        plotter = pv.Plotter(shape=(1, 2))

                        plotter.subplot(0, 0)
                        filename = 'Geometries/'+Mesh_object.name_mesh
                        mesh3 = pv.read(filename)
                        # Define the parameter to adjust and its initial value
                        parameter = 1e-3

                        Param_trial = torch.tensor([parameter],dtype=torch.float32)
                        Param_trial = Param_trial[:,None] # Add axis so that dimensions match
                        Para_coord_list = nn.ParameterList((Param_trial,Param_trial))

                        ROM_model.eval()
                        u_sol = ROM_model(torch.tensor(Nodes[:,1:]),Para_coord_list)
                        match ROM_model.n_para:
                            case 1:
                                u3 = torch.stack([(u_sol[0,:,0]),(u_sol[1,:,0]),torch.zeros(u_sol[0,:,0].shape[0])],dim=1)
                            case 2:
                                u3 = torch.stack([(u_sol[0,:,0,0]),(u_sol[1,:,0,0]),torch.zeros(u_sol[0,:,0,0].shape[0])],dim=1)
                        mesh3.point_data['U'] = u3.data
                        mesh3.point_data['Ux'] = u3[:,0].data
                        mesh3.point_data['Uy'] = u3[:,1].data
                        mesh3.point_data['Uz'] = u3[:,2].data
                        u3[:,2]+=0
                        plotter.add_mesh(mesh3.warp_by_vector(vectors="U",factor=20.0,inplace=True), scalars='Uy', cmap='viridis', scalar_bar_args={r'title': 'Uy, theta = 0', 'vertical': False}, show_edges=True)

                        # Function to update the solution based on the parameter
                        def update_solution_E(value):
                            # plotter.clear()
                            parameter = value
                            stiffness = torch.tensor([parameter],dtype=torch.float32)
                            stiffness = stiffness[:,None] # Add axis so that dimensions match
                            Param_trial = torch.tensor([0],dtype=torch.float32)
                            Param_trial = Param_trial[:,None] # Add axis so that dimensions match
                            Para_coord_list = nn.ParameterList((stiffness,Param_trial))
                            u_sol = ROM_model(torch.tensor(Nodes[:,1:]),Para_coord_list)
                            match ROM_model.n_para:
                                case 1:
                                    u3 = torch.stack([(u_sol[0,:,0]),(u_sol[1,:,0]),torch.zeros(u_sol[0,:,0].shape[0])],dim=1)
                                case 2:
                                    u3 = torch.stack([(u_sol[0,:,0,0]),(u_sol[1,:,0,0]),torch.zeros(u_sol[0,:,0,0].shape[0])],dim=1)
                            mesh3 = pv.read(filename)
                            u3[:,2]+=200*value
                            # mesh.warp_by_vector(vectors="U",factor=-20.0,inplace=True)
                            mesh3.point_data['U'] = u3.data
                            mesh3.point_data['Ux'] = u3[:,0].data
                            mesh3.point_data['Uy'] = u3[:,1].data
                            mesh3.point_data['Uz'] = u3[:,2].data
                            plotter.add_mesh(mesh3.warp_by_vector(vectors="U",factor=20.0,inplace=True), scalars='Uy', cmap='viridis', scalar_bar_args={r'title': 'Uy, theta = 0', 'vertical': False}, show_edges=True)
                            return
                        labels = dict(zlabel='E (MPa)', xlabel='x (mm)', ylabel='y (mm)')

                        parameters_vect = [2e-3,3e-3,4e-3,5e-3,6e-3,7e-3,8e-3,9e-3,10e-3]

                        for param in parameters_vect:
                            update_solution_E(param)
                        plotter.show_grid(
                            color='gray',
                            location='outer',
                            grid='back',
                            ticks='outside',
                            xtitle='x (mm)',
                            ytitle='y (mm)',
                            ztitle='E (MPa)',
                            font_size=10,
                        )
                        plotter.add_text("theta = 0", font_size=10)

                        plotter.add_axes(**labels)
                        # plotter.show()




                        plotter.subplot(0, 1)

                        filename = 'Geometries/'+Mesh_object.name_mesh
                        mesh2 = pv.read(filename)
                        # Define the parameter to adjust and its initial value
                        parameter = 1e-3

                        Param_trial = torch.tensor([parameter],dtype=torch.float32)
                        Param_trial = Param_trial[:,None] # Add axis so that dimensions match
                        Para_coord_list = nn.ParameterList((Param_trial,Param_trial))

                        ROM_model.eval()
                        u_sol = ROM_model(torch.tensor(Nodes[:,1:]),Para_coord_list)
                        match ROM_model.n_para:
                            case 1:
                                u2 = torch.stack([(u_sol[0,:,0]),(u_sol[1,:,0]),torch.zeros(u_sol[0,:,0].shape[0])],dim=1)
                            case 2:
                                u2 = torch.stack([(u_sol[0,:,0,0]),(u_sol[1,:,0,0]),torch.zeros(u_sol[0,:,0,0].shape[0])],dim=1)
                        mesh2.point_data['U'] = u2.data
                        mesh2.point_data['Ux'] = u2[:,0].data
                        mesh2.point_data['Uy'] = u2[:,1].data
                        mesh2.point_data['Uz'] = u2[:,2].data
                        u2[:,2]+=0
                        # plotter.add_mesh(mesh.warp_by_vector(vectors="U",factor=20.0,inplace=True), scalars=scalar_field_name, cmap='viridis', scalar_bar_args={'title': 'Displacement', 'vertical': False}, show_edges=True)

                        # Function to update the solution based on the parameter
                        def update_solution_t(value):
                            # plotter.clear()
                            parameter = value
                            stiffness = torch.tensor([3e-3],dtype=torch.float32)
                            stiffness = stiffness[:,None] # Add axis so that dimensions match
                            Param_trial = torch.tensor([parameter],dtype=torch.float32)
                            Param_trial = Param_trial[:,None] # Add axis so that dimensions match
                            Para_coord_list = nn.ParameterList((stiffness,Param_trial))
                            u_sol = ROM_model(torch.tensor(Nodes[:,1:]),Para_coord_list)
                            match ROM_model.n_para:
                                case 1:
                                    u2 = torch.stack([(u_sol[0,:,0]),(u_sol[1,:,0]),torch.zeros(u_sol[0,:,0].shape[0])],dim=1)
                                case 2:
                                    u2 = torch.stack([(u_sol[0,:,0,0]),(u_sol[1,:,0,0]),torch.zeros(u_sol[0,:,0,0].shape[0])],dim=1)
                            mesh2 = pv.read(filename)
                            u2[:,2]+=0.25*value
                            # mesh.warp_by_vector(vectors="U",factor=-20.0,inplace=True)
                            mesh2.point_data['U'] = u2.data
                            mesh2.point_data['Ux'] = u2[:,0].data
                            mesh2.point_data['Uy'] = u2[:,1].data
                            mesh2.point_data['Uz'] = u2[:,2].data
                            plotter.add_mesh(mesh2.warp_by_vector(vectors="U",factor=20.0,inplace=True), scalars='Uy', cmap='viridis', scalar_bar_args={r'title': 'Uy, E = 5e-3', 'vertical': False}, show_edges=True)
                            return
                        labels = dict(zlabel='E (MPa)', xlabel='x (mm)', ylabel='y (mm)')

                        parameters_vect = [0,torch.pi/4,torch.pi/2,3*torch.pi/4,torch.pi,5*torch.pi/4,3*torch.pi/2,7*torch.pi/4,2*torch.pi]

                        for param in parameters_vect:
                            update_solution_t(param)
                        plotter.show_grid(
                            color='gray',
                            location='outer',
                            grid='back',
                            ticks='outside',
                            xtitle='x (mm)',
                            ytitle='y (mm)',
                            ztitle='theta (rad)',
                            font_size=10,
                        )
                        plotter.add_axes(**labels)
                        plotter.add_text("E = 5e-3", font_size=10)

                        plotter.show()

           
else:
    if config["postprocess"]["exportVTK"]:
        Pplot.ExportFinalResult_VTK(Model_FEM,Mat,config["postprocess"]["Name_export"]+
        "_"+config["geometry"]["Name"])
        Pplot.ExportHistoryResult_VTK(Model_FEM,Mat,config["postprocess"]["Name_export"]+
        "_"+config["geometry"]["Name"])
       
# %%
