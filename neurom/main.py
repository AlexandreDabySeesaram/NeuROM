





def main():
    #%% Libraries import
    # import HiDeNN library
    from .HiDeNN_PDE import MeshNN, NeuROM, MeshNN_2D, MeshNN_1D
    # Import pre-processing functions
    from .src import Pre_processing as pre
    # Import torch librairies
    import torch
    import torch.nn as nn
    # Import mechanical functions
    from .src.PDE_Library import RHS, PotentialEnergyVectorised, \
        Derivative, AnalyticGradientSolution, AnalyticSolution, AnalyticBiParametricSolution
    # Import Training funcitons
    from .src.Training import Test_GenerateShapeFunctions, Training_InitialStage, \
        Training_FinalStageLBFGS, FilterTrainingData, Training_NeuROM, Training_NeuROM_FinalStageLBFGS, \
        Mixed_Training_InitialStage, Training_FinalStageLBFGS_Mixed, Training_2D_NeuROM, Training_2D_FEM, Training_NeuROM_multi_level, Training_1D_FEM_LBFGS,\
        Training_1D_FEM_Gradient_Descent
    #Import post processing libraries
    from .Post import Plots as Pplot
    import time
    import os
    import torch._dynamo as dynamo
    mps_device = torch.device("mps")
    from importlib import reload  # Python 3.4+
    import numpy as np

    ####################################################
    ###                                              ###
    ###             /!\   WARNING   /!\              ###
    ###      import vtkmodules.util.pickle_support   ###
    ###         in serialization.py of pytorch       ###
    ###                                              ###
    #################################################### 
    #%% Specify default configuratoin file

    ####################################################
    ###                                              ###
    ###                  USER INPUT                  ###
    ###                                              ###
    ####################################################



    ####################################################
    ###                                              ###
    ####################################################
    #%% Import config file
    # Read script arguments

    # Add possibility to specify name of config file with argparse
    with open(args.cf, mode="rb") as f:
        config = tomllib.load(f)

    #%% Initialise hardware

    match config["hardware"]["FloatPrecision"]:
        case "simple":
            tensor_float_type = torch.float32
        case "double":
            tensor_float_type = torch.float64
        case "half":
            tensor_float_type = torch.float16

    if config["interpolation"]["dimension"] > 1:
        if (config['training']['h_adapt_MaxGeneration'] > 1 and not config['solver']['FrozenMesh']) and (config["hardware"]["device"] == 'mps' or config["hardware"]["device"] == 'cuda'):
            print('***** WARNING: Changing device to CPU due to h-adaptivity not implemented on GPU yet')
            config["hardware"]["device"] = 'cpu'


    match config["hardware"]["device"]:
        case 'mps':
            device = torch.device("mps")
            if tensor_float_type != torch.float32:
                print('***** WARNING: Changing float precision to simple for mps compatibility')
                tensor_float_type = torch.float32
        case 'cuda':
            device = torch.device("cuda")
        case 'cpu':
            device = torch.device("cpu")
            


    #%% Initialise material

    if config["interpolation"]["dimension"] == 1:
        Mat = pre.Material(             flag_lame = True,                               # If True should input lmbda and mu instead of E and nu
                                        coef1     = config["material"]["E"],            # Young Modulus
                                        coef2     = config["geometry"]["A"]             # Section area of the 1D bar
                            )
    elif config["interpolation"]["dimension"] == 2:
        try:
            Mat = pre.Material(         flag_lame = False,                              # If True should input lmbda and mu instead of E and nu
                                        coef1     = config["material"]["E"],            # Young Modulus
                                        coef2     = config["material"]["nu"]            # Poisson's ratio
                            )
        except:
            Mat = pre.Material(         flag_lame = True,                               # If True should input lmbda and mu instead of E and nu
                                        coef1     = config["material"]["lmbda"],        # First Lame's coef
                                        coef2     = config["material"]["mu"]            # Second Lame's coef
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
                                    config["geometry"]["Name"],                         # Create the mesh object
                                    MaxElemSize, 
                                    config["interpolation"]["order"], 
                                    config["interpolation"]["dimension"]
                            )

    Mesh_object.AddBorders(config["Borders"]["Borders"])
    Mesh_object.AddBCs(                                                                 # Include Boundary physical domains infos (BCs+volume)
                                    config["geometry"]["Volume_element"],
                                    Excluded,
                                    config["DirichletDictionryList"]
                        )                   

    Mesh_object.MeshGeo()                                                               # Mesh the .geo file if .msh does not exist
    Mesh_object.ReadMesh()                                                              # Parse the .msh file

    match config["interpolation"]["dimension"]:
        case 1:
            if config["solver"]["IntegralMethod"] == "Gaussian_quad":
                Mesh_object.ExportMeshVtk1D()
        case 2 | 3:
            Mesh_object.ExportMeshVtk()

    # if config["interpolation"]["dimension"] ==2:
    #     Mesh_object.ExportMeshVtk()

    if config["interpolation"]["dimension"] ==1 and config["solver"]["IntegralMethod"] == "Trapezoidal":
        Mesh_object.AssemblyMatrix()                                                        # Build the assembly weight matrix

    if int(Mesh_object.dim) != int(Mesh_object.dimension):
        raise ValueError("The dimension of the provided geometry does not match the job dimension")

    #%% Application of the Space HiDeNN
    if not config["solver"]["ParametricStudy"]: 
        match config["interpolation"]["dimension"]:
            case 1:
                if config["solver"]["TrainingStrategy"]=="Integral":
                    match config["solver"]["IntegralMethod"]:                               # Build the model
                        case "Gaussian_quad":
                            Model_FEM = MeshNN_1D(Mesh_object, config["interpolation"]["n_integr_points"])  
                        case "Trapezoidal":
                            Model_FEM = MeshNN(Mesh_object)

                if config["solver"]["TrainingStrategy"]=="Mixed":
                    if config["solver"]["IntegralMethod"] == "Gaussian_quad":
                        Model_FEM = MeshNN_1D(Mesh_object, config["interpolation"]["n_integr_points"])
                        Model_test = MeshNN_1D(Mesh_object, config["interpolation"]["n_integr_points"])  
                        Model_test.Freeze_Mesh()

            case 2:
                Model_FEM = MeshNN_2D(Mesh_object, n_components = 2)
            
            case 3:
                Model_FEM = MeshNN_3D(Mesh_object, n_components = 3)

        # Set the coordinates as trainable
        Model_FEM.UnFreeze_Mesh()
        # Set the coordinates as untrainable
        Model_FEM.Freeze_Mesh()
        if not config["solver"]["FrozenMesh"]:
            Model_FEM.UnFreeze_Mesh()
        Model_FEM.UnFreeze_FEM()


    #%% Application of NeuROM
    # Parameter space-definition

    if config["solver"]["ParametricStudy"]:
        match config["solver"]["N_ExtraCoordinates"]:
            case 2:
                ParameterHypercube = torch.tensor([ [   config["parameters"]["para_1_min"],
                                                        config["parameters"]["para_1_max"],
                                                        config["parameters"]["N_para_1"]],
                                                    [   config["parameters"]["para_2_min"],
                                                        config["parameters"]["para_2_max"],
                                                        config["parameters"]["N_para_2"]]])
            case 1:
                ParameterHypercube = torch.tensor([[    config["parameters"]["para_1_min"],
                                                        config["parameters"]["para_1_max"],
                                                        config["parameters"]["N_para_1"]]])

        ROM_model = NeuROM(                                                                 # Build the surrogate (reduced-order) model
                                                        Mesh_object, 
                                                        ParameterHypercube, 
                                                        config,
                                                        config["solver"]["n_modes_ini"],
                                                        config["solver"]["n_modes_max"]
                        )

    #%% Load coarser model  

    if config["training"]["LoadPreviousModel"]:
        match config["solver"]["N_ExtraCoordinates"]:       
            case 2:
                match config["interpolation"]["dimension"]:
                    case 1:
                        PreviousFullModel = 'TrainedModels/1D_Bi_Stiffness_np_10_new'

                    case 2:
                        PreviousFullModel = 'TrainedModels/2D_Bi_Parameters_el_0.2'
            case 1:
                match config["solver"]["IntegralMethod"]:
                    case "Trapezoidal":
                        PreviousFullModel = 'TrainedModels/1D_Mono_Stiffness_np_100'
                    case "Gaussian_quad":
                        PreviousFullModel = 'TrainedModels/1D_Mono_Stiffness_Gauss_np_100'


    if config["training"]["LoadPreviousModel"] and config["solver"]["ParametricStudy"]:
        ROM_model.Init_from_previous(PreviousFullModel)
        ROM_model.UnfreezeTruncated()
    #%% Training 
    if config["solver"]["ParametricStudy"]:
        ROM_model.Freeze_Mesh()                                                             # Set space mesh cordinates as untrainable
        ROM_model.Freeze_MeshPara()                                                         # Set parameters mesh cordinates as untrainable

    if config["solver"]["ParametricStudy"]: 
        if not config["solver"]["FrozenMesh"]:
            ROM_model.UnFreeze_Mesh()                                                       # Set space mesh cordinates as trainable
        if not config["solver"]["FrozenParaMesh"]:
            ROM_model.UnFreeze_MeshPara()                                                   # Set parameters mesh cordinates as trainable

        ROM_model.TrainingParameters(   loss_decrease_c = config["training"]["loss_decrease_c"], 
                                        Max_epochs = config["training"]["n_epochs"], 
                                        learning_rate = config["training"]["learning_rate"])
                                        
        if config["training"]["TrainingRequired"]:
            ROM_model.train()
            ROM_model.to(tensor_float_type)
            ROM_model.to(device)
            optimizer = torch.optim.Adam([p for p in ROM_model.parameters() if p.requires_grad], lr=config["training"]["learning_rate"])
            match config["interpolation"]["dimension"]:
                case 1:
                    ROM_model, Mesh_object = Training_NeuROM_multi_level(ROM_model,config, Mat)         
                case 2:
                    ROM_model, Mesh_object = Training_NeuROM_multi_level(ROM_model,config, Mat)         
            ROM_model.to(torch.device("cpu"))
            ROM_model.eval()
    else:
        Model_FEM.to(tensor_float_type)
        Model_FEM.to(device)
        if not config["solver"]["FrozenMesh"]:
            Model_FEM.UnFreeze_Mesh()    

        if config["interpolation"]["dimension"]==2:
            Model_FEM.RefinementParameters( MaxGeneration = config["training"]["h_adapt_MaxGeneration"], 
                                        Jacobian_threshold = config["training"]["h_adapt_J_thrshld"])

            Model_FEM.TrainingParameters(   loss_decrease_c = config["training"]["loss_decrease_c"], 
                                            Max_epochs = config["training"]["n_epochs"], 
                                            learning_rate = config["training"]["learning_rate"])

        match config["interpolation"]["dimension"]:
            case 1:
                if config["training"]["TwoStageTraining"] == True:
                    if config["solver"]["TrainingStrategy"]=="Mixed":
                        if config["solver"]["IntegralMethod"] == "None":
                            Model_FEM = Training_1D_FEM_Gradient_Descent(Model_FEM, config, Mat)     
                            Model_FEM = Training_1D_FEM_LBFGS(Model_FEM, config, Mat)
                        else:
                            Model_FEM = Training_1D_FEM_Gradient_Descent(Model_FEM, config, Mat, Model_test)     
                            Model_FEM = Training_1D_FEM_LBFGS(Model_FEM, config, Mat, Model_test)
                    else:
                        Model_FEM = Training_1D_FEM_Gradient_Descent(Model_FEM, config, Mat)     
                        Model_FEM = Training_1D_FEM_LBFGS(Model_FEM, config, Mat)
                else: 
                    if config["solver"]["TrainingStrategy"]=="Mixed":
                        if config["solver"]["IntegralMethod"] == "None":
                            Model_FEM = Training_1D_FEM_LBFGS(Model_FEM, config, Mat)
                        else:
                            Model_FEM = Training_1D_FEM_LBFGS(Model_FEM, config, Mat, Model_test)
                    else:
                        Model_FEM = Training_1D_FEM_LBFGS(Model_FEM, config, Mat)

            case 2:
                time_start = time.time()
                Model_FEM = Training_2D_FEM(Model_FEM, config, Mat)
                time_end = time.time()


    #%% Post-processing
    with open(args.cf, mode="rb") as f:
        config = tomllib.load(f)
    print("*************** POST-PROCESSING ***************\n")
    if config["solver"]["ParametricStudy"]:
        Training_coordinates = torch.tensor([[i/50] for i in range(2,500)], 
                        dtype=torch.float32, 
                        requires_grad=True)

        if config["training"]["TrainingRequired"]:
            if min(ROM_model.training_recap["Loss_vect"]) > 0:                              # Find sign of the converged loss
                sign = "Positive"
            else:
                sign = "Negative"
        if config["solver"]["N_ExtraCoordinates"] == 2:                                     # define type of parametric study for saving files
            Study = "_BiPara"
        else: 
            Study = "_MonoPara" 
        
        if config["training"]["LoadPreviousModel"]:
            Initialisation_state = "_Initialised_"                                          # Initialised from previous model
        else:
            Initialisation_state = "_Raw_"                                                 # Trained from scratch
        if config["solver"]["FrozenMesh"]:
            Mesh_state = '_FrozenMesh_'
        else:
            Mesh_state = '_FreeMesh_'

        val = str(np.format_float_scientific(config["training"]["loss_decrease_c"], precision=2))            # Convergence criterion

        if config["postprocess"]["Plot_loss_mode"]:                                         # Plot loss and modes
            Pplot.Plot_PosNegLoss_Modes(ROM_model.training_recap["Mode_vect"],ROM_model.training_recap["Loss_vect"],
                                        'Loss_Modes'+"_"+
                                        config["geometry"]["Name"]+"_"+
                                        config["postprocess"]["Plot_name"]+
                                        'sub_levels_'+str(config["training"]["multiscl_max_refinment"])+'_'+
                                        str(config["interpolation"]["np"])+"_"+
                                        Study+
                                        Mesh_state+
                                        Initialisation_state+
                                        val
                                        , sign = sign,tikz = True, Zoom_required = True)
        if config["postprocess"]["Plot_loss_decay_mode"]:                                   # Plot loss rate and modes
            Pplot.Plot_Lossdecay_Modes(ROM_model.training_recap["Mode_vect"],
                                        ROM_model.training_recap["Loss_decrease_vect"],
                                        'Loss_rate_Modes'+"_"+
                                        config["geometry"]["Name"]+"_"+
                                        config["postprocess"]["Plot_name"]+
                                        'sub_levels_'+str(config["training"]["multiscl_max_refinment"])+'_'+
                                        str(config["interpolation"]["np"])+"_"+
                                        Study+
                                        Mesh_state
                                        +Initialisation_state+
                                        val,
                                        config["training"]["loss_decrease_c"],
                                        True)

        if config["postprocess"]["Plot_error_mode"]:                                        # Plot loss rate and modes
            Pplot.Plot_L2error_Modes(ROM_model.training_recap["Mode_vect"],
                                        ROM_model.training_recap["L2_error"],
                                        'L2_error_Modes'+"_"+
                                        config["geometry"]["Name"]+"_"+
                                        config["postprocess"]["Plot_name"]+
                                        'sub_levels_'+str(config["training"]["multiscl_max_refinment"])+'_'+
                                        str(config["interpolation"]["np"])+"_"+
                                        Study+
                                        Mesh_state
                                        +Initialisation_state+
                                        val,
                                        True)

        if config["postprocess"]["Plot_ROM_FOM"]:
            match config["interpolation"]["dimension"]:
                case 1:
                    match config["solver"]["N_ExtraCoordinates"]:
                        case 2:
                            Pplot.Plot_BiParametric_Young(ROM_model,
                                                        Training_coordinates,config["geometry"]["A"],
                                                        AnalyticBiParametricSolution,
                                                        name_model = 'Plot_ROM_FOM'+"_"+
                                                        config["postprocess"]["Plot_name"]+
                                                        str(config["interpolation"]["np"])+"_"+
                                                        Study+
                                                        Mesh_state+
                                                        val,
                                                        tikz=True)
                        case 1:
                            Pplot.Plot_Parametric_Young(ROM_model,
                                                        Training_coordinates,config["geometry"]["A"],
                                                        AnalyticSolution,
                                                        name_model = 'Plot_ROM_FOM'+"_"+
                                                        config["postprocess"]["Plot_name"]+
                                                        str(config["interpolation"]["np"])+"_"+
                                                        Mesh_state+
                                                        Study+
                                                        val,
                                                        tikz=True)                
        if config["postprocess"]["Interactive_pltot"]:

            match config["interpolation"]["dimension"]:
                case 1:
                    match config["solver"]["N_ExtraCoordinates"]:
                        case 2:
                            Pplot.Plot_BiParametric_Young_Interactive(  ROM_model,
                                                        Training_coordinates,
                                                        config["geometry"]["A"],
                                                        AnalyticBiParametricSolution,
                                                        'name_model')
                        case 1:
                            ROM_model.eval()
                            Pplot.Plot_Parametric_Young_Interactive(    ROM_model,
                                                        Training_coordinates,
                                                        config["geometry"]["A"],
                                                        AnalyticSolution,
                                                        'name_model')                                
                case 2:
                    Pplot.Plot_2D_PyVista(ROM_model, 
                                    Mesh_object, 
                                    config, 
                                    E = config["postprocess"]["Default_E"], 
                                    theta = config["postprocess"]["Default_theta"], 
                                    scalar_field_name = config["postprocess"]["scalar_field_name"], 
                                    scaling_factor = config["postprocess"]["scaling_factor"], 
                                    Interactive_parameter = config["postprocess"]["Interactive_parameter"],
                                    Plot_mesh = config["postprocess"]["Plot_mesh"],
                                    color_map = config["postprocess"]["colormap"])
            
    else:
        match config["interpolation"]["dimension"]:
            case 2:
                if config["postprocess"]["exportVTK"]:
                    Pplot.ExportFinalResult_VTK(Model_FEM,Mat,config["postprocess"]["Name_export"])
                    # Pplot.ExportSamplesforEval(Model_FEM,Mat,config)
                if config["postprocess"]["exportVTK_history"]:
                    Pplot.ExportHistoryResult_VTK(Model_FEM,Mat,config["postprocess"]["Name_export"])

            case 1:
                Pplot.Plot_Eval_1d(Model_FEM,config,Mat)


                

                    


       
# %%
if (__name__ == "__main__") or (__name__=='neurom.main'):
    Boolean_main = False

    if __name__=="__main__":
        __package__='neurom'
        Boolean_main = True
    import argparse
    import tomllib
    from .src import MyHeaders


    # Default_config_file = 'Configuration/config_2D_ROM.toml'
    Default_config_file = 'Configuration/config_2D.toml'
    # Default_config_file = 'Configuration/config_1D.toml'


    parser = argparse.ArgumentParser()
    parser.add_argument('-cf',type=str, help = 'path to the desired configuration file', default=Default_config_file, action = 'store')

    jupyter = MyHeaders.is_notebook()
    if jupyter:
        args = parser.parse_args('')
    else:
        args = parser.parse_args()
    inputs = vars(args)
    print(f"* Executing job in {args.cf}")
    if Boolean_main:
        main()