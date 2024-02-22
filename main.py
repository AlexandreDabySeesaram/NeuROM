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
np = 23                                             # Number of Nodes in the Mesh
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
TrainingRequired = True                           # Boolean leading to Loading pre trained model or retraining from scratch
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

    # # If GPU
    # if BoolGPU:
    #     BeamModel.to(mps_device)
    #     TrialCoordinates = torch.tensor([[i/50] for i in range(2,502)], 
    #                                 dtype=torch.float32, requires_grad=True).to(mps_device)


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



#%% Check model
if BoolPlotPost:
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.size"] = "13"


    PaperPara = torch.tensor([150])
    PaperPara = PaperPara[:,None] # Add axis so that dimensions match
    u_150 = BeamROM(TrialCoordinates,PaperPara)
    u_analytical_150 = AnalyticSolution(A,PaperPara.item(),TrialCoordinates.data)
    plt.plot(TrialCoordinates.data,u_analytical_150, color="#01426A", label = r'$E = 150~$MPa Analytical solution')
    plt.plot(TrialCoordinates.data,u_150.data,'--', color="#01426A", label = r'$E = 150~$MPa HiDeNN solution')

    PaperPara = torch.tensor([200])
    PaperPara = PaperPara[:,None] # Add axis so that dimensions match
    u_200 = BeamROM(TrialCoordinates,PaperPara)
    u_analytical_200 = AnalyticSolution(A,PaperPara.item(),TrialCoordinates.data)
    plt.plot(TrialCoordinates.data,u_analytical_200, color="#00677F", label = r'$E = 200~$MPa Analytical solution')
    plt.plot(TrialCoordinates.data,u_200.data,'--',color="#00677F", label = r'$E = 200~$MPa HiDeNN solution')

    PaperPara = torch.tensor([100])
    PaperPara = PaperPara[:,None] # Add axis so that dimensions match
    u_100 = BeamROM(TrialCoordinates,PaperPara)
    u_analytical_100 = AnalyticSolution(A,PaperPara.item(),TrialCoordinates.data)
    plt.plot(TrialCoordinates.data,u_analytical_100,color="#A92021", label = r'$E = 100~$MPa Analytical solution')
    plt.plot(TrialCoordinates.data,u_100.data,'--',color="#A92021", label = r'$E = 100~$MPa HiDeNN solution')
    plt.legend(loc="upper left")
    plt.xlabel('x (mm)')
    plt.ylabel('u (mm)')
    plt.savefig('Results/Para_displacements'+name_model+'.pdf', transparent=True)  
    plt.show()
    plt.clf()


#%% Interactive plot using matplotlib using jupyter ipywidgets
if BoolPlotPost:
    from ipywidgets import interact, widgets

    def interactive_plot(E):
        # Calculate the corresponding function values for each x value
        u_analytical_E = AnalyticSolution(A,E,TrialCoordinates.data)
        E = torch.tensor([E])
        E = E[:,None] # Add axis so that dimensions match
        u_E = BeamROM(TrialCoordinates,E)
        error_tensor = u_analytical_E - u_E
        # Reative error in percentage
        error_norm = 100*torch.sqrt(torch.sum(error_tensor*error_tensor))/torch.sqrt(torch.sum(u_analytical_E*u_analytical_E))
        error_scientific_notation = f"{error_norm:.2e}"
        # error_str = str(error_norm.item())
        title_error =  r'$\frac{\Vert u_{exact} - u_{ROM}\Vert}{\Vert u_{exact}\Vert}$ = '+error_scientific_notation+ '$\%$'
        # Plot the function
        plt.plot(TrialCoordinates.data,u_analytical_E,color="#A92021", label = 'Ground truth')
        plt.plot(TrialCoordinates.data, u_E.data, label = 'Discrete solution')
        plt.title(title_error)
        plt.xlabel('x (mm)')
        plt.ylabel('u(x,E) (mm)')
        plt.legend(loc="upper left")
        plt.grid(True)
        plt.ylim((0,0.02))
        plt.show()

    # Create an interactive slider
    slider = widgets.FloatSlider(value=0, min=100, max=200, step=0.01, description='E (GPa)')

    # Connect the slider to the interactive plot function
    interactive_plot_widget = interact(interactive_plot, E=slider)




#%% Plot Modes
if BoolPlotPost:
    Space_modes = [BeamROM.Space_modes[l](TrialCoordinates) for l in range(BeamROM.n_modes)]
    u_i = torch.cat(Space_modes,dim=1)  
    for mode in range(BeamROM.n_modes):
        Para_mode_List = [BeamROM.Para_modes[mode][l](TrialPara)[:,None] for l in range(BeamROM.n_para)]
        if mode == 0:
            lambda_i = torch.unsqueeze(torch.cat(Para_mode_List,dim=1), dim=0)
            # Para_modes = torch.unsqueeze(Para_modes, dim=0)
        else:
            New_mode = torch.unsqueeze(torch.cat(Para_mode_List,dim=1), dim=0)
            lambda_i = torch.vstack((lambda_i,New_mode))

    for mode in range(BeamROM.n_modes):
        plt.plot(TrialCoordinates.data,u_i[:,mode].data,label='Mode'+str(mode+1))
        plt.xlabel('x (mm)')
        plt.legend(loc="upper left")
    plt.savefig('Results/Space_modes'+str(BeamROM.n_modes)+'.pdf', transparent=True)  
    plt.clf()
        # plt.show()

    for mode in range(BeamROM.n_modes):
        plt.plot(TrialPara.data,lambda_i[mode,:,0].data,label='Mode'+str(mode+1))
        plt.xlabel('E (GPa)')
        plt.legend(loc="upper left")
    plt.savefig('Results/Para_modes'+str(BeamROM.n_modes)+'.pdf', transparent=True)  
    plt.clf()
        # plt.show()

