#%% Libraries import
# import HiDeNN library
from HiDeNN_PDE import MeshNN, NeuROM
# Import pre-processing functions
import Bin.Pre_processing as pre
# Import torch librairies
import torch
import torch.nn as nn

mps_device = torch.device("mps")
# Import mechanical functions
from Bin.PDE_Library import RHS, PotentialEnergyVectorised, \
        Derivative, AnalyticGradientSolution, AnalyticSolution
# Import Training funcitons
from Bin.Training import Test_GenerateShapeFunctions, Training_InitialStage, \
    Training_FinalStageLBFGS, FilterTrainingData, Training_NeuROM, Mixed_Training_InitialStage
#Import post processing libraries
import Post.Plots as Pplot
import time
import os

class Dataset(torch.utils.data.Dataset):

    def __init__(self, X):
        self.X = X
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):    
        x = self.X[index]

        return x

def InitializeMesh(order,dimension):
    if order ==1:
        MaxElemSize = L/(np-1)                      # Compute element size
    elif order ==2:
        n_elem = 0.5*(np-1)
        MaxElemSize = L/n_elem                     # Compute element size

    Beam_mesh = pre.Mesh('Rectange',MaxElemSize, order)    # Create the mesh object
    Volume_element = 100                        # Volume element correspond to the 1D elem in 1D
    Beam_mesh.AddBCs(Volume_element,
                    DirichletDictionryList)    # Include Boundary physical domains infos (BCs+volume)
    Beam_mesh.MeshGeo()                         # Mesh the .geo file if .msh does not exist
    Beam_mesh.ReadMesh()                        # Parse the .msh file
    Beam_mesh.AssemblyMatrix()                  # Build the assembly weight matrix

    return Beam_mesh 


#%% Pre-processing (could be put in config file later)
# Geometry of the Mesh
L = 10                                      # Length of the Beam
np = 5                                     # Number of Nodes in the Mesh
A = 1                                       # Section of the beam
E = 175                                     # Young's Modulus (should be 175)
alpha =0.0                                 # Weight for the Mesh regularisation 
name_model = 'ROM_1Para_np_'+str(np)
# User defines all boundary conditions 
DirichletDictionryList = [  {"Entity": 1, 
                             "Value": 0, 
                             "normal":1}, 
                            {"Entity": 2, 
                             "Value": 10, 
                             "normal":1}]

    
order_u = 1
order_du = 2

TrainingStrategy = 'Mixed'   # 'Integral', 'Mixed'

Beam_mesh_u = InitializeMesh(order_u)
Beam_mesh_du = InitializeMesh(order_du)


#%% Application of the NN
BeamModel_u = MeshNN(Beam_mesh_u,alpha)     # Create the associated model
# Boundary conditions
u_0 = 0                                 #Left BC
u_L = 0                                 #Right BC
BeamModel_u.SetBCs(u_0,u_L)

# Set the boundary values as trainable
# Set the coordinates as trainable
BeamModel_u.UnFreeze_Mesh()
# Set the coordinates as untrainable
# BeamModel.Freeze_Mesh()
# Set the require output requirements


BeamModel_du = MeshNN(Beam_mesh_du,alpha)     # Create the associated model
# Boundary conditions
u_0 = 0                                 #Left BC
u_L = 0                                 #Right BC
BeamModel_du.SetBCs(u_0,u_L)

BeamModel_du.UnFreeze_Mesh()
BeamModel_du.UnFreeze_BC()



BoolPlot = False                        # Boolean for plots used for gif
BoolPlotPost = False                    # Boolean for plots used for Post
BoolCompareNorms = True                 # Boolean for comparing energy norm to L2 norm
BoolGPU = False                         # Boolean enabling GPU computations (autograd function is not working currently on mac M2)
TrainingRequired = False                # Boolean leading to Loading pre trained model or retraining from scratch

n_plot_points = 500

#%% Define hyperparameters
learning_rate = 1.0e-3
n_epochs = 20000
MSE = nn.MSELoss()
BoolFilterTrainingData = True

if TrainingStrategy == 'Integral':

    # Training loop (Non parametric model, Integral loss)
    print("Training loop (Non parametric model, Integral loss)")
    optimizer = torch.optim.Adam(BeamModel_u.parameters(), lr=learning_rate)

    TrialCoordinates = torch.tensor([[i] for i in torch.linspace(0,L,n_plot_points)], dtype=torch.float32, requires_grad=True)
    print("TrialCoordinates = ", TrialCoordinates.shape)

    # # If GPU
    # if BoolGPU:
    #     BeamModel.to(mps_device)
    #     TrialCoordinates = torch.tensor([[i/50] for i in range(2,502)], 
    #                                 dtype=torch.float32, requires_grad=True).to(mps_device)

    # Training initial stage
    error, error2, InitialCoordinates, Coord_trajectories, BeamModel_u = Training_InitialStage(BeamModel_u, A, E, L, 
                                                                                            TrialCoordinates, optimizer, n_epochs, 
                                                                                            BoolCompareNorms, MSE, BoolFilterTrainingData)
    # Training final stage
    Training_FinalStageLBFGS(BeamModel_u, A, E, L, InitialCoordinates, 
                            TrialCoordinates, n_epochs, BoolCompareNorms, 
                            MSE, BoolFilterTrainingData,
                            error, error2, Coord_trajectories)

if TrainingStrategy == 'Mixed':


    PlotCoordTensor = torch.tensor([[i] for i in torch.linspace(0,L,500)], dtype=torch.float32, requires_grad=True)
    CoordinatesDataset = Dataset(PlotCoordTensor)
    CoordinatesBatchSet = torch.utils.data.DataLoader(CoordinatesDataset, batch_size=25, shuffle=True)
    print("Number of batches per epoch = ", len(CoordinatesBatchSet))

    # Training loop (Non parametric model, Mixed formulation)
    print("Training loop (Non parametric model, Mixed formulation)")
    optimizer = torch.optim.Adam(list(BeamModel_u.parameters()) + list(BeamModel_du.parameters()), lr=learning_rate)

    # # If GPU
    # if BoolGPU:
    #     BeamModel.to(mps_device)
    #     TrialCoordinates = torch.tensor([[i/50] for i in range(2,502)], 
    #                                 dtype=torch.float32, requires_grad=True).to(mps_device)

    # Training initial stage
    error, error2, InitialCoordinates, Coord_trajectories, BeamModel_u = Mixed_Training_InitialStage(BeamModel_u, BeamModel_du, A, E, L, 
                                                                                            CoordinatesBatchSet, PlotCoordTensor, optimizer, n_epochs, 
                                                                                            BoolCompareNorms, MSE, BoolFilterTrainingData)

    # Training final stage
    # Training_FinalStageLBFGS(BeamModel_u, A, E, L, InitialCoordinates, 
    #                         TrialCoordinates, n_epochs, BoolCompareNorms, 
    #                         MSE, FilterTrainingData,
    #                         error, error2, Coord_trajectories)


'''
#%% Parametric definition and initialisation of Reduced-order model
n_modes = 1
mu_min = 100
mu_max = 200
N_mu = 10

# # Para Young
# Eu_min = 100
# Eu_max = 200
# N_E = 10

# # Para Area
# A_min = 0.1
# A_max = 10
# N_A = 10


# ParameterHypercube = torch.tensor([[Eu_min,Eu_max,N_E],[A_min,A_max,N_A]])
BCs=[u_0,u_L]
BeamROM = NeuROM(Beam_mesh, BCs, n_modes, mu_min, mu_max,N_mu)


#%% Training
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
    # Train model
    BeamROM.UnFreeze_Para()
    optimizer = torch.optim.Adam(BeamROM.parameters(), lr=learning_rate)
    start_time = time.time()
    Loss_vect =  Training_NeuROM(BeamROM, A, L, TrialCoordinates,TrialPara, optimizer, n_epochs, BoolCompareNorms, MSE)
    stop_time = time.time()
    print(f'* Duration of training = {stop_time-start_time}s')
    BeamROM.Freeze_Space()

    # Save model
    # torch.save(BeamROM.state_dict(), 'TrainedModels/'+name_model)


#%% Check model
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
# plt.savefig('Results/Para_displacements_.pdf', transparent=True)  
plt.show()
plt.clf()

#%% Interactive plot using matplotlib 
# from matplotlib.widgets import Slider

# def update(val):
#     # Callback function to update the plot when the slider is changed
#     E_value = slider.val
#     plt.clf()  # Clear the previous plot
#     interactive_plot(E_value)

# def interactive_plot(E):
#     # Calculate the corresponding function values for each x value
#     u_analytical_E = AnalyticSolution(A, E, TrialCoordinates.data)
#     E = torch.tensor([E])
#     E = E[:, None]  # Add axis so that dimensions match
#     u_E = BeamROM(TrialCoordinates, E)

#     # Plot the function
#     plt.plot(TrialCoordinates.data, u_analytical_E, color="#A92021", label='Ground truth')
#     plt.plot(TrialCoordinates.data, u_E.data, label='Discrete solution')
#     plt.title('Displacement')
#     plt.xlabel('x (mm)')
#     plt.ylabel('u(x,E) (mm)')
#     plt.legend(loc="upper left")
#     plt.grid(True)
#     plt.ylim((0, 0.02))
#     plt.show()

# # Create a figure and axis
# fig, ax = plt.subplots()
# plt.subplots_adjust(bottom=0.25)  # Adjust the bottom margin for the slider

# # Create a slider
# slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
# slider = Slider(slider_ax, 'E (MPa)', 100, 200, valinit=150, valstep=0.01)

# # Connect the slider to the update function
# slider.on_changed(update)

# # Initial plot
# interactive_plot(slider.val)

# plt.show()

#%% Interactive plot using matplotlib using jupyter ipywidgets
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
'''


