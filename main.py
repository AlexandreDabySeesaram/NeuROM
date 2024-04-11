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

class Dataset(torch.utils.data.Dataset):

    def __init__(self, X):
        self.X = X
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):    
        x = self.X[index]

        return x


#%% Pre-processing (could be put in config file later)
# Defintition of the structure and meterial
L = 10                                              # Length of the Beam
np = 19                                             # Number of Nodes in the Mesh
A = 1                                               # Section of the beam
E = 175                                             # Young's Modulus (should be 175)
# User defines all boundary conditions 
DirichletDictionryList = [  {"Entity": 1, 
                             "Value": 0, 
                             "normal":1}, 
                            {"Entity": 2, 
                             "Value": 0.01, 
                             "normal":1}]

# Definition of the space discretisation
alpha = 0.0                                       # Weight for the Mesh regularisation 
order = 2                                          # Order of the shape functions
dimension = 1

if order ==1:
    MaxElemSize = L/(np-1)                         # Compute element size
elif order ==2:
    n_elem = 0.5*(np-1)
    MaxElemSize = L/n_elem                         # Compute element size

if dimension ==1:
    Beam_mesh = pre.Mesh('Beam',MaxElemSize, order, dimension)    # Create the mesh object
if dimension ==2:
    Beam_mesh = pre.Mesh('Rectangle',MaxElemSize, order, dimension)    # Create the mesh object

Volume_element = 100                               # Volume element correspond to the 1D elem in 1D
Beam_mesh.AddBCs(Volume_element,
                 DirichletDictionryList)           # Include Boundary physical domains infos (BCs+volume)
Beam_mesh.MeshGeo()                                # Mesh the .geo file if .msh does not exist
Beam_mesh.ReadMesh()                               # Parse the .msh file
Beam_mesh.AssemblyMatrix()                         # Build the assembly weight matrix

#%% Options & Hyperparameters
BoolPlot = False                                   # Boolean for plots used for gif
BoolPlotPost = False                               # Boolean for plots used for Post
BoolCompareNorms = True                            # Boolean for comparing energy norm to L2 norm
BoolGPU = False                                    # Boolean enabling GPU computations (autograd function is not working currently on mac M2)
TrainingRequired = True                           # Boolean leading to Loading pre trained model or retraining from scratch
SaveModel = False                                  # Boolean leading to Loading pre trained model or retraining from scratch
ParametricStudy = False                             # Boolean to switch between space model and parametric sturdy
TrainingStrategy = 'Mixed'                         # 'Integral' or 'Mixed'
LoadPreviousModel = False                           # Boolean to enable reusing a previously trained model
n_epochs = 3000                                    # Maximum number of iterations for the training stage
learning_rate = 0.001                              # optimizer learning rate
BoolFilterTrainingData = True                         # Slightly move training samples if they are on the mesh nodes exactly
BoolCompile = False                                 # Enable compilation of the model
BiPara = True                                       # Enable 2 Young modulus
Visualisatoin_only = False
#%% Application of the Space HiDeNN
BeamModel = MeshNN(Beam_mesh)                # Create the associated model
# Boundary conditions
u_0 = DirichletDictionryList[0]['Value']           #Left BC
u_L = DirichletDictionryList[1]['Value']           #Right BC
BeamModel.SetBCs(u_0,u_L)

# Set the boundary values as trainable
# Set the coordinates as trainable
BeamModel.Freeze_Mesh()
# Set the coordinates as untrainable
# BeamModel.Freeze_Mesh()
# Set the require output requirements

#%% Application of NeuROM
n_modes = 100
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

if BiPara:
    ParameterHypercube = torch.tensor([[Eu_min,Eu_max,N_E],[Eu_min,Eu_max,N_A]])
else:
    ParameterHypercube = torch.tensor([[Eu_min,Eu_max,N_E]])

# Boundary conditions
u_0 = DirichletDictionryList[0]['Value']           #Left BC
u_L = DirichletDictionryList[1]['Value']           #Right BCF
BCs = [u_0,u_L]
BeamROM = NeuROM(Beam_mesh, BCs, n_modes, ParameterHypercube)
name_model = 'ROM_1Para_np_'+str(np)+'_order_'+str(order)+'_nmodes_'\
            +str(n_modes)+'_npara_'+str(ParameterHypercube.shape[0])



#%% Load coarser model  
PreviousFullModel = 'TrainedModels/FullModel_np_100'
if LoadPreviousModel and os.path.isfile(PreviousFullModel):
    # torch.save(BeamROM, 'TrainedModels/FullModel') # To save a full coarse model
    BeamROM_coarse = torch.load(PreviousFullModel) # To load a full coarse model
    # BeamROM_coarse_dict = torch.load('TrainedModels/ROM_1Para_np_50_order_1_nmodes_1_npara_1')
    # BeamROM_coarse_dic['Space_modes.0.NodalValues_uu']
    newcoordinates = [coord for coord in BeamROM.Space_modes[0].coordinates]
    newcoordinates = torch.cat(newcoordinates,dim=0)
    Nb_modes_fine = len(BeamROM.Space_modes)
    Nb_modes_coarse = len(BeamROM_coarse.Space_modes)
    IndexesNon0BCs = [i for i, BC in enumerate(BCs) if BC != 0]
    if IndexesNon0BCs and BeamROM_coarse.Space_modes[0].u_0 == 0 and BeamROM_coarse.Space_modes[0].u_L == 0:
        NewNodalValues = BeamROM_coarse.Space_modes[0](newcoordinates) 
        BeamROM.Space_modes[1].InterpoLayer_uu.weight.data = NewNodalValues[2:,0]
        BeamROM.Para_modes[1][0].InterpoLayer.weight.data = BeamROM_coarse.Para_modes[0][0].InterpoLayer.weight.data.clone().detach()
        BeamROM.Space_modes[1].Freeze_FEM()
        BeamROM.Para_modes[1][0].Freeze_FEM()
    else :
        for mode in range(Nb_modes_coarse):
            NewNodalValues = BeamROM_coarse.Space_modes[mode](newcoordinates)
            BeamROM.Space_modes[mode].InterpoLayer_uu.weight.data = NewNodalValues[2:,0]
            BeamROM.Para_modes[mode][0].InterpoLayer.weight.data = BeamROM_coarse.Para_modes[mode][0].InterpoLayer.weight.data.clone().detach()
        if Nb_modes_coarse<Nb_modes_fine:
            for mode in range(Nb_modes_coarse,Nb_modes_fine):
                # print(mode)
                BeamROM.Space_modes[mode].InterpoLayer_uu.weight.data = 0*NewNodalValues[2:,0]
                BeamROM.Para_modes[mode][0].InterpoLayer.weight.data.fill_(0)
elif not os.path.isfile(PreviousFullModel):
    print('******** WARNING LEARNING FROM SCRATCH ********\n')

#%% Training 
if not ParametricStudy:

    if TrainingStrategy == 'Integral':
        # Training loop (Non parametric model, Integral loss)
        print("Training loop (Non parametric model, Integral loss)")
        optimizer = torch.optim.SGD(BeamModel.parameters(), lr=learning_rate)
        TrialCoordinates = torch.tensor([[i] for i in torch.linspace(0,L,500)], dtype=torch.float32, requires_grad=True)
        
        BeamModel.UnFreeze_Mesh()

        # If GPU
        if BoolGPU:
            BeamModel.to(mps_device)
            TrialCoordinates = torch.tensor([[i/50] for i in range(2,502)], 
                                        dtype=torch.float32, requires_grad=True).to(mps_device)

        # Training initial stage
        error, error2, InitialCoordinates, Coord_trajectories, BeamModel = Training_InitialStage(BeamModel, A, E, L, 
                                                                                                TrialCoordinates, optimizer, n_epochs, 
                                                                                                BoolCompareNorms, nn.MSELoss(), BoolFilterTrainingData)

        # Training final stage
        Training_FinalStageLBFGS(BeamModel, A, E, L, InitialCoordinates, 
                                TrialCoordinates, n_epochs, BoolCompareNorms, 
                                nn.MSELoss(), FilterTrainingData,
                                error, error2, Coord_trajectories)

    if TrainingStrategy == 'Mixed':

        BeamModel_u = BeamModel

        #np = 15
        order_du = 1
        if order_du ==1:
            MaxElemSize_du = L/(np-1)                         # Compute element size
        elif order ==2:
            n_elem = 0.5*(np-1)
            MaxElemSize_du = L/n_elem  

        if dimension ==1:
            Beam_mesh_du = pre.Mesh('Beam',MaxElemSize_du, order_du, dimension)    # Create the mesh object
        if dimension ==2:
            Beam_mesh_du = pre.Mesh('Rectangle',MaxElemSize_du, order_du, dimension)    # Create the mesh object

        Volume_element = 100                               # Volume element correspond to the 1D elem in 1D
        Beam_mesh_du.AddBCs(Volume_element,
                        DirichletDictionryList)           # Include Boundary physical domains infos (BCs+volume)
        Beam_mesh_du.MeshGeo()                                # Mesh the .geo file if .msh does not exist
        Beam_mesh_du.ReadMesh()                               # Parse the .msh file
        Beam_mesh_du.AssemblyMatrix()                         # Build the assembly weight matrix
        BeamModel_du = MeshNN(Beam_mesh_du)     # Create the associated model
        #BeamModel_du.SetBCs(u_0,u_L)
        BeamModel_du.Freeze_Mesh()
        BeamModel_du.UnFreeze_BC()

        # Training loop (Non parametric model, Mixed formulation)
        print("Training loop (Non parametric model, Mixed formulation)")
        
        optimizer = torch.optim.Adam(list(BeamModel_u.parameters())+list(BeamModel_du.parameters()), lr=0.001)


        PlotCoordTensor = torch.tensor([[i] for i in torch.linspace(0,L,5000)], dtype=torch.float32, requires_grad=True)

        PlotCoordTensor = FilterTrainingData(BeamModel_u, PlotCoordTensor)
        PlotCoordTensor = FilterTrainingData(BeamModel_du, PlotCoordTensor)

        # # If GPU
        if BoolGPU:
            BeamModel.to(mps_device)
            TrialCoordinates = torch.tensor([[i/50] for i in range(2,502)], 
                                        dtype=torch.float32, requires_grad=True).to(mps_device)

        # Training initial stage

        BeamModel_u.UnFreeze_Mesh()
        BeamModel_du.UnFreeze_Mesh()

        CoordinatesDataset = Dataset(PlotCoordTensor)
        CoordinatesBatchSet = torch.utils.data.DataLoader(CoordinatesDataset, batch_size=200, shuffle=True)
        print("Number of batches per epoch = ", len(CoordinatesBatchSet))


        error_pde, error_constit, error2, InitialCoordinates_u, InitialCoordinates_du, Coord_trajectories = Mixed_Training_InitialStage(BeamModel_u, BeamModel_du, A, E, L, 
                                                                                        CoordinatesBatchSet, PlotCoordTensor, 
                                                                                        optimizer, n_epochs, 
                                                                                        BoolCompareNorms, nn.MSELoss(), BoolFilterTrainingData, 
                                                                                        L,1)


        '''
        TrialCoordinates = torch.tensor([[i] for i in torch.linspace(0,L,500)], dtype=torch.float32, requires_grad=True)
        # Second stage on 500 points only.
        Training_FinalStageLBFGS_Mixed(BeamModel_u, BeamModel_du, A, E, L, InitialCoordinates_u, InitialCoordinates_du,
                                TrialCoordinates, n_epochs, BoolCompareNorms, 
                                nn.MSELoss(), BoolFilterTrainingData,
                                error_pde, error_constit, error2, Coord_trajectories,
                                L,1) 
        '''

else:
    BeamROM.Freeze_Mesh()
    BeamROM.Freeze_MeshPara()
    TrialCoordinates = torch.tensor([[i/50] for i in range(2,500)], 
                                    dtype=torch.float32, requires_grad=True)
    TrialPara = torch.linspace(mu_min,mu_max,50, 
                                    dtype=torch.float32, requires_grad=True)
    TrialPara = TrialPara[:,None] # Add axis so that dimensions match

    TrialPara2 = torch.linspace(mu_min,mu_max,50, 
                                    dtype=torch.float32, requires_grad=True)
    TrialPara2 = TrialPara2[:,None] # Add axis so that dimensions match

    if BiPara:
        Para_coord_list = nn.ParameterList((TrialPara,TrialPara2))
    else:
        Para_coord_list = [TrialPara]

    if not TrainingRequired:
        if IndexesNon0BCs:
            name_model+='_BCs'
        # Load pre trained model
        if os.path.isfile('TrainedModels/'+name_model):
            BeamROM.load_state_dict(torch.load('TrainedModels/'+name_model))
            print('************ LOADING MODEL COMPLETE ***********\n')
        else: 
            TrainingRequired = True
            print('**** WARNING NO PRE TRAINED MODEL WAS FOUND ***\n')


    if TrainingRequired:
        # prof = torch.profiler.profile(
        # on_trace_ready=torch.profiler.tensorboard_trace_handler('.logs/NeuROM'),
        # record_shapes=True)
        
        # prof.start()
        BeamROM.train()
        if BoolGPU:
            BeamROM.to(mps_device)
            TrialCoordinates = TrialCoordinates.to(mps_device)
            TrialPara = TrialPara.to(mps_device)
            BeamROM(TrialCoordinates,TrialPara)

        if BoolCompile:
            BeamROM_compiled = torch.compile(BeamROM, backend="inductor", mode = 'max-autotune-no-cudagraphs',dynamic=True)
            optimizer = torch.optim.Adam(BeamROM_compiled.parameters(), lr=learning_rate)
            u_copm = BeamROM_compiled(TrialCoordinates,Para_coord_list)
            Loss_vect, L2_error, training_time, Mode_vect, Loss_decrease_vect =  Training_NeuROM(BeamROM_compiled, A, L, TrialCoordinates,TrialPara, optimizer, n_epochs,BiPara)
            Loss_vect, L2_error =  Training_NeuROM_FinalStageLBFGS(BeamROM_compiled, A, L, TrialCoordinates,TrialPara, optimizer, n_epochs, 10,Loss_vect,L2_error,training_time,BiPara)

        else:
            optimizer = torch.optim.Adam([p for p in BeamROM.parameters() if p.requires_grad], lr=learning_rate)
            Loss_vect, L2_error, training_time, Mode_vect,Loss_decrease_vect =  Training_NeuROM(BeamROM, A, L, TrialCoordinates,Para_coord_list, optimizer, n_epochs,BiPara)
            Loss_vect, L2_error =  Training_NeuROM_FinalStageLBFGS(BeamROM, A, L, TrialCoordinates,Para_coord_list, optimizer, n_epochs, 10,Loss_vect,L2_error,training_time,BiPara)
        BeamROM.eval()
    
        # Save model
        if SaveModel:
            torch.save(BeamROM.state_dict(), 'TrainedModels/'+name_model)

# prof.stop()


    #%% Post-processing

    if BoolPlotPost:
        Pplot.Plot_Parametric_Young(BeamROM,TrialCoordinates,A,AnalyticSolution,name_model)
        Pplot.Plot_Parametric_Young_Interactive(BeamROM,TrialCoordinates,A,AnalyticSolution,name_model)
        Pplot.PlotModes(BeamROM,TrialCoordinates,TrialPara,A,AnalyticSolution,name_model)
        Pplot.Plot_Compare_Loss2l2norm(Loss_vect,L2_error,'2StageTraining')

    if False:
        Space_modes = [BeamROM.Space_modes[l](TrialCoordinates) for l in range(BeamROM.n_modes)]
        u_i = torch.cat(Space_modes,dim=1) 
if Visualisatoin_only:
    Space_modes = [BeamROM.Space_modes[l](TrialCoordinates) for l in range(BeamROM.n_modes)]
    u_i = torch.cat(Space_modes,dim=1) 

    # Pplot.Plot_Parametric_Young_Interactive(BeamROM,TrialCoordinates,A,AnalyticSolution,name_model)
    # Pplot.AppInteractive(BeamROM,TrialCoordinates,A,AnalyticSolution)

u_eval = BeamROM(TrialCoordinates,Para_coord_list)
import matplotlib.pyplot as plt
# Pplot.PlotModesBi(BeamROM,TrialCoordinates,Para_coord_list,A,AnalyticSolution,name_model)
# BeamROM = torch.load('TrainedModels/FullModel_BiParametric')


if Visualisatoin_only:
    TrialCoordinates = torch.tensor([[i/10] for i in range(2,100)], 
                                    dtype=torch.float32, requires_grad=True)
    TrialPara = torch.linspace(mu_min,mu_max,50, 
                                    dtype=torch.float32, requires_grad=True)
    TrialPara = TrialPara[:,None] # Add axis so that dimensions match

    TrialPara2 = torch.linspace(mu_min,mu_max,50, 
                                    dtype=torch.float32, requires_grad=True)
    TrialPara2 = TrialPara2[:,None] # Add axis so that dimensions match

    if BiPara:
        Para_coord_list = nn.ParameterList((TrialPara,TrialPara2))
    else:
        Para_coord_list = [TrialPara]
    BeamROM = torch.load('TrainedModels/FullModel_BiParametric')
    BeamROM = torch.load('TrainedModels/FullModel_BiParametric_np100')

    Pplot.Plot_BiParametric_Young_Interactive(BeamROM,TrialCoordinates,A,AnalyticBiParametricSolution,name_model)
    Pplot.Plot_Loss_Modes(Mode_vect,Loss_vect,'Loss_Modes')

