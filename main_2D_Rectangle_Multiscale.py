#%% Libraries import
# import HiDeNN library
from HiDeNN_PDE import MeshNN_2D, NeuROM
# Import pre-processing functions
import Bin.Pre_processing as pre
# Import torch librairies
import torch
import torch.nn as nn
# Import mechanical functions
from Bin.PDE_Library import RHS, PotentialEnergyVectorised, \
     Derivative, AnalyticGradientSolution, AnalyticSolution, AnalyticBiParametricSolution, GetRealCoord
# Import Training funcitons

#Import post processing libraries
import Post.Plots as Pplot
import time
import os
import torch._dynamo as dynamo
import numpy as numpy
import meshio

mps_device = torch.device("mps")


from Bin.Training import LBFGS_Stage2_2D, GradDescend_Stage1_2D, Generate_Training_IDs
from Bin.PDE_Library import Mixed_2D_loss
from Post.Evaluation_wrt_NumSolution import NumSol_eval

class Dataset(torch.utils.data.Dataset):

    def __init__(self, X):
        self.X = X
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):    
        x = self.X[index]

        return x

def GetMaxElemSize(L, np, order):
    if order ==1:
        MaxElemSize = L/(np-1)                         # Compute element size
    elif order ==2:
        n_elem = 0.5*(np-1)
        MaxElemSize = L/n_elem                         # Compute element size

    return MaxElemSize




L = 10                                              # Length of the Beam
np = 2                                             # Number of Nodes in the Mesh

loss = [[],[]]

lmbda = 1.25
mu = 1.0
E = mu*(3*lmbda+2*mu)/(lmbda+mu)                                             # Young's Modulus (should be 175)


# Definition of the space discretisation
order_u = 2                                          # Order of the shape functions
order_du = 1
dimension = 2

#MaxElemSize = GetMaxElemSize(L, np, order_du)

def DefineModels(dimension, order_u, order_du, MaxElemSize, MinElemSize):

    if dimension ==1:
        Domain_mesh_u = pre.Mesh('Beam',MaxElemSize, MinElemSize, order_u, dimension)    # Create the mesh object
        Domain_mesh_du = pre.Mesh('Beam',MaxElemSize, MinElemSize, order_du, dimension)    # Create the mesh object
    if dimension ==2:
        # Domain_mesh_u = pre.Mesh('Rectangle',MaxElemSize, order_u, dimension)    # Create the mesh object
        # Domain_mesh_du = pre.Mesh('Rectangle',MaxElemSize, order_du, dimension)    # Create the mesh object
        Domain_mesh_u = pre.Mesh('Rectangle', MaxElemSize, MinElemSize, order_u, dimension)    # Create the mesh object
        Domain_mesh_du = pre.Mesh('Rectangle', MaxElemSize, MinElemSize, order_du, dimension)    # Create the mesh object

    Volume_element = 100                               # Volume element correspond to the 1D elem in 1D

    ####################################################################
    # User defines all boundary conditions

    '''
    DirichletDictionryList = [  {"Entity": 111, "Value": 0, "Normal": 0, "Relation": False, "Constitutive": False},
                                {"Entity": 111, "Value": 0, "Normal": 1, "Relation": False, "Constitutive": False},
                                #{"Entity": 113, "Value": 0, "Normal": 0, "Relation": False, "Constitutive": False},
                                #{"Entity": 113, "Value": 1, "Normal": 1, "Relation": False, "Constitutive": False}
                                ]

    DirichletDictionryList = [  {"Entity": 111, "Value": 0, "Normal": 0, "Relation": False, "Constitutive": False},
                                {"Entity": 111, "Value": 0, "Normal": 1, "Relation": False, "Constitutive": False}
                                ]
    '''
    DirichletDictionryList = [  {"Entity": 111, "Value": 0, "Normal": 0, "Relation": False, "Constitutive": False},
                                {"Entity": 111, "Value": 0, "Normal": 1, "Relation": False, "Constitutive": False},
                                {"Entity": 113, "Value": 1, "Normal": 0, "Relation": False, "Constitutive": False},
                                {"Entity": 113, "Value": 0, "Normal": 1, "Relation": False, "Constitutive": False}
                                ]


    Excluded_elements_u = []
    Borders = [111,112,113,114,115]
    Domain_mesh_u.AddBorders(Borders)
    Domain_mesh_u.AddBCs(Volume_element, Excluded_elements_u, DirichletDictionryList)           # Include Boundary physical domains infos (BCs+volume)
    Domain_mesh_u.MeshGeo()                                # Mesh the .geo file if .msh does not exist
    Domain_mesh_u.ReadMesh()                               # Parse the .msh file
    Domain_mesh_u.ExportMeshVtk()

    ####################################################################
    # Normal: 0     component: x, du_dx
    # Normal: 1     component: y, dv_dy
    # Normal: 2     component: x, du_dy = dv_dx

  
    DirichletDictionryList = [  {"Entity": 112, "Value": 0.0, "Normal": 0, "Relation": False, "Constitutive": False},
                                {"Entity": 112, "Value": 0, "Normal":  2, "Relation": False, "Constitutive": False},
                                {"Entity": 114, "Value": 0.0, "Normal": 0, "Relation": False, "Constitutive": False},
                                {"Entity": 114, "Value": 0.0, "Normal": 2, "Relation": False, "Constitutive": False}
                                ]

    '''
    DirichletDictionryList = [  {"Entity": 112, "Value": 0, "Normal": 0, "Relation": False, "Constitutive": False},
                                {"Entity": 112, "Value": 0, "Normal":  2, "Relation": False, "Constitutive": False},
                                {"Entity": 113, "Value": 0.0, "Normal": 2, "Relation": False, "Constitutive": False},
                                {"Entity": 113, "Value": 0.05, "Normal": 1, "Relation": False, "Constitutive": False},
                                #{"Entity": 111, "Value": 0.0, "Normal": 0, "Relation": False, "Constitutive": True},
                                #{"Entity": 114, "Value": 0.0, "Normal": 0, "Relation": False, "Constitutive": True}
                                ]
    '''
    '''
    DirichletDictionryList = [  {"Entity": 112, "Value": 0, "Normal": 0, "Relation": False, "Constitutive": False},
                            {"Entity": 112, "Value": 0, "Normal":  2, "Relation": False, "Constitutive": False},
                            {"Entity": 113, "Value": 0.0, "Normal": 2, "Relation": False, "Constitutive": False},
                            {"Entity": 113, "Value": 0.05, "Normal": 1, "Relation": False, "Constitutive": False},
                            #{"Entity": 111, "Value": 0.0, "Normal": 0, "Relation": False, "Constitutive": True},
                            #{"Entity": 114, "Value": 0.0, "Normal": 0, "Relation": False, "Constitutive": True}
                            ]
    '''

    Excluded_elements_du = [200]
    Borders = [111,112,113,114,115]
    Domain_mesh_du.AddBorders(Borders)

    Domain_mesh_du.AddBCs(Volume_element, Excluded_elements_du, DirichletDictionryList)           # Include Boundary physical domains infos (BCs+volume)
    Domain_mesh_du.MeshGeo()                                # Mesh the .geo file if .msh does not exist
    Domain_mesh_du.ReadMesh()                               # Parse the .msh file
    Domain_mesh_du.ExportMeshVtk()

    ####################################################################

    print("Model u")
    Model_u = MeshNN_2D(Domain_mesh_u, 2, 0)                # Create the associated model
    Model_u.UnFreeze_Values()
    Model_u.Freeze_Mesh()

    print()
    print("Model du")
    Model_du = MeshNN_2D(Domain_mesh_du, 3, 0)                # Create the associated model
    Model_du.UnFreeze_Values()
    Model_du.Freeze_Mesh()

    #print("Model du param")
    #for param in Model_du.parameters():
    #    if param.requires_grad == True:
    #        print(param)

    ####################################################################

    return Model_u, Model_du, Domain_mesh_u, Domain_mesh_du


def DoTheRest( Model_u, Model_du, Domain_mesh_u, Domain_mesh_du,  E, mu, lmbda, L, w0, w1):
    def GenerateData(n_train):
        TrailCoord_1d_x = torch.tensor([i for i in torch.linspace(0,L,n_train)],dtype=torch.float64, requires_grad=True)
        TrailCoord_1d_y = torch.tensor([i for i in torch.linspace(0,5*L,5*n_train)],dtype=torch.float64,  requires_grad=True)

        PlotCoordinates = torch.cartesian_prod(TrailCoord_1d_x,TrailCoord_1d_y)
        IDs_u = torch.tensor(Domain_mesh_du.GetCellIds(PlotCoordinates),dtype=torch.int)
        IDs_du = torch.tensor(Domain_mesh_du.GetCellIds(PlotCoordinates),dtype=torch.int)

        inside_domain = torch.where(IDs_u > -1)
        PlotCoordinates = PlotCoordinates[inside_domain]
        PlotCoordinates = PlotCoordinates.clone().detach().requires_grad_(True)

        IDs_u = IDs_u[inside_domain]
        IDs_du = IDs_du[inside_domain]

        return PlotCoordinates, IDs_u, IDs_du

    n_train = 50
    PlotCoordinates, IDs_u, IDs_du = GenerateData(n_train)

    points_per_elem = 6 #max(int(1000/Domain_mesh_du.NElem), 6)

    cell_ids, ref_coords = Generate_Training_IDs(points_per_elem, Domain_mesh_du)


    constit_cell_IDs_u = []
    constit_point_coord = []

    # for j in range(len(Model_du.constit_BC_node_IDs)):
    #     constit_point_IDs = Model_du.constit_BC_node_IDs[j]
    #     #print("1: constit_point_IDs : ", constit_point_IDs)
    #     #print()
    #     point_coord = [(Model_du.coordinates[i]).clone().detach().requires_grad_(True) for i in constit_point_IDs]
    #     print("point_coord = ", point_coord)
    #     cell_IDs_u = torch.tensor([torch.tensor(Domain_mesh_u.GetCellIds(i),dtype=torch.int)[0] for i in point_coord])
    #     #print("cell_IDs_u = ", cell_IDs_u)
    #     #print()
    #     constit_point_coord.append((torch.cat(point_coord)).clone().detach().requires_grad_(True))
    #     constit_cell_IDs_u.append(cell_IDs_u)

    ####################################################

    u_predicted = Model_u(PlotCoordinates, IDs_u) 
    du_predicted = Model_du(PlotCoordinates, IDs_du) 
    _, _, s11, s22, s12 =  Mixed_2D_loss(u_predicted[0,:], u_predicted[1,:],
                                                        du_predicted[0,:], du_predicted[1,:], du_predicted[2,:],
                                                        PlotCoordinates, lmbda = 1.25, mu = 1.0)

    Pplot.Plot2Dresults(u_predicted, PlotCoordinates , "_u_Initial")
    Pplot.Plot2Dresults_Derivative(du_predicted, s11, s22, s12, PlotCoordinates, "_Stress_Initial")
    Pplot.Export_Stress_to_vtk(Domain_mesh_du, Model_du, 0)
    Pplot.Export_Displacement_to_vtk(Domain_mesh_u.name_mesh, Model_u, 0)


    optimizer = torch.optim.Adam(list(Model_u.parameters())+list(Model_du.parameters()))

    print("**************** START TRAINING 1st stage ***************\n")
    loss = [[],[]]


    current_BS = int(cell_ids.shape[0]/2) #6*Domain_mesh_du.NElem
    n_epochs = 4000
    CoordinatesBatchSet = torch.utils.data.DataLoader([[cell_ids[i], ref_coords[i]] for i in range((cell_ids.shape)[0])], 
                                                            batch_size=current_BS, shuffle=True)

    print("Elements = ", Domain_mesh_du.NElem)
    print("Points per element = ", points_per_elem)
    print("Number of training points = ", cell_ids.shape[0])
    print("Batch size = ", CoordinatesBatchSet.batch_size)
    print("Number of batches per epoch = ", len(CoordinatesBatchSet))
    print()

    Model_u, Model_du, loss = GradDescend_Stage1_2D(Model_u, Model_du, Domain_mesh_u, Domain_mesh_du, IDs_u, IDs_du, PlotCoordinates,
                                CoordinatesBatchSet, w0, w1, n_epochs, optimizer, len(CoordinatesBatchSet),
                                n_train, loss, constit_point_coord, constit_cell_IDs_u, lmbda, mu)

    
    # print("*************** 2nd stage LBFGS ***************\n")
    # n_epochs = 100
    # Model_u, Model_du = LBFGS_Stage2_2D(Model_u, Model_du, Domain_mesh_u, Domain_mesh_du, IDs_u, IDs_du, PlotCoordinates, 
    #                             cell_ids, ref_coords, 
    #                             w0, w1, n_train, n_epochs, constit_point_coord, constit_cell_IDs_u, lmbda, mu)

    return Model_u, Model_du


def InitializeWeights(NewModel_u, NewModel_du, Model_u, Model_du, Domain_mesh_u, Domain_mesh_du):

    print(" * Initialization based on previous resolution")
    # # # # # Model du # # # # # 
    original_coord_du = [(Model_du.coordinates[i]).clone().detach().requires_grad_(True) for i in range(len(Model_du.coordinates))]
    original_coord_du = (torch.cat(original_coord_du)).clone().detach().requires_grad_(True)

    val_s0 = [(NewModel_du.nodal_values[0][i]) for i in range(len(NewModel_du.coordinates))]
    val_s1 = [(NewModel_du.nodal_values[1][i]) for i in range(len(NewModel_du.coordinates))]
    val_s2 = [(NewModel_du.nodal_values[2][i]) for i in range(len(NewModel_du.coordinates))]

    node_IDs_s11 = []
    node_IDs_s22 = []
    node_IDs_s12 = []

    for j in range(len(val_s0)):
        if val_s0[j].requires_grad == True:
            node_IDs_s11.append(j)
        if val_s1[j].requires_grad == True:
            node_IDs_s22.append(j)
        if val_s2[j].requires_grad == True:
            node_IDs_s12.append(j)

    stress_all_coord = [(NewModel_du.coordinates[i]).clone().detach().requires_grad_(True) for i in range(len(NewModel_du.coordinates))]
    stress_all_cell_IDs = torch.tensor([torch.tensor(Domain_mesh_du.GetCellIds(i),dtype=torch.int)[0] for i in stress_all_coord])
    stress_all_coord = (torch.cat(stress_all_coord)).clone().detach().requires_grad_(True)

    # # # # # Model du # # # # # 
    original_coord_u = [(Model_u.coordinates[i]).clone().detach().requires_grad_(True) for i in range(len(Model_u.coordinates))]
    original_coord_u = (torch.cat(original_coord_u)).clone().detach().requires_grad_(True)

    val_u0 = [(NewModel_u.nodal_values[0][i]) for i in range(len(NewModel_u.coordinates))]
    val_u1 = [(NewModel_u.nodal_values[1][i]) for i in range(len(NewModel_u.coordinates))]

    node_IDs_u0 = []
    node_IDs_u1 = []

    for j in range(len(val_u0)):
        if val_u0[j].requires_grad == True:
            node_IDs_u0.append(j)
        if val_u1[j].requires_grad == True:
            node_IDs_u1.append(j)

    displ_all_coord = [(NewModel_u.coordinates[i]).clone().detach().requires_grad_(True) for i in range(len(NewModel_u.coordinates))]
    displ_all_cell_IDs = torch.tensor([torch.tensor(Domain_mesh_u.GetCellIds(i),dtype=torch.int)[0] for i in displ_all_coord])
    displ_all_coord = (torch.cat(displ_all_coord)).clone().detach().requires_grad_(True)

    # # # # # # # # # # # # # # # 

    old_stress = Model_du(stress_all_coord, stress_all_cell_IDs)
    old_displ = Model_u(displ_all_coord, displ_all_cell_IDs)

    for i in node_IDs_s11:
        if stress_all_cell_IDs[i] > -1:
            NewModel_du.nodal_values[0][i] = torch.nn.Parameter(torch.tensor([old_stress[0,i]]))
        else:
            dif = original_coord_du - stress_all_coord[i]
            dist = torch.norm(dif, dim=1)
            closest = torch.argmin(dist)
            NewModel_du.nodal_values[2][i] = torch.nn.Parameter(torch.tensor([old_stress[0,closest]]))

    for i in node_IDs_s22:
        if stress_all_cell_IDs[i] > -1:
            NewModel_du.nodal_values[1][i] = torch.nn.Parameter(torch.tensor([old_stress[1,i]]))
        else:
            dif = original_coord_du - stress_all_coord[i]
            dist = torch.norm(dif, dim=1)
            closest = torch.argmin(dist)
            NewModel_du.nodal_values[2][i] = torch.nn.Parameter(torch.tensor([old_stress[1,closest]]))

    for i in node_IDs_s12:
        if stress_all_cell_IDs[i] > -1:
            NewModel_du.nodal_values[2][i] = torch.nn.Parameter(torch.tensor([old_stress[2,i]]))
        else:
            dif = original_coord_du - stress_all_coord[i]
            dist = torch.norm(dif, dim=1)
            closest = torch.argmin(dist)
            NewModel_du.nodal_values[2][i] = torch.nn.Parameter(torch.tensor([old_stress[2,closest]]))

    # # # # # # # # # # # # # # # 

    for i in node_IDs_u0:
        if displ_all_cell_IDs[i] > -1:
            NewModel_u.nodal_values[0][i] = torch.nn.Parameter(torch.tensor([old_displ[0,i]]))
        else:
            dif = original_coord_u - displ_all_coord[i]
            dist = torch.norm(dif, dim=1)
            closest = torch.argmin(dist)
            NewModel_u.nodal_values[0][i] = Model_u.nodal_values[0][closest]

    for i in node_IDs_u1:
        if displ_all_cell_IDs[i] > -1:
            NewModel_u.nodal_values[1][i] = torch.nn.Parameter(torch.tensor([old_displ[1,i]]))
        else:
            dif = original_coord_u - displ_all_coord[i]
            dist = torch.norm(dif, dim=1)
            closest = torch.argmin(dist)
            NewModel_u.nodal_values[1][i] = Model_u.nodal_values[1][closest]



    return NewModel_u, NewModel_du

def OneCycle(MaxElemSize, MinElemSize, Model_u, Model_du, Domain_mesh_du, dimension, order_u, order_du, w0, w1 ):

    NewModel_u, NewModel_du, NewDomain_mesh_u, NewDomain_mesh_du = DefineModels(dimension, order_u, order_du, MaxElemSize, MinElemSize)
    NewModel_u, NewModel_du = InitializeWeights(NewModel_u, NewModel_du, Model_u, Model_du, Domain_mesh_u, Domain_mesh_du)
    NewModel_u, NewModel_du = DoTheRest(NewModel_u, NewModel_du, NewDomain_mesh_u, NewDomain_mesh_du,  E, mu, lmbda, L, w0, w1)

    return NewModel_u, NewModel_u, NewModel_du, NewDomain_mesh_u, NewDomain_mesh_du


########################################################################

num_sol_name_displ = "Square_case_1/Nodes_order2_displacement.npy"
num_sol_name_stress = "Square_case_1/Nodes_order1_stress.npy"


start_train_time = time.time()

MaxElemSize_init = 50
MinElemSize_init = 25

MaxElemSize = MaxElemSize_init
MinElemSize = MinElemSize_init
w0 = numpy.sqrt(500)
w1 = 1


Model_u, Model_du, Domain_mesh_u, Domain_mesh_du = DefineModels(dimension, order_u, order_du, MaxElemSize, MinElemSize)
# #Model_u, Model_du = Num_to_NN(Model_u, Model_du, num_sol_name_displ, num_sol_name_stress)
# Model_u.load_state_dict(torch.load("Results/Model_u.pt"))
# Model_du.load_state_dict(torch.load("Results/Model_du.pt"))  
Model_u, Model_du = DoTheRest( Model_u, Model_du, Domain_mesh_u, Domain_mesh_du,  E, mu, lmbda, L, w0, w1)


for i in range(3):

    MaxElemSize = MaxElemSize/5
    MinElemSize = MinElemSize/5

    Model_u, Model_u, Model_du, Domain_mesh_u, Domain_mesh_du = OneCycle(MaxElemSize, MinElemSize, Model_u, Model_du, Domain_mesh_du, dimension, order_u,\
                                                                order_du, w0, w1 )
                                                         

print("Total time : ", time.time() - start_train_time) 

torch.save(Model_u.state_dict(),"Results/Model_u.pt")
torch.save(Model_du.state_dict(),"Results/Model_du.pt")

cell_ids, ref_coords = Generate_Training_IDs(2, Domain_mesh_du)

TrialCoordinates, TrialIDs_u, TrialIDs_du = GetRealCoord(Model_du, Domain_mesh_du, cell_ids, ref_coords)

u_predicted = Model_u(TrialCoordinates, TrialIDs_u) 
du_predicted = Model_du(TrialCoordinates, TrialIDs_du)

du = torch.autograd.grad(u_predicted[0,:], TrialCoordinates, grad_outputs=torch.ones_like(u_predicted[0,:]), create_graph=True)[0]
dv = torch.autograd.grad(u_predicted[1,:], TrialCoordinates, grad_outputs=torch.ones_like(u_predicted[0,:]), create_graph=True)[0]
    
s_11, s_22, s_12 = Stress(du[:,0], dv[:,1], 0.5*(du[:,1] + dv[:,0]), lmbda, mu)

stress_NN = torch.stack([s_11, s_22, s_12],dim=1)


numpy.save("Results/u.npy", numpy.array(u_predicted.detach()))
numpy.save("Results/du.npy", numpy.array(du_predicted.detach()))
numpy.save("Results/stress_u.npy", numpy.array(stress_NN.detach()))
numpy.save("Results/coordinates.npy", numpy.array(TrialCoordinates.detach()))
numpy.save("Results/cell_IDs_u.npy", numpy.array(TrialIDs_u.detach()))
numpy.save("Results/cell_IDs_du.npy", numpy.array(TrialIDs_du.detach()))