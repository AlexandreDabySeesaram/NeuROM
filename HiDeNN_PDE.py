#%% Libraries import
import time 
# Import pre-processing functions
import Bin.Pre_processing as pre
# Import mechanical functions
from Bin.PDE_Library import RHS, PotentialEnergyVectorised, \
        Derivative, AnalyticGradientSolution, AnalyticSolution
# Import torch librairies
import torch
import torch.nn as nn
torch.set_default_dtype(torch.float32)
#Import post processing libraries
import Post.Plots as Pplot
import matplotlib.pyplot as plt
mps_device = torch.device("mps")

#%% Define the model for a 1D linear Beam mesh
class LinearBlock(nn.Module):
    """This is the new implementation of the linear block 
     See [Zhang et al. 2021] Linear block. The input parameters are:
        - the coordinate x where the function is evaluated
        - If used for left part: x_b = x_i else if used right part x_b = x_ip1
        - If used for left part: x_a = x_im1 else if used right part x_a = x_i  """
    def __init__(self):
        super(LinearBlock, self).__init__()
        self.relu = nn.ReLU()

    def forward(self,x, x_a, x_b, y_a, y_b):
        
        mid = self.relu(-x + x_b)
        mid = self.relu(1 - mid/(x_b-x_a))
        mid = (y_b-y_a)*mid + y_a
        return mid

class ElementBlock_Bar_2(nn.Module):
    """Bar 2 (linear 1D) element block
    Returns the N_i(x)'s for each nodes within the element"""
    def __init__(self, i , connectivity):
        super(ElementBlock_Bar_2, self).__init__()
        self.i = i
        self.LinearBlock = LinearBlock()
        self. connectivity = connectivity

    def forward(self, x, coordinates):
        i = self.i
        # For the outter Nodes, phantom elements are created 
        #  to cancel out shape functions beyond the geometry of the structure in question, 
        # to prevent any form of extrapolation beyond its boundaries 
        if i ==-1:
            x_left = coordinates[0]-coordinates[1]/100
            x_right = coordinates[0]  
        elif i ==-2:
            x_left = coordinates[1]
            x_right = coordinates[1]*(1+1/100) 
        else:
            x_left = coordinates[self.connectivity[i,0].astype(int)-1]
            x_right = coordinates[self.connectivity[i,-1].astype(int)-1]

        left = self.LinearBlock(x, x_left, x_right, 0, 1)
        right = self.LinearBlock(x, x_left, x_right, 1, 0)
        # out = torch.cat((left, right),dim=1)

        out = torch.cat((right, left),dim=1)


        return out


class MeshNN(nn.Module):
    """This is the main Neural Network building the FE interpolation, the coordinates 
    parameters are trainable are correspond to the coordinates of the nodes in the Mesh 
    which are passed as parameters to the sub NN where they are fixed. 
    Updating those parameters correspond to r-adaptativity
    The Interpolation layer weights correspond to the nodal values. Updating them 
    is equivqlent to solving the PDE. """
    def __init__(self, mesh, alpha = 0.005):
        super(MeshNN, self).__init__()
        self.alpha = alpha # set the weight for the Mesh regularisation 
        self.coordinates = nn.ParameterList([nn.Parameter(torch.tensor([[mesh.Nodes[i][1]]])) \
                                             for i in range(len(mesh.Nodes))])
        self.dofs = mesh.NNodes*mesh.dim # Number of Dofs
        self.NElem = mesh.NElem
        self.NBCs = len(mesh.ListOfDirichletsBCsIds) # Number of prescribed Dofs
        self.Functions = nn.ModuleList([ElementBlock_Bar_2(i,mesh.Connectivity) \
                                        for i in range(self.NElem)])
        self.InterpoLayer_uu = nn.Linear(self.dofs-self.NBCs,1,bias=False)
        self.NodalValues_uu = nn.Parameter(data=0.1*torch.ones(self.dofs-self.NBCs), requires_grad=False)
        self.InterpoLayer_uu.weight.data = self.NodalValues_uu
        self.Functions_dd = nn.ModuleList([ElementBlock_Bar_2(-1,mesh.Connectivity),
                                           ElementBlock_Bar_2(-2,mesh.Connectivity)])
        self.AssemblyLayer = nn.Linear(2*(self.NElem+2),self.dofs,bias=False)
        # self.AssemblyLayer.weight.data = torch.tensor(mesh.weights_assembly_total)
        self.AssemblyLayer.weight.data = torch.tensor(mesh.weights_assembly_total,dtype=torch.float32).detach()
        self.AssemblyLayer.weight. requires_grad=False

        self.InterpoLayer_dd = nn.Linear(2,1,bias=False)
        self.InterpoLayer_dd.weight.requires_grad = False
        self.SumLayer = nn.Linear(2,1,bias=False)
        self.SumLayer.weight.data.fill_(1)
        self.SumLayer.weight.requires_grad = False


    def forward(self,x):
        # Compute shape functions 
        intermediate_uu = [self.Functions[l](x,self.coordinates) for l in range(self.NElem)]
        intermediate_dd = [self.Functions_dd[l](x,self.coordinates) for l in range(2)]
        out_uu = torch.cat(intermediate_uu, dim=1)
        out_dd = torch.cat(intermediate_dd, dim=1)
        joined_vector = torch.cat((out_uu,out_dd),dim=1)
        recomposed_vector_u = self.AssemblyLayer(joined_vector) -1
        u_u = self.InterpoLayer_uu(recomposed_vector_u[:,2:])
        u_d = self.InterpoLayer_dd(recomposed_vector_u[:,:2])
        u = torch.stack((u_u,u_d), dim=1)
        return self.SumLayer(u)
    
    def SetBCs(self,u_0,u_L):
        """Set the two Dirichlet boundary conditions
            Inputs are:
                - u_0 the left BC
                - u_L the right BC """
        self.u_0 = torch.tensor(u_0, dtype=torch.float32)
        self.u_L = torch.tensor(u_L, dtype=torch.float32)
        self.InterpoLayer_dd.weight.data = torch.tensor([self.u_0,self.u_L], requires_grad=False)
        self.InterpoLayer_dd.weight.requires_grad = False

    def Freeze_Mesh(self):
        """Set the coordinates as untrainable parameters"""
        for param in self.coordinates:
            param.requires_grad = False

    def UnFreeze_Mesh(self):
        """Set the coordinates as trainable parameters """
        for param in self.coordinates:
            param.requires_grad = True
        #Freeze external coorinates to keep geometry    
        self.coordinates[0].requires_grad = False
        self.coordinates[-1].requires_grad = False

    def UnFreeze_FEM(self):
        """Set the nodale values as trainable parameters """
        self.InterpoLayer_uu.weight.requires_grad = True
            
    def Freeze_FEM(self):
        """Set the nodale values as untrainable parameters """
        self.InterpoLayer_uu.weight.requires_grad = False
np_vect = []
Total_time_vect= []
evaluation_time_vect= []
Loss_time_vect= []
Backward_time_vect=[]
optimiser_time_vect=[]
#%% Pre-processing (could be put in config file later)
# Geometry of the Mesh
L = 10                                      # Length of the Beam
np = 100                                     # Number of Nodes in the Mesh
A = 1                                       # Section of the beam
E = 175                                     # Young's Modulus (should be 175)
alpha =0.005                                # Weight for the Mesh regularisation 

# User defines all boundary conditions 
DirichletDictionryList = [{"Entity": 1, 
                           "Value": 0, 
                           "normal":1}, 
                            {"Entity": 2, 
                             "Value": 10, 
                             "normal":1}]

MaxElemSize = L/(np-1)                      # Compute element size
Beam_mesh = pre.Mesh('Beam',MaxElemSize)    # Create the mesh object
Volume_element = 100                        # Volume element correspond to the 1D elem in 1D
Beam_mesh.AddBCs(Volume_element,
                 DirichletDictionryList)    # Include Boundary physical domains infos (BCs+volume)
Beam_mesh.MeshGeo()                         # Mesh the .geo file if .msh does not exist
Beam_mesh.ReadMesh()                        # Parse the .msh file
Beam_mesh.AssemblyMatrix()                  # Build the assembly weight matrix


#%% Application of the NN
BeamModel = MeshNN(Beam_mesh,alpha)     # Create the associated model
# Boundary conditions
u_0 = 0                                 #Left BC
u_L = 0                                 #Right BC
BeamModel.SetBCs(u_0,u_L)

# Set the coordinates as trainable
BeamModel.UnFreeze_Mesh()
# Set the coordinates as untrainable
BeamModel.Freeze_Mesh()
# Set the require output requirements
BoolPlot = False                        # Boolean for plots used for gif
BoolPlotPost = False                    # Boolean for plots used for Post
BoolCompareNorms = False                # Boolean for comparing energy norm to L2 norm
BoolGPU = False                         # Boolean enabling GPU computations (autograd function is not working currently on mac M2)




#%% Define loss and optimizer
learning_rate = 0.001
n_epochs = 200
optimizer = torch.optim.Adam(BeamModel.parameters(), lr=learning_rate)
MSE = nn.MSELoss()



#%% Training loop
TrialCoordinates = torch.tensor([[i/50] for i in range(2,500)], 
                                dtype=torch.float32, requires_grad=True)

# If GPU
if BoolGPU:
    BeamModel.to(mps_device)
    TrialCoordinates = torch.tensor([[i/50] for i in range(2,500)], 
                                dtype=torch.float32, requires_grad=True).to(mps_device)

# Store the initial coordinates before training (could be merged with Coord_trajectories)
InitialCoordinates = [BeamModel.coordinates[i].data.item() for i in range(len(BeamModel.coordinates))]
error = []              # Stores the loss
error2 = []             # Stores the L2 error compared to the analytical solution
Coord_trajectories = [] # Stores the trajectories of the coordinates while training

print("**************** START TRAINING ***************\n")
start_train_time = time.time()

evaluation_time = 0
loss_time = 0
optimizer_time = 0
backward_time = 0
for epoch in range(n_epochs):
    # predict = forward pass with our model
    start_time = time.time()
    u_predicted = BeamModel(TrialCoordinates) 
    evaluation_time += time.time() - start_time
    start_time = time.time()
    # loss 
    l = PotentialEnergyVectorised(A,E,u_predicted,TrialCoordinates,RHS(TrialCoordinates))
    loss_time += time.time() - start_time
    TrialCoordinates.requires_grad = False
    # Mesh regularisation term
    # Compute the ratio of the smallest jacobian and the largest jacobian
    # Jacobians = [BeamModel.coordinates[i]-BeamModel.coordinates[i-1] \
    #              for i in range(1,len(BeamModel.coordinates))]
    # Jacobians = torch.stack(Jacobians)
    # Ratio = torch.max(Jacobians)/torch.min(Jacobians)
    # # Add the ratio to the loss
    # l+=BeamModel.alpha*(Ratio-1)
    # calculate gradients = backward pass
    start_time = time.time()
    l.backward()
    backward_time += time.time() - start_time
    TrialCoordinates.requires_grad = True
    # update weights
    start_time = time.time()
    optimizer.step()
    optimizer_time += time.time() - start_time
    # zero the gradients after updating
    optimizer.zero_grad()


    with torch.no_grad():
        # Stores the loss
        error.append(l.item())
        # Stores the coordinates trajectories
        Coordinates_i = [BeamModel.coordinates[i].data.item() \
                         for i in range(len(BeamModel.coordinates))]
        Coord_trajectories.append(Coordinates_i)
        if BoolCompareNorms:
            # Copute and store the L2 error w.r.t. the analytical solution
            error2.append(MSE(AnalyticSolution(A,E,TrialCoordinates.data),u_predicted).data)

    if (epoch+1) % 100 == 0:
        print('epoch ', epoch+1, ' loss = ', l.item())
        
stopt_train_time = time.time()

print("*************** END OF TRAINING ***************\n")

print(f'* Training time: {stopt_train_time-start_train_time}s\n\
* Evaluation time: {evaluation_time}s\n\
* Loss time: {loss_time}s\n\
* Backward time: {backward_time}s\n\
* Optimiser time: {optimizer_time}s\n')
np_vect.append(np)
Total_time_vect.append(stopt_train_time-start_train_time)
evaluation_time_vect.append(evaluation_time)
Loss_time_vect.append(loss_time)
Backward_time_vect.append(backward_time)
optimiser_time_vect.append(optimizer_time)





#%% Post-processing
if BoolPlotPost:
    plt.plot(np_vect, evaluation_time_vect, label = 'Evaluation time')
    plt.plot(np_vect, Total_time_vect, label = 'Total time')
    plt.plot(np_vect, Loss_time_vect, label = 'Loss time')
    plt.plot(np_vect, Backward_time_vect, label = 'Backward time')
    plt.plot(np_vect, optimiser_time_vect, label = 'Optimiser time')
    plt.xlabel('Dofs', fontsize=15)
    plt.ylabel('Time (s)', fontsize=15)
    plt.legend(loc="upper left", fontsize=15)
    plt.savefig('Results/Profiler.pdf', transparent=True)  


    # Retrieve coordinates
    Coordinates = Coord_trajectories[-1]
    # Tests on trained data and compare to reference
    Pplot.PlotSolution_Coordinates_Analytical(A,E,InitialCoordinates,Coordinates,
                                            TrialCoordinates,AnalyticSolution,BeamModel,
                                            'Solution_displacement')
    # Plots the gradient & compare to reference
    Pplot.PlotGradSolution_Coordinates_Analytical(A,E,InitialCoordinates,Coordinates,
                                                TrialCoordinates,AnalyticGradientSolution,
                                                BeamModel,Derivative,'Solution_gradients')
    # plots zoomed energy loss
    Pplot.PlotEnergyLoss(error,0,'Loss')

    # plots zoomed energy loss
    Pplot.PlotEnergyLoss(error,2500,'Loss_zoomed')

    # Plots trajectories of the coordinates while training
    Pplot.PlotTrajectories(Coord_trajectories,'Trajectories')

    if BoolCompareNorms:
        Pplot.Plot_Compare_Loss2l2norm(error,error2,'Loss_Comaprison')

# %%
