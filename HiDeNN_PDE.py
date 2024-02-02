
import torch
import random
import torch.nn as nn
torch.set_default_dtype(torch.float64)

#%% Define the model for a 1D linear Beam mesh
    
class LinearBlock(nn.Module):
    """This is an implementation of the linear block 
     See [Zhang et al. 2021] Linear block. The input parameters are:
        - the coordinate x where the function is evaluated
        - If used for left part: x_b = x_i else if used right part x_b = x_ip1
        - If used for left part: x_a = x_im1 else if used right part x_a = x_i  """
    def __init__(self):
        super(LinearBlock, self).__init__()
        self.relu = nn.ReLU()

    def forward(self,x, x_a, x_b):
        mid = self.relu(x)
        mid = torch.nn.functional.linear(mid,(-1/(x_b-x_a)),torch.tensor([1],dtype=torch.float64))
        return self.relu(mid)

class Shapefunction(nn.Module):
    """This is an implementation of linear 1D shape function in a DNN representation
    See [Zhang et al. 2021] consisting of a linear(1,2) function leading the the Linear{Left+Right} layers which output then passes through Linear(2,1). 
    The input parameters are:
        - the coordinate x where the function is evaluated
        - the index i of the node associated to the shape function """
    
    def __init__(self, i):
        super(Shapefunction, self).__init__()
        # Index of the node associated to the shape function
        self.i = i
        # Layer end
        self.l2 = nn.Linear(2,1)
        # Set weight and bias for end layer and freeze them
        self.l2.weight.data.fill_(1)
        self.l2.bias.data.fill_(-1)
        self.l2.bias.requires_grad = False
        self.l2.weight.requires_grad = False
        # Defines the linear block function
        self.LinearBlock = LinearBlock()
        # Defines threshold so that two coordinates cannot go too close to one another
        self.threshold_p = torch.tensor(1-1/150,dtype=torch.float64)
        self.threshold_m = torch.tensor(1+1/150,dtype=torch.float64)
    
    def forward(self, x, coordinates):
        """ The forward function takes as an input the coordonate x at which the NN is evaluated and the parameters' list coordinates where the nodes' corrdinates of the mesh are stored"""
        i = self.i
        # For the SF on the left
        if i == -1: # for the outter Shape functions the index acts as a tag, -1 for the left, -2 for the right
            x_i = coordinates[0]
            x_im1 = coordinates[0]-coordinates[-1]*1/100
            x_ip1 = coordinates[0+1]
        # For the SF on the right
        elif i == -2: # for the outter Shape functions the index acts as a tag, -1 for the left, -2 for the right
            x_i = coordinates[-1]
            x_im1 = coordinates[-2]
            x_ip1 = coordinates[-1]*(1+1/100)
        else:
            x_i = coordinates[i]
            x_im1 = coordinates[i-1]
            x_ip1 = coordinates[i+1] 

        #  Stop nodes from getting too close 
        x_i = torch.minimum(x_i, self.threshold_p*x_ip1)
        x_i = torch.maximum(x_i, self.threshold_m*x_im1)
            
        l1 = torch.nn.functional.linear(x,torch.tensor([[-1],[1]],
                                                       dtype=torch.float64),
                                                       torch.tensor([1,-1])*x_i[0])
        top = self.LinearBlock(l1[:,0].view(-1,1),x_im1,x_i)
        bottom = self.LinearBlock(l1[:,1].view(-1,1),x_i,x_ip1)
        l2 = torch.cat((top,bottom),1)
        l3 = self.l2(l2)
        return l3

class MeshNN(nn.Module):
    """This is the main Neural Network building the FE interpolation, the coordinates parameters are trainable are correspond to the coordinates of the nodes in the Mesh which are passed as parameters to the sub NN where they are fixed. 
    Updating those parameters correspond to r-adaptativity
    The Interpolation layer weights correspond to the nodal values. Updating them is equivqlent to solving the PDE. """
    def __init__(self, np, L, alpha = 0.005):
        super(MeshNN, self).__init__()
        self.alpha = alpha # set the weight for the Mesh regularisation 
        self.coordinates = nn.ParameterList([nn.Parameter(torch.tensor([[i]])) for i in torch.linspace(0,L,np)])
        self.np = np 
        self.L = L 
        self.Functions = nn.ModuleList([Shapefunction(i) for i in range(1,np-1)])
        self.InterpoLayer_uu = nn.Linear(self.np-2,1,bias=False)
        self.NodalValues_uu = nn.Parameter(data=torch.ones(np-2), requires_grad=False)
        self.InterpoLayer_uu.weight.data = self.NodalValues_uu
        self.Functions_dd = nn.ModuleList([Shapefunction(-1),
                                                         Shapefunction(-2)])
        self.InterpoLayer_dd = nn.Linear(2,1,bias=False)
        self.InterpoLayer_dd.weight.requires_grad = False
        self.SumLayer = nn.Linear(2,1,bias=False)
        self.SumLayer.weight.data.fill_(1)
        self.SumLayer.weight.requires_grad = False


    def forward(self,x):
        # Compute shape functions 
        intermediate_uu = [self.Functions[l](x,self.coordinates) for l in range(self.np-2)]
        intermediate_dd = [self.Functions_dd[l](x,self.coordinates) for l in range(2)]
        out_uu = torch.cat(intermediate_uu, dim=1)
        out_dd = torch.cat(intermediate_dd, dim=1)
        u_uu = self.InterpoLayer_uu(out_uu)
        u_dd = self.InterpoLayer_dd(out_dd)
        u = torch.stack((u_uu,u_dd), dim=1)
        return self.SumLayer(u)
        ######### LEGACY  #############
        # Previous implementation (that gives slightly different results)
        # out_uu = torch.stack(intermediate_uu)
        # out_dd = torch.stack(intermediate_dd)
        # u_uu = self.InterpoLayer_uu(out_uu.T)
        # u_dd = self.InterpoLayer_dd(out_dd.T)
        # u = torch.cat((u_uu,u_dd),0)
        # return self.SumLayer(u.T)
        ######### LEGACY  #############
    
    def SetBCs(self,u_0,u_L):
        """Set the two Dirichlet boundary conditions
            Inputs are:
                - u_0 the left BC
                - u_L the right BC """
        self.u_0 = torch.tensor(u_0, dtype=torch.float64)
        self.u_L = torch.tensor(u_L, dtype=torch.float64)
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


#%% Application of the NN
# Geometry of the Mesh
L = 10                           # Length of the Beam
np = 23                          # Number of Nodes in the Mesh
A = 1                            # Section of the beam
E = 175                          # Young's Modulus (should be 175)
alpha =0.005                     # Weight for the Mesh regularisation 
MeshBeam = MeshNN(np,L,alpha)    # Creates the associated model
# Boundary conditions
u_0 = 0                     #Left BC
u_L = 0                     #Right BC
MeshBeam.SetBCs(u_0,u_L)
# Import mechanical functions

from Bin.PDE_Library import RHS, PotentialEnergy, \
    PotentialEnergyVectorised, AlternativePotentialEnergy, \
        Derivative, AnalyticGradientSolution, AnalyticSolution


# Set the coordinates as trainable
MeshBeam.UnFreeze_Mesh()
# Set the coordinates as untrainable
# MeshBeam.Freeze_Mesh()
# Set the require output requirements
BoolPlot = False             # Bool for plots used for gif
BoolCompareNorms = True      # Bool for comparing energy norm to L2 norm


#%% Define loss and optimizer
learning_rate = 0.001
n_epochs = 5000
optimizer = torch.optim.Adam(MeshBeam.parameters(), lr=learning_rate)
MSE = nn.MSELoss()


#%% Training loop
TrialCoordinates = torch.tensor([[i/50] for i in range(2,500)], 
                                dtype=torch.float64, requires_grad=True)
# Store the initial coordinates before training (could be merged with Coord_trajectories)
InitialCoordinates = [MeshBeam.coordinates[i].data.item() for i in range(len(MeshBeam.coordinates))]
error = []              # Stores the loss
error2 = []             # Stores the L2 error compared to the analytical solution
Coord_trajectories = [] # Stores the trajectories of the coordinates while training

for epoch in range(n_epochs):
    # predict = forward pass with our model
    u_predicted = MeshBeam(TrialCoordinates) 
    # loss (several ways to compute the energy loss)
    # l = AlternativePotentialEnergy(A,E,u_predicted,TrialCoordinates,RHS(TrialCoordinates))
    l = PotentialEnergyVectorised(A,E,u_predicted,TrialCoordinates,RHS(TrialCoordinates))
    # l = PotentialEnergy(A,E,u_predicted,TrialCoordinates,RHS(TrialCoordinates))

    # Mesh regularisation term
    # Compute the ratio of the smallest jacobian and the largest jacobian
    Jacobians = [MeshBeam.coordinates[i]-MeshBeam.coordinates[i-1] for i in range(1,len(MeshBeam.coordinates))]
    Jacobians = torch.stack(Jacobians)
    Ratio = torch.max(Jacobians)/torch.min(Jacobians)
    # Add the ratio to the loss
    l+=MeshBeam.alpha*(Ratio-1)

    # calculate gradients = backward pass
    l.backward()
    # update weights
    optimizer.step()
    # zero the gradients after updating
    optimizer.zero_grad()

    # Training strategy
    # if epoch >= 100:
    #     MeshBeam.Freeze_Mesh()
    #     MeshBeam.UnFreeze_FEM()

    with torch.no_grad():
        # Stores the loss
        error.append(l.item())
        # Stores the coordinates trajectories
        Coordinates_i = [MeshBeam.coordinates[i].data.item() for i in range(len(MeshBeam.coordinates))]
        Coord_trajectories.append(Coordinates_i)
        if BoolCompareNorms:
            # Copute and store the L2 error w.r.t. the analytical solution
            error2.append(MSE(AnalyticSolution(A,E,TrialCoordinates.data),u_predicted).data)

    if (epoch+1) % 100 == 0:
        print('epoch ', epoch+1, ' loss = ', l.item())
        if BoolPlot:
            Pplot.PlotSolution_Coordinates_Analytical(A,E,InitialCoordinates,Coordinates_i,
                                          TrialCoordinates,AnalyticSolution,MeshBeam,
                                          '/Gifs/Solution_'+str(epoch))
            Pplot.PlotGradSolution_Coordinates_Analytical(A,E,InitialCoordinates,Coordinates_i,
                                              TrialCoordinates,AnalyticGradientSolution,
                                          '/Gifs/Solution_gardient_'+str(epoch))
    

#%% Post-processing
# Retrieve coordinates
Coordinates = Coord_trajectories[-1]

import Post.Plots as Pplot
# Tests on trained data and compare to reference
Pplot.PlotSolution_Coordinates_Analytical(A,E,InitialCoordinates,Coordinates,
                                          TrialCoordinates,AnalyticSolution,MeshBeam,
                                          'Solution_displacement')
# Plots the gradient & compare to reference
Pplot.PlotGradSolution_Coordinates_Analytical(A,E,InitialCoordinates,Coordinates,
                                              TrialCoordinates,AnalyticGradientSolution,
                                              MeshBeam,Derivative,'Solution_gradients')

# plots zoomed energy loss
Pplot.PlotEnergyLoss(error,0,'Loss')

# plots zoomed energy loss
Pplot.PlotEnergyLoss(error,2500,'Loss_zoomed')

# Plots trajectories of the coordinates while training
Pplot.PlotTrajectories(Coord_trajectories,'Trajectories')

if BoolCompareNorms:
    Pplot.Plot_Compare_Loss2l2norm(error,error2,'Loss_Comaprison')

# %% Test of mesh and parser
    
# User efines all boundary conditions 
DirichletDictionryList = [{"Entity": 1, 
                           "Value": 0, 
                           "normal":1}, 
                            {"Entity": 2, 
                             "Value": 10, 
                             "normal":1}]

import Bin.Pre_processing as pre

Beam_mesh = pre.Mesh('Beam',1)
Volume_element = 100   # Volume element correspond to the 1D elem in 1D
Beam_mesh.AddBCs(Volume_element,DirichletDictionryList) 
Beam_mesh.MeshGeo()
# %%
