
import torch
import random
import torch.nn as nn
import numpy as np



#%% Define the model for a 1D linear Beam mesh
class LinearLeft(nn.Module):
    """This is an implementation of the left (increasing) part of the linear shape function in a DNN representation
     See [Zhang et al. 2021] Linear block. The input parameters are:
        - the coordinate x where the function is evaluated
        - the left node of the element x_im1
        - the node associated to the shape functino being built x_i   """
    def __init__(self):
        super(LinearLeft, self).__init__()
        self.relu = nn.ReLU()

    def forward(self,x, x_im1, x_i):
        mid = self.relu(x)
        mid = torch.nn.functional.linear(mid,(-1/(x_i-x_im1)),torch.tensor([1],dtype=torch.float32))
        return self.relu(mid)

class LinearRight(nn.Module):
    """This is an implementation of the Right (decreasing) part of the linear shape function in a DNN representation
     See [Zhang et al. 2021] Linear block. The input parameters are:
        - the coordinate x where the function is evaluated
        - the right node of the element x_ip1
        - the node associated to the shape functino being built x_i   """
    def __init__(self):
        super(LinearRight, self).__init__()
        self.relu = nn.ReLU()

    def forward(self,x,x_ip1,x_i):
        mid = self.relu(x)
        mid = torch.nn.functional.linear(mid,(-1/(x_ip1-x_i)),torch.tensor([1],dtype=torch.float32))
        return self.relu(mid)

class Shapefunction(nn.Module):
    """This is an implementation of linear 1D shape function in a DNN representation
    See [Zhang et al. 2021] consisting of a linear(1,2) function leading the the Linear{Left+Right} layers which output then passes through Linear(2,1). 
    The input parameters are:
        - the coordinate x where the function is evaluated
        - the index i of the node associated to the shape function 
        
        All the weights and biais are set as untranable and are evaluated given the nodal position fed while initialising the network"""
    
    def __init__(self, i,x_i, r_adaptivity = False):
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
        # Defines the right and left linear functions
        self.Linears = nn.ModuleList([LinearLeft(),LinearRight()])
        # Defines threshold so that two coordinates cannot go too close to one another
        self.threshold_p = torch.tensor(1-1/150,dtype=torch.float32)
        self.threshold_m = torch.tensor(1+1/150,dtype=torch.float32)
    
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
                                                       dtype=torch.float32),
                                                       torch.tensor([1,-1])*x_i[0])
        top = self.Linears[0](l1[:,0].view(-1,1),x_im1,x_i)
        bottom = self.Linears[1](l1[:,1].view(-1,1),x_ip1,x_i)
        l2 = torch.cat((top,bottom),1)
        l3 = self.l2(l2)
        return l3

class MeshNN(nn.Module):
    """This is the main Neural Network building the FE interpolation, the coordinates parameters are trainable are correspond to the coordinates of the nodes in the Mesh which are passed as parameters to the sub NN where they are fixed. 
    Updating those parameters correspond to r-adaptativity
    The Interpolation layer weights correspond to the nodal values. Updating them is equivqlent to solving the PDE. """
    def __init__(self, np, L):
        super(MeshNN, self).__init__()
        self.coordinates = nn.ParameterList([nn.Parameter(torch.tensor([[i]])) for i in torch.linspace(0,L,np)])
        self.np = np 
        self.L = L 
        self.Functions = nn.ModuleList([Shapefunction(i,self.coordinates[i],
                                                       r_adaptivity = False) for i in range(1,np-1)])
        self.InterpoLayer_uu = nn.Linear(self.np-2,1,bias=False)
        self.NodalValues_uu = nn.Parameter(data=torch.ones(np-2), requires_grad=False)
        self.InterpoLayer_uu.weight.data = self.NodalValues_uu
        self.Functions_dd = nn.ModuleList([Shapefunction(-1,
                                                         self.coordinates[0], 
                                                         r_adaptivity = False),
                                                         Shapefunction(-2,self.coordinates[-1], 
                                                                       r_adaptivity = False)])
        self.InterpoLayer_dd = nn.Linear(2,1,bias=False)
        self.InterpoLayer_dd.weight.requires_grad = False
        self.SumLayer = nn.Linear(2,1,bias=False)
        self.SumLayer.weight.data.fill_(1)
        self.SumLayer.weight.requires_grad = False


    def forward(self,x):
        # Compute shape functions 
        intermediate_uu = [self.Functions[l](x,self.coordinates) for l in range(self.np-2)]
        intermediate_dd = [self.Functions_dd[l](x,self.coordinates) for l in range(2)]
        ####################### more eleguant but differs
        # For some reason the results differ slightly with this implementation
        # out_uu = torch.hstack(intermediate_uu)
        # out_dd = torch.hstack(intermediate_dd)
        # u_uu = self.InterpoLayer_uu(out_uu)
        # u_dd = self.InterpoLayer_dd(out_dd)
        # u = torch.cat((u_uu.view(-1,1),u_dd.view(-1,1)),1)
        # return self.SumLayer(u)
        #######################
        out_uu = torch.stack(intermediate_uu)
        out_dd = torch.stack(intermediate_dd)
        u_uu = self.InterpoLayer_uu(out_uu.T)
        u_dd = self.InterpoLayer_dd(out_dd.T)
        u = torch.cat((u_uu,u_dd),0)
        return self.SumLayer(u.T)
    
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


#%% Application of the NN
# Geometry of the Mesh
L = 10                      # Length of the Beam
np = 23                     # Number of Nodes in the Mesh
A = 1                       # Section of the beam
E = 175                     # Young's Modulus (should be 175)
MeshBeam = MeshNN(np,L)     # Creates the associated model
# Boundary conditions
u_0 = 0                     #Left BC
u_L = 0                     #Right BC
MeshBeam.SetBCs(u_0,u_L)
# Set the coordinates as trainable
MeshBeam.UnFreeze_Mesh()
BoolPlot = False             # Bool for plots used for gif
BoolCompareNorms = True      # Bool for comparing energy norm to L2 norm
MSE = nn.MSELoss()


#%% Define loss and optimizer
learning_rate = 0.001
n_epochs = 5000
optimizer = torch.optim.Adam(MeshBeam.parameters(), lr=learning_rate)
import numpy as np

def RHS(x):
    """Defines the right hand side (RHS) of the equation (the body force)"""
    b = -(4*np.pi**2*(x-2.5)**2-2*np.pi)/(torch.exp(np.pi*(x-2.5)**2)) \
        - (8*np.pi**2*(x-7.5)**2-4*np.pi)/(torch.exp(np.pi*(x-7.5)**2))
    return  b

def PotentialEnergy(A,E,u,x,b):
    """Computes the potential energy of the Beam, which will be used as the loss of the HiDeNN"""
    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    integral = 0
    for i in range(1,u.data.shape[0]):
        integral += (0.25*A*E*(x[i]-x[i-1])*(du_dx[i]**2+du_dx[i-1]**2)) \
            - 0.5*((x[i]-x[i-1])*(u[i]*b[i]+u[i-1]*b[i-1]))
    return integral

def PotentialEnergyVectorised(A, E, u, x, b):
    """Computes the potential energy of the Beam, which will be used as the loss of the HiDeNN"""
    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    # Calculate dx
    dx = x[1:] - x[:-1]
    # Vectorised calculation of the integral terms
    int_term1 = 0.25 * A * E * dx * (du_dx[1:]**2 + du_dx[:-1]**2)
    int_term2 = 0.5 * dx * (u[1:] * b[1:] + u[:-1] * b[:-1])

    # Vectorised calculation of the integral using the trapezoidal rule
    integral = torch.sum(int_term1 - int_term2)

    return integral

def AlternativePotentialEnergy(A,E,u,x,b):
    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    f_x = 0.5*(A*E*du_dx**2) - u*b
    f_x = f_x.view(-1)
    dx = torch.diff(x.view(-1))
    av = 0.5*(f_x[1:]+f_x[:-1])*dx
    return torch.sum(av)

def Derivative(u,x):
    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    return du_dx

def AnalyticSolution(A,E,x):
    out = (1/(A*E)*(torch.exp(-np.pi*(x-2.5)**2)-np.exp(-6.25*np.pi))) \
        + (2/(A*E)*(torch.exp(-np.pi*(x-7.5)**2)-np.exp(-56.25*np.pi))) \
            - (x/(10*A*E))*(np.exp(-6.25*np.pi) - np.exp(-56.25*np.pi))
    return out

def AnalyticGradientSolution(A,E,x):
    out = (2/(A*E)*((-np.pi)*(x-2.5)*torch.exp(-np.pi*(x-2.5)**2))) \
        + (4/(A*E)*((-np.pi)*(x-7.5)*torch.exp(-np.pi*(x-7.5)**2))) \
            - (1/(10*A*E))*(np.exp(-6.25*np.pi) - np.exp(-56.25*np.pi))
    return out

#%% Training loop
TrialCoordinates = torch.tensor([[i/50] for i in range(2,500)], 
                                dtype=torch.float32, requires_grad=True)
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

# %%
