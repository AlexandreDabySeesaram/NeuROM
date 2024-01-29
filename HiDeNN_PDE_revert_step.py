
import torch
import random
import torch.nn as nn
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg')
import numpy as np
import copy
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

torch.set_default_dtype(torch.float64)


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
        mid = torch.nn.functional.linear(mid,(-1/(x_i-x_im1)),torch.tensor([1],dtype=torch.double))
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
        mid = torch.nn.functional.linear(mid,(-1/(x_ip1-x_i)),torch.tensor([1],dtype=torch.double))
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
        self.threshold_p = torch.tensor(1-1/150,dtype=torch.double)
        self.threshold_m = torch.tensor(1+1/150,dtype=torch.double)
    
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

        l1 = torch.nn.functional.linear(x, torch.tensor([[-1],[1]],dtype=torch.double), torch.tensor([1,-1],dtype=torch.double)*x_i[0])
        top = self.Linears[0](l1[:,0].view(-1,1),x_im1,x_i)
        bottom = self.Linears[1](l1[:,1].view(-1,1),x_ip1,x_i)
        l2 = torch.cat((top,bottom),1)
        l3 = self.l2(l2)
        return l3

class MeshNN(nn.Module):
    """This is the main Neural Network building the FE interpolation, the coordinates parameters are trainable are correspond to the coordinates of the nodes in the Mesh which are passed as parameters to the sub NN where they are fixed. 
    Updating those parameters correspond to r-adaptativity
    The Interpolation layer weights correspond to the nodal values. Updating them is equivqlent to solving the PDE. """
    def __init__(self, np, L, alpha = 0.005):
        super(MeshNN, self).__init__()
        self.coordinates = nn.ParameterList([nn.Parameter(torch.tensor([[i]], dtype=torch.double)) for i in torch.linspace(0,L,np)])
        self.np = np 
        self.L = L 
        self.Functions = nn.ModuleList([Shapefunction(i,self.coordinates[i], r_adaptivity = False) for i in range(1,np-1)])
        self.InterpoLayer_uu = nn.Linear(self.np-2,1,bias=False)
        self.NodalValues_uu = nn.Parameter(data=0.5*torch.ones(np-2), requires_grad=False)
        self.InterpoLayer_uu.weight.data = self.NodalValues_uu
        self.Functions_dd = nn.ModuleList([Shapefunction(-1,self.coordinates[0], r_adaptivity = False),Shapefunction(-2,self.coordinates[-1], r_adaptivity = False)])
        self.InterpoLayer_dd = nn.Linear(2,1,bias=False)
        self.InterpoLayer_dd.weight.requires_grad = False
        self.SumLayer = nn.Linear(2,1,bias=False)
        self.SumLayer.weight.data.fill_(1)
        self.SumLayer.weight.requires_grad = False
        self.alpha = alpha


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
    
    def SetBCs(self,u_0,u_L):
        """Set the two Dirichlet boundary conditions
            Inputs are:
                - u_0 the left BC
                - u_L the right BC """
        self.u_0 = torch.tensor(u_0, dtype=torch.double)
        self.u_L = torch.tensor(u_L, dtype=torch.double)
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
n_points = 23                     # Number of Nodes in the Mesh
A = 1                       # Section of the beam
E = 175                     # Young's Modulus (should be 175)
MeshBeam = MeshNN(n_points,L, 0)     # Creates the associated model
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
learning_rate = 1.0e-3
n_epochs = 15000
optimizer = torch.optim.SGD(MeshBeam.parameters(), lr=learning_rate)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, cooldown=50, factor=0.25)


def RHS(x):
    """Defines the right hand side (RHS) of the equation (the body force)"""
    b = -(4*np.pi**2*(x-2.5)**2-2*np.pi)/(torch.exp(np.pi*(x-2.5)**2)) - (8*np.pi**2*(x-7.5)**2-4*np.pi)/(torch.exp(np.pi*(x-7.5)**2))
    return  b

def PotentialEnergy(A,E,u,x,b):
    """Computes the potential energy of the Beam, which will be used as the loss of the HiDeNN"""
    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    integral = 0
    for i in range(1,u.data.shape[0]):
        integral += (0.25*A*E*(x[i]-x[i-1])*(du_dx[i]**2+du_dx[i-1]**2)) - 0.5*((x[i]-x[i-1])*(u[i]*b[i]+u[i-1]*b[i-1]))
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
    out = (1/(A*E)*(torch.exp(-np.pi*(x-2.5)**2)-np.exp(-6.25*np.pi))) + (2/(A*E)*(torch.exp(-np.pi*(x-7.5)**2)-np.exp(-56.25*np.pi))) - (x/(10*A*E))*(np.exp(-6.25*np.pi) - np.exp(-56.25*np.pi))
    return out

def AnalyticGradientSolution(A,E,x):
    out = (2/(A*E)*((-np.pi)*(x-2.5)*torch.exp(-np.pi*(x-2.5)**2))) + (4/(A*E)*((-np.pi)*(x-7.5)*torch.exp(-np.pi*(x-7.5)**2))) - (1/(10*A*E))*(np.exp(-6.25*np.pi) - np.exp(-56.25*np.pi))
    return out

def plot_everything():
    Coordinates = Coord_trajectories[-1]
    # Tests on trained data and compare to reference
    #plt.plot(InitialCoordinates,[coord*0 for coord in InitialCoordinates],'+k', markersize=2, label = 'Initial Nodes')
    plt.scatter(InitialCoordinates,[coord*0 for coord in InitialCoordinates], s=6, color="pink", alpha=0.5)

    plt.plot(Coordinates,[coord*0 for coord in Coordinates],'.k', markersize=2, label = 'Mesh Nodes')
    plt.plot(TrialCoordinates.data,AnalyticSolution(A,E,TrialCoordinates.data), label = 'Ground Truth')
    plt.plot(TrialCoordinates.data,MeshBeam(TrialCoordinates).data,'--', label = 'HiDeNN')
    plt.xlabel(r'$\underline{x}$ [m]')
    plt.ylabel(r'$\underline{u}\left(\underline{x}\right)$')
    plt.legend(loc="upper left")
    # plt.title('Displacement')
    plt.savefig('Results/Solution_displacement.pdf', transparent=True)  
    #plt.show()
    plt.clf()

    # Plots the gradient & compare to reference
    #plt.plot(InitialCoordinates,[coord*0 for coord in InitialCoordinates],'+k', markersize=2, label = 'Initial Nodes')
    plt.scatter(InitialCoordinates,[coord*0 for coord in InitialCoordinates], s=6, color="pink", alpha=0.5)
    plt.plot(Coordinates,[coord*0 for coord in Coordinates],'.k', markersize=2, label = 'Mesh Nodes')
    plt.plot(TrialCoordinates.data,AnalyticGradientSolution(A,E,TrialCoordinates.data), label = 'Ground Truth')
    plt.plot(TrialCoordinates.data,Derivative(MeshBeam(TrialCoordinates),TrialCoordinates).data,'--', label = 'HiDeNN')
    plt.xlabel(r'$\underline{x}$ [m]')
    plt.ylabel(r'$\frac{d\underline{u}}{dx}\left(\underline{x}\right)$')
    plt.legend(loc="upper left")
    # plt.title('Displacement first derivative')
    plt.savefig('Results/Solution_gradients.pdf', transparent=True)  
    #plt.show()
    plt.clf()

    #plt.plot(error)
    #plt.xlabel(r'epochs')
    #plt.ylabel(r'$J\left(\underline{u}\left(\underline{x}\right)\right)$')
    #plt.savefig('Results/Loss.pdf')  
    #plt.clf()
    #plt.show()

    #plt.plot(error[2500:])
    #plt.xlabel(r'epochs')
    #plt.ylabel(r'$J\left(\underline{u}\left(\underline{x}\right)\right)$')
    #plt.savefig('Results/Loss_zoomed.pdf')  
    #plt.show()
    #plt.clf()

    plt.plot(Coord_trajectories)
    plt.xlabel(r'epochs')
    plt.ylabel(r'$x_i\left(\underline{x}\right)$')
    plt.savefig('Results/Trajectories.pdf', transparent=True)  
    #plt.show()
    plt.clf()

    error3 = error-np.min(error)
    plt.semilogy(error2)

    plt.ylabel(r'$\Vert \underline{u}_{ex} - \underline{u}_{NN} \Vert^2$')
    ax2 = plt.gca().twinx()
    ax2.semilogy(error3, color='#F39C12', label="Training loss = "+ str(np.format_float_scientific(error[-1], precision=2)))
    ax2.set_ylabel(r'$J\left(\underline{u}\left(\underline{x}\right)\right)$')
    plt.legend()
    plt.savefig('Results/Loss_Comaprison.pdf')  
    plt.clf()

    plt.ylabel(r'Learning rate')
    plt.xlabel(r'epochs')
    plt.semilogy(Learning_Rate, color='#F50C12', label="Final learning rate = "+ str(np.format_float_scientific(Learning_Rate[-1], precision=2)))
    plt.legend()
    plt.savefig('Results/Learning rate.pdf')  
    plt.clf()

#%% Training loop

TrialCoordinates = torch.tensor([[i/50] for i in range(2,500)], dtype=torch.double, requires_grad=True)
InitialCoordinates = [MeshBeam.coordinates[i].data.item() for i in range(len(MeshBeam.coordinates))]
error = []
error2 = []
#CoordinatesGT = [0.0, 0.464534729719162, 0.909274160861969, 1.3622970581054688, 1.8098012208938599, 2.281587600708008, 2.719331741333008, 3.1904959678649902, 3.638345718383789, 4.090588569641113, 4.545549392700195, 4.996726036071777, 5.453673839569092, 5.906627655029297, 6.368518829345703, 6.811662197113037, 7.281474590301514, 7.718526840209961, 8.18874454498291, 8.635870933532715, 9.087241172790527, 9.535466194152832, 10.0]
Coord_trajectories = []
Learning_Rate = []

loss_old = 1.0e3
stagnancy_counter = 0
epoch = 0

while epoch<n_epochs and stagnancy_counter < 200:

    coord_old = [MeshBeam.coordinates[i].data.item() for i in range(len(MeshBeam.coordinates))]
    weights_old = copy.deepcopy(MeshBeam.InterpoLayer_uu.weight.data.detach())

    optimizer.zero_grad()
    u_predicted = MeshBeam(TrialCoordinates) 
    l = PotentialEnergyVectorised(A,E,u_predicted,TrialCoordinates,RHS(TrialCoordinates))

    #Jacobians = [MeshBeam.coordinates[i]-MeshBeam.coordinates[i-1] for i in range(1,len(MeshBeam.coordinates))]
    #Jacobians = torch.stack(Jacobians)
    #Ratio = torch.max(Jacobians)/torch.min(Jacobians)
    # Add the ratio to the loss
    #l+=MeshBeam.alpha*(Ratio-1)

    l.backward()

    # update weights
    optimizer.step()

    u_predicted = MeshBeam(TrialCoordinates) 
    l_new = PotentialEnergyVectorised(A,E,u_predicted,TrialCoordinates,RHS(TrialCoordinates))

    #print("current  = ", l.item())
    #print("next     = ", l_new.item())

    if l.item() < 0:
        limit = l.item()*(1-0.05)
    else:
        limit = (1+0.05)*l.item()

    if l_new.item() > limit:
        print(" revert ", l_new)
        for j in range(len(coord_old)):
            MeshBeam.coordinates[j].data = torch.Tensor([[coord_old[j]]])
        MeshBeam.InterpoLayer_uu.weight.data = torch.Tensor(weights_old)
        optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"]*0.5
        print(optimizer.param_groups[0]["lr"])
    else:
        Learning_Rate.append(optimizer.param_groups[0]["lr"])

        loss_current = l.item()
        loss_decrease = np.abs(loss_old - loss_current)
        loss_old = loss_current

        if loss_decrease < 1.0e-8:
            stagnancy_counter = stagnancy_counter +1
        else:
            stagnancy_counter = 0
    
        coord_new = [MeshBeam.coordinates[i].data.item() for i in range(len(MeshBeam.coordinates))]

        coord_dif = np.array([x - coord_new[i - 1] for i, x in enumerate(coord_new) if i > 0])
        if np.all(coord_dif > ((L/n_points)/10)) == False:
            for j in range(coord_dif.shape[0]):
                if coord_dif[j] < (L/n_points)/10:

                    MeshBeam.coordinates[j].data = torch.Tensor([[coord_old[j]]])
                    MeshBeam.coordinates[j+1].data = torch.Tensor([[coord_old[j+1]]])

        with torch.no_grad():
            error.append(l.item())
            Coordinates_i = [MeshBeam.coordinates[i].data.item() for i in range(len(MeshBeam.coordinates))]
            Coord_trajectories.append(Coordinates_i)
            if BoolCompareNorms:
                error2.append(MSE(AnalyticSolution(A,E,TrialCoordinates.data),u_predicted).data)

        if (epoch+1) % 200 == 0:
            print('epoch ', epoch+1, ' loss = ', l.item())
            plot_everything()

        epoch = epoch+1
    #print()