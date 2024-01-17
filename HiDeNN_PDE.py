
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg')
import numpy as np
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

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
        self.l1 = nn.Linear(1,1)
        self.l1.bias.data.fill_(1)
        self.l1.bias.requires_grad = False

    def forward(self,x, x_im1, x_i):
        # self.l1.weight.data.fill_(-1/(x_i-x_im1))
        with torch.no_grad():
            self.l1.weight.copy_(-1/(x_i-x_im1))
            self.l1.weight.requires_grad = False
        mid = self.relu(x)
        mid = self.l1(mid)
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
        self.l1 = nn.Linear(1,1)
        self.l1.bias.data.fill_(1)
        self.l1.bias.requires_grad = False

    def forward(self,x,x_ip1,x_i):
        with torch.no_grad():
            self.l1.weight.copy_(-1/(x_ip1-x_i))
            self.l1.weight.requires_grad = False
        mid = self.relu(x)
        mid = self.l1(mid)
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
        # Layer 1
        self.l1 = nn.Linear(1,2)
        # Set weight and bias for first layer and freeze them
        self.l1.weight.data.fill_(-1)
        self.l1.weight[1].data.fill_(1)
        self.l1.weight.requires_grad = False
        # Initialisation of the weights
        with torch.no_grad():
            self.l1.bias[0].copy_(x_i)
            self.l1.bias[1].copy_(-x_i) 
            self.l1.bias.requires_grad = r_adaptivity
        # Layer end
        self.l2 = nn.Linear(2,1)
        # Set weight and bias for end layer and freeze them
        self.l2.weight.data.fill_(1)
        self.l2.bias.data.fill_(-1)
        self.l2.bias.requires_grad = False
        self.l2.weight.requires_grad = False
        # Defines the right and left linear functions
        self.Linears = nn.ModuleList([LinearLeft(),LinearRight()])
    
    def forward(self, x, coordinates):
        """ The forward function takes as an input the coordonate x at which the NN is evaluated and the parameters' list coordinates where the nodes' corrdinates of the mesh are stored"""
        i = self.i
        # For the SF on the left
        if i == -1:
            x_i = coordinates[0]
            x_im1 = coordinates[0]-1/1000
            x_ip1 = coordinates[0+1]
        # For the SF on the right
        elif i == -2:
            x_i = coordinates[-1]
            x_im1 = coordinates[-2]
            x_ip1 = coordinates[-1]+1/1000
        else:
            x_i = coordinates[i]
            x_im1 = coordinates[i-1]
            x_ip1 = coordinates[i+1]
        with torch.no_grad():
            # self.l1.bias[0].copy_(x_i)
            self.l1.bias[1].copy_(-x_i) 
        # self.l1.bias.requires_grad = False
        l1 = self.l1(x)
        top = self.Linears[0](l1[:,0].view(-1,1),x_im1,x_i)
        bottom = self.Linears[1](l1[:,1].view(-1,1),x_ip1,x_i)
        l2 = torch.empty((bottom.shape[0],2),dtype=torch.float32)
        l2[:,0] = top[:,0]
        l2[:,1] = bottom[:,0]
        l3 = self.l2(l2)
        return l3

class MeshNN(nn.Module):
    """This is the main Neural Network building the FE interpolation, the coordinates parameters are trainable are correspond to the coordinates of the nodes in the Mesh which are passed as parameters to the sub NN where they are fixed. 
    Updating those parameters correspond to r-adaptativity
    The Interpolation layer weights correspond to the nodal values. Updating them is equivqlent to solving the PDE. """
    def __init__(self, np, L):
        super(MeshNN, self).__init__()
        self.coordinates = nn.ParameterList([nn.Parameter(torch.tensor(i)) for i in torch.linspace(0,L,np)])
        self.np = np 
        self.L = L 
        self.Functions = nn.ModuleList([Shapefunction(i,self.coordinates[i], r_adaptivity = True) for i in range(1,np-1)])
        self.InterpoLayer_uu = nn.Linear(self.np-2,1,bias=False)
        self.NodalValues_uu = nn.Parameter(data=torch.ones(np-2), requires_grad=False)
        self.InterpoLayer_uu.weight.data = self.NodalValues_uu
        self.Functions_dd = nn.ModuleList([Shapefunction(-1,self.coordinates[0], r_adaptivity = False),Shapefunction(-2,self.coordinates[-1], r_adaptivity = False)])
        self.InterpoLayer_dd = nn.Linear(2,1,bias=False)
        self.InterpoLayer_dd.weight.requires_grad = False
        self.SumLayer = nn.Linear(2,1,bias=False)
        self.SumLayer.weight.data.fill_(1)
        self.SumLayer.weight.requires_grad = False


    def forward(self,x):
        # Updates the coordinates
        for i in range(1,self.np-1):
            self.coordinates[i] = self.Functions[i-1].l1.bias[0]
        # Compute shape functions 
        intermediate_uu = [self.Functions[l](x,self.coordinates) for l in range(self.np-2)]
        intermediate_dd = [self.Functions_dd[l](x,self.coordinates) for l in range(2)]
        out_uu = torch.stack(intermediate_uu)
        out_dd = torch.stack(intermediate_dd)
        u_uu = self.InterpoLayer_uu(out_uu.T)
        u_dd = self.InterpoLayer_dd(out_dd.T)
        u = torch.empty((2,u_uu.shape[1]),dtype=torch.float32)
        u[0,:] = u_uu[0,:]
        u[1,:] = u_dd[0,:]       
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




#%% Application of the NN
# Geometry of the Mesh
L = 10                      # Length of the Beam
np = 30                     # Number of Nodes in the Mesh
A = 1                       # Section of the beam
E = 175                     # Young's Modulus (should be 175)
MeshBeam = MeshNN(np,L)     # Creates the associated model
# Boundary conditions
u_0 = 0                    #Left BC
u_L = 0                  #Right BC
MeshBeam.SetBCs(u_0,u_L)


#%% Debug Training samples (function Y to learn and its support X)
import numpy as np
X = torch.tensor([[i/10] for i in range(2,100)], dtype=torch.float32)
Y = torch.tensor([[np.cos(float(i))] for i in X], dtype=torch.float32)

#%% Define loss and optimizer
learning_rate = 0.001
n_epochs = 350
optimizer = torch.optim.Adam(MeshBeam.parameters(), lr=learning_rate)
loss = nn.MSELoss()

def RHS(x):
    b = -(4*np.pi**2*(x-2.5)**2-2*np.pi)/(torch.exp(np.pi*(x-2.5)**2)) - (8*np.pi**2*(x-7.5)**2-4*np.pi)/(torch.exp(np.pi*(x-7.5)**2))
    return  b

def PotentialEnergy(A,E,u,x,b):
    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    integral = 0
    for i in range(1,u.data.shape[0]):
        integral += (0.25*A*E*(x[i]-x[i-1])*(du_dx[i]**2+du_dx[i-1]**2)) - 0.5*((x[i]-x[i-1])*(u[i]*b[i]+u[i-1]*b[i-1]))
    return integral


def AnalyticSolution(A,E,x):
    out = (1/(A*E)*(torch.exp(-np.pi*(x-2.5)**2)-np.exp(-6.25*np.pi))) + (2/(A*E)*(torch.exp(-np.pi*(x-7.5)**2)-np.exp(-56.25*np.pi))) - (x/(10*A*E))*(np.exp(-6.25*np.pi) - np.exp(-56.25*np.pi))
    return out


#%% Training loop

TrialCoordinates = torch.tensor([[i/10] for i in range(2,100)], dtype=torch.float32, requires_grad=True)

error = []
for epoch in range(n_epochs):
    # predict = forward pass with our model
    u_predicted = MeshBeam(TrialCoordinates)  ######## regarder ce que ca donne ici pour une entrée vectorielle

    # loss
    l = PotentialEnergy(A,E,u_predicted,TrialCoordinates,RHS(TrialCoordinates))

    # calculate gradients = backward pass
    l.backward()
    
    # update weights
    optimizer.step()

    # zero the gradients after updating
    optimizer.zero_grad()
    with torch.no_grad():
        error.append(l.item())
    if (epoch+1) % 100 == 0:
        print('epoch ', epoch+1, ' loss = ', l.item())



#%% Post-processing

# Tests on trained data and compare to reference
# plt.plot(MeshBeam.coordinates.data,0*MeshBeam.coordinates.data,'.k', markersize=2, label = 'Mesh Nodes')
plt.plot(X,AnalyticSolution(A,E,X), label = 'Ground Truth')
plt.plot(X,MeshBeam(X).data,'--', label = 'HiDeNN')
plt.xlabel(r'$\underline{x}$ [m]')
plt.ylabel(r'$u\left(\underline{x}\right)$')
plt.legend(loc="upper left")
plt.title('On test coordinates')
plt.show()


# # Tests extrapolation on unseen coordinates 
X2 = torch.tensor([[(i-50)/10] for i in range(2,200)], dtype=torch.float32)
Y2 = torch.tensor([[np.cos(float(i))] for i in X2], dtype=torch.float32)
# plt.plot(MeshBeam.coordinates.data,0*MeshBeam.coordinates.data,'.k', markersize=2, label = 'Mesh Nodes')
plt.plot(X2,AnalyticSolution(A,E,X2), label = 'Ground Truth')
plt.plot(X2,MeshBeam(X2).data,'--', label = 'HiDeNN')
plt.xlabel(r'$\underline{x}$ [m]')
plt.ylabel(r'$f\left(\underline{x}\right)$')
plt.legend(loc="upper left")
plt.title('On new coordinates')
plt.show()





# %%
