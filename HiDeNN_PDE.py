import torch
import random
import torch.nn as nn
torch.set_default_dtype(torch.float64)
import Post.Plots as Pplot
import numpy as numpy

#%% Define the model for a 1D linear Beam mesh
    
class MultiplicationBlock(nn.Module):
    """This is an implementation of the multiplication block 
     See [Zhang et al. 2021] Multiplication block.
     Input parameters - the two tensors to be multiplied."""
    
    def __init__(self):
        super(MultiplicationBlock, self).__init__()
        self.relu = nn.ReLU()

        self.SumLayer = nn.Linear(2,1,bias=False)
        self.SumLayer.weight.data.fill_(1)
        self.SumLayer.weight.requires_grad = False

        self.Diff = nn.Linear(3,1,bias=False)
        self.Diff.weight.data = nn.Parameter(data=torch.tensor([[-0.5, 0.5,-0.5]]), requires_grad=False)

    def forward(self,x, y):
        mid = torch.cat((x,y),dim=1)
        mid = self.SumLayer(mid)
        mid = torch.pow(mid,2)
        mid = torch.cat((torch.pow(x,2),mid,torch.pow(y,2)),dim=1)
        mid = self.Diff(mid)
        return mid


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

class ShapefunctionQuadratic(nn.Module):
    """This is an implementation of quadratic 1D shape function in a DNN representation
    See [Zhang et al. 2021] consisting of a linear(1,2) function leading the the Linear{Left+Right} layers which output then passes through Linear(2,1). 
    The input parameters are:
        - the coordinate x where the function is evaluated
        - the index i of the node associated to the shape function """
    
    def __init__(self, i):
        super(ShapefunctionQuadratic, self).__init__()
        # Index of the node associated to the shape function
        self.i = i
        # Defines the linear block function
        self.LinearBlock = LinearBlock()
        self.Multiplication = MultiplicationBlock()
        self.relu = nn.ReLU()

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
            x_ip1 = coordinates[1]
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


        left = self.LinearBlock(x, self.relu(x_i), self.relu(x_ip1), 0, x_ip1 - x_i)
        right = self.LinearBlock(x, x_i, self.relu(x_ip1), self.relu(x_ip1) - x_i, 0)
        
        prod = -self.Multiplication(left,right)/(((x_i + x_ip1)*0.5 - x_i)*((x_i + x_ip1)*0.5 - x_ip1))
        #prod = -(left*right)/(((x_i + x_ip1)*0.5 - x_i)*((x_i + x_ip1)*0.5 - x_ip1))
        return prod


class ElementBlock(nn.Module):

    def __init__(self, i ):
        super(ElementBlock, self).__init__()
        self.i = i
        self.LinearBlock = LinearBlock()

    def forward(self, x, coordinates):
        i = self.i

        # for the outter Shape functions the index acts as a tag, -1 for the left, -2 for the right
        if i == -1:     # Left
            x_left = coordinates[0]-coordinates[-1]*1/100
            x_right = coordinates[0]
        elif i == -2:   # Right
            x_left = coordinates[-1]
            x_right = coordinates[-1]*(1+1/100)
        else:
            x_left = coordinates[i]
            x_right = coordinates[i+1]

        left = self.LinearBlock(x, x_left, x_right, 0, 1)
        right = self.LinearBlock(x, x_left, x_right, 1, 0)

        out = torch.cat((left, right),dim=1)

        return out

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

        left = self.LinearBlock(x, x_im1, x_i, 0, 1)
        right = self.LinearBlock(x, x_i, x_ip1, 1, 0)
        l3 = left + right - 1

        return l3

class MeshNN(nn.Module):
    """This is the main Neural Network building the FE interpolation, the coordinates parameters are trainable are correspond to the coordinates of the nodes in the Mesh which are passed as parameters to the sub NN where they are fixed. 
    Updating those parameters correspond to r-adaptativity
    The Interpolation layer weights correspond to the nodal values. Updating them is equivqlent to solving the PDE. """
    def __init__(self, n_elem, L, connectivity_matrix, alpha = 0.005):
        super(MeshNN, self).__init__()
        self.alpha = alpha # set the weight for the Mesh regularisation 

        self.elements = torch.arange(n_elem)
        self.n_elem = n_elem
        self.np = n_elem+1 
        self.connectivity_matrix = connectivity_matrix
        self.coordinates = nn.ParameterList([nn.Parameter(torch.tensor([[i]])) for i in torch.linspace(0,L,self.np)])
        
        self.L = L 
        self.Functions = nn.ModuleList([ElementBlock(i) for i in range(self.n_elem)])
        
        self.CompositionLayer = nn.Linear(2*(self.n_elem+2),self.np,bias=False)
        self.CompositionLayer.weight.data = self.connectivity_matrix

        self.InterpoLayer_uu = nn.Linear(self.np-2,1,bias=False)
        self.NodalValues_uu = nn.Parameter(data=0.1*torch.ones(self.np-2), requires_grad=False)
        self.InterpoLayer_uu.weight.data = self.NodalValues_uu

        self.Functions_dd = nn.ModuleList([ElementBlock(-1),ElementBlock(-2)])

        self.InterpoLayer_dd = nn.Linear(1,1,bias=False)
        self.InterpoLayer_dd.weight.requires_grad = False

        self.SumLayer = nn.Linear(2,1,bias=False)
        self.SumLayer.weight.data.fill_(1)
        self.SumLayer.weight.requires_grad = False


    def forward(self,x):
        # Compute shape functions 
        intermediate_uu = [self.Functions[l](x,self.coordinates) for l in range(self.n_elem)]
        intermediate_dd = [self.Functions_dd[l](x,self.coordinates) for l in range(2)]

        out_uu = torch.cat(intermediate_uu, dim=1)
        out_dd = torch.cat(intermediate_dd, dim=1)
        joined_vector = torch.cat((out_dd,out_uu),dim=1)
        #print("joined_vector = ", joined_vector.shape)
        recomposed_vector = self.CompositionLayer(joined_vector) - 1
        #print("recomposed vector = ", recomposed_vector.shape)

        recomposed = recomposed_vector[:,1:-1]
        recomposed_BC = torch.stack((recomposed_vector[:,0],recomposed_vector[:,-1]),dim=1)

        u_inter = self.InterpoLayer_uu(recomposed)
        u_BC = self.InterpoLayer_dd(recomposed_BC)

        u = torch.stack((u_inter,u_BC), dim=1)
        return self.SumLayer(u), recomposed_vector

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
n_elem = 22                          # Number of Nodes in the Mesh
A = 1                            # Section of the beam
E = 175                          # Young's Modulus (should be 175)
alpha =0.0                   # Weight for the Mesh regularisation 

connectivity_matrix = torch.zeros((n_elem+1,2*(n_elem+2)))

# node 0
connectivity_matrix[0,0] = 1.0
connectivity_matrix[0,5] = 1.0

# node n
connectivity_matrix[n_elem,3] = 1.0     
connectivity_matrix[n_elem,-2] = 1.0     


for node in range(1,n_elem):
    row = node
    left = node-1

    cols = [4+left*2, 4+left*2+3]
    print("node = ", node)
    print("cols = ", cols)
    for col in cols:
        connectivity_matrix[row,col] = 1


print("connectivity matrix = ", connectivity_matrix.data)
print()

MeshBeam = MeshNN(n_elem,L,connectivity_matrix,alpha)    # Creates the associated model
# Boundary conditions
u_0 = 0                     #Left BC
u_L = 0                     #Right BC
MeshBeam.SetBCs(u_0,u_L)
# Import mechanical functions


from Bin.Training import Test_GenerateShapeFunctions, Training_InitialStage


# Set the coordinates as trainable
MeshBeam.UnFreeze_Mesh()
# Set the coordinates as untrainable
# MeshBeam.Freeze_Mesh()
# Set the require output requirements
BoolPlot = False             # Bool for plots used for gif
BoolCompareNorms = True      # Bool for comparing energy norm to L2 norm


#%% Define loss and optimizer
learning_rate = 1.0e-3
n_epochs = 25000
optimizer = torch.optim.Adam(MeshBeam.parameters(), lr=learning_rate)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, cooldown=100, factor=0.9)

MSE = nn.MSELoss()


TrialCoordinates = torch.tensor([[i/50] for i in range(-50,550)], 
                                dtype=torch.float64, requires_grad=True)

Test_GenerateShapeFunctions(MeshBeam, TrialCoordinates)

'''

Training_InitialStage(MeshBeam, A, E, L, n_elem, TrialCoordinates, optimizer, n_epochs, BoolCompareNorms, MSE)
'''

'''
optim = torch.optim.LBFGS(MeshBeam.parameters(),
                    #history_size=5, 
                    #max_iter=15, 
                    #tolerance_grad = 1.0e-9,
                    line_search_fn="strong_wolfe")


print("Minimal error = ", numpy.min(error))

epoch = 0
stagnancy_counter = 0

while epoch<n_epochs and stagnancy_counter < 3:

    def closure():
        optim.zero_grad()
        u_predicted = MeshBeam(TrialCoordinates) 
        l = PotentialEnergyVectorised(A,E,u_predicted,TrialCoordinates,RHS(TrialCoordinates))
        l.backward()
        return l

    optim.step(closure)
    l = closure()

    with torch.no_grad():
        # Stores the loss
        error.append(l.item())
        # Stores the coordinates trajectories
        Coordinates_i = [MeshBeam.coordinates[i].data.item() for i in range(len(MeshBeam.coordinates))]
        Coord_trajectories.append(Coordinates_i)
        Learning_Rate.append(optimizer.param_groups[0]["lr"])

        if BoolCompareNorms:
            # Copute and store the L2 error w.r.t. the analytical solution
            error2.append(MSE(AnalyticSolution(A,E,TrialCoordinates.data),u_predicted).data)

    loss_current = l.item()
    loss_decrease = (loss_old - loss_current)/numpy.abs(loss_old)

    if loss_decrease >= 0 and loss_decrease < 1.0e-8:
        stagnancy_counter = stagnancy_counter +1
    else:
        stagnancy_counter = 0

    if loss_decrease > 0:
        loss_old = loss_current

    if (epoch+1) % 1 == 0:
        print('epoch ', epoch+1, ' loss = ', numpy.format_float_scientific( l.item(), precision=4))

    epoch = epoch+1

plot_everything()
'''