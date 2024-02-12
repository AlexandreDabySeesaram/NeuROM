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
        # to cancel out shape functions beyond the geometry of the structure in question, 
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
        out = torch.cat((left, right),dim=1) # Katka's left right implementation {[N2 N1] [N3 N2] [N4 N3]}

        # out = torch.cat((right, left),dim=1) #  {[N1 N2] [N2 N3] [N3 N4]}

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
        # self.InterpoLayer_dd.weight.requires_grad = False
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
        plt.plot(x.data,joined_vector.data[:,0], label = '$N_1$')
        plt.plot(x.data,joined_vector.data[:,1], label = '$N_2$')
        plt.plot(x.data,joined_vector.data[:,2], label = '$N_3$')
        plt.legend(loc="upper left")
        plt.savefig('Results/Sep_SF_K.pdf', transparent=True)
        plt.clf()        
        recomposed_vector_u = self.AssemblyLayer(joined_vector) -1
        u_u = self.InterpoLayer_uu(recomposed_vector_u[:,2:])
        u_d = self.InterpoLayer_dd(recomposed_vector_u[:,:2])
        u = torch.stack((u_u,u_d), dim=1)
        plt.plot(x.data,recomposed_vector_u.data[:,0], label = '$N_1$')
        plt.plot(x.data,recomposed_vector_u.data[:,1], label = '$N_2$')
        plt.plot(x.data,recomposed_vector_u.data[:,2], label = '$N_3$')
        # plt.plot(x.data,recomposed_vector_u.data[:,3], label = '$N_4$')
        # plt.plot(x.data,recomposed_vector_u.data[:,-2], label = '$N_{p-3}$')
        plt.legend(loc="upper left")
        plt.savefig('Results/Assembled_SF_K.pdf', transparent=True)
        plt.clf()

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
        self.coordinates[1].requires_grad = False

    def UnFreeze_FEM(self):
        """Set the nodale values as trainable parameters """
        self.InterpoLayer_uu.weight.requires_grad = True
            
    def Freeze_FEM(self):
        """Set the nodale values as untrainable parameters """
        self.InterpoLayer_uu.weight.requires_grad = False

class InterpPara(nn.Module):
    """This class act as the 1D mesh in the parametric space and therefore output a parameter mode in the Tensor Decomposition (TD) sens """
    def __init__(self, mu_min, mu_max,N_mu):
        super(InterpPara, self).__init__()
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.N_mu = N_mu
        self.coordinates = nn.ParameterList([nn.Parameter(torch.tensor([[i]])) \
                                             for i in torch.linspace(self.mu_min,self.mu_max,self.N_mu)])    
        n_elem = len(self.coordinates)-1 # linear 1D discretisation of the parametric field
        Assembly_matrix = torch.zeros((n_elem,2*n_elem))

