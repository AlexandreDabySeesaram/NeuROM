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
        
        mid = self.relu(-x + x_b.T)
        mid = self.relu(1 - mid/(x_b.T-x_a.T))
        mid = (y_b-y_a)*mid + y_a
        return mid

class ElementBlock_Bar_Lin(nn.Module):
    """Bar 2 (linear 1D) element block
    Returns the N_i(x)'s for each nodes within the element"""
    def __init__(self, connectivity):
        super(ElementBlock_Bar_Lin, self).__init__()
        self.LinearBlock = LinearBlock()
        self. connectivity = connectivity.astype(int)



    def forward(self, x, coordinates, i):
        # i = self.i
        # For the outter Nodes, phantom elements are created 
        # to cancel out shape functions beyond the geometry of the structure in question, 
        # to prevent any form of extrapolation beyond its boundaries 

        if -1  in i:
            x_left_0 = coordinates[0]-coordinates[1]/100
            x_right_0 = coordinates[0]  
            x_left_2 = coordinates[1]
            x_right_2 = coordinates[1]*(1+1/100)
            x_left = [x_left_0,x_left_2]
            x_right = [x_right_0,x_right_2]
        else:
            x_left = [coordinates[row-1] for row in self.connectivity[i,0]]
            x_right = [coordinates[row-1] for row in self.connectivity[i,-1]]

        left = self.LinearBlock(x, torch.cat(x_left), torch.cat(x_right), torch.tensor([0]), torch.tensor([1]))
        right = self.LinearBlock(x, torch.cat(x_left), torch.cat(x_right), torch.tensor([1]), torch.tensor([0]))
        out = torch.stack((left, right),dim=2).view(right.shape[0],-1) # Katka's left right implementation {[N2 N1] [N3 N2] [N4 N3]}

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
        # self.Functions = nn.ModuleList([ElementBlock_Bar_Lin(i,mesh.Connectivity) \
                                        # for i in range(self.NElem)])
        self.ElementBlock = ElementBlock_Bar_Lin(mesh.Connectivity)
        self.InterpoLayer_uu = nn.Linear(self.dofs-self.NBCs,1,bias=False)
        self.NodalValues_uu = nn.Parameter(data=0.1*torch.ones(self.dofs-self.NBCs), requires_grad=False)
        self.InterpoLayer_uu.weight.data = self.NodalValues_uu
        # self.Functions_dd = nn.ModuleList([ElementBlock_Bar_Lin(-1,mesh.Connectivity),
        #                                    ElementBlock_Bar_Lin(-2,mesh.Connectivity)])
        self.AssemblyLayer = nn.Linear(2*(self.NElem+2),self.dofs,bias=False)
        # self.AssemblyLayer.weight.data = torch.tensor(mesh.weights_assembly_total)
        self.AssemblyLayer.weight.data = torch.tensor(mesh.weights_assembly_total,dtype=torch.float32).detach()
        self.AssemblyLayer.weight. requires_grad=False

        self.InterpoLayer_dd = nn.Linear(2,1,bias=False)
        # self.InterpoLayer_dd.weight.requires_grad = False
        self.SumLayer = nn.Linear(2,1,bias=False)
        self.SumLayer.weight.data.fill_(1)
        self.SumLayer.weight.requires_grad = False
        self.ElemList = torch.arange(self.NElem)



    def forward(self,x):
        # Compute shape functions 
        out_uu = self.ElementBlock(x,self.coordinates,self.ElemList)
        out_dd = self.ElementBlock(x,self.coordinates,torch.tensor(-1))
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
        import numpy as np
        super(InterpPara, self).__init__()
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.N_mu = N_mu.int()
        self.n_elem = self.N_mu-1 # linear 1D discretisation of the parametric field
        # Parametric mesh coordinates
        self.coordinates = nn.ParameterList([nn.Parameter(torch.tensor([[i]])) \
                                             for i in torch.linspace(self.mu_min,self.mu_max,self.N_mu)])    

        ### Assembly layer 
        self.AssemblyLayer = nn.Linear(2*(self.n_elem),self.N_mu)
        weights_assembly = torch.zeros((self.N_mu,2*self.n_elem))
        NodeList = np.linspace(1,self.N_mu,self.N_mu)
        self.Connectivity = np.stack((NodeList[:-1],NodeList[1:]),axis=-1) # The nodes are sorted in the ascending order (from 1 to n_elem-1)
        elem_range = np.arange(self.Connectivity.shape[0])
        ne_values = np.arange(2) # 2 nodes per linear 1D element
        ne_values_j = np.array([1,0]) # Katka's left right implementation {[N2 N1] [N3 N2] [N4 N3]} otherwise same as ne_value
        i_values = self.Connectivity[:, ne_values]-1 
        j_values = 2 * (elem_range[:, np.newaxis])+ ne_values_j 
        weights_assembly[i_values.flatten().astype(int), j_values.flatten().astype(int)] = 1
        self.weights_assembly = weights_assembly
        self.AssemblyLayer.weight.data = self.weights_assembly
        self.AssemblyLayer.weight.requires_grad=False
        self.AssemblyLayer.bias.requires_grad=False
        self.AssemblyLayer.bias.fill_(-1)
        self.AssemblyLayer.bias[0] = torch.tensor(0)
        self.AssemblyLayer.bias[-1] = torch.tensor(0)

        self.ElementBlock = ElementBlock_Bar_Lin(self.Connectivity)

        # self.Functions = nn.ModuleList([ElementBlock_Bar_Lin(i,self.Connectivity) for i in range(self.n_elem)])

        # Interpolation (nodal values) layer
        # self.NodalValues_para = nn.Parameter(data=torch.linspace(self.mu_min,self.mu_max,self.N_mu).pow(-1), requires_grad=False)
        self.NodalValues_para = nn.Parameter(data=torch.ones(self.N_mu), requires_grad=False)  
        self.InterpoLayer = nn.Linear(self.N_mu,1,bias=False)
        # Initialise with linear mode
        self.InterpoLayer.weight.data = self.NodalValues_para
        self.ElemList = torch.arange(self.n_elem)

    def forward(self,mu):
        # out_elements = [self.ElementBlock(mu,self.coordinates,i) for i in range(self.n_elem)]
        out_elements = self.ElementBlock(mu,self.coordinates,self.ElemList)        # intermediate = [self.Functions[l](mu,self.coordinates) for l in range(self.n_elem)]
        Assembled_vector = self.AssemblyLayer(out_elements)
        out_interpolation = self.InterpoLayer(Assembled_vector)
        return out_interpolation
    
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
        self.InterpoLayer.weight.requires_grad = True
            
    def Freeze_FEM(self):
        """Set the nodale values as untrainable parameters """
        self.InterpoLayer.weight.requires_grad = False

class NeuROM(nn.Module):
    """This class builds the Reduced-order model from the interpolation NN for space and parameters space"""
    def __init__(self, mesh, BCs, n_modes, ParametersList):
        super(NeuROM, self).__init__()
        self.n_modes = n_modes
        self.n_para = len(ParametersList)
        # self.mu_min = mu_min
        # self.mu_max = mu_max
        # self.N_mu = N_mu
        self.Space_modes = nn.ModuleList([MeshNN(mesh) for i in range(self.n_modes)])
        self.Para_Nets = nn.ModuleList([InterpPara(Para[0], Para[1], Para[2]) for Para in ParametersList])

        self.Para_modes = nn.ModuleList([self.Para_Nets for i in range(self.n_modes)])
        # Set BCs 
        # First modes get the Boundary conditions
        self.Space_modes[0].SetBCs(BCs[0],BCs[1])
        # Following modes are homogeneous (admissible to 0)
        for i in range(1,self.n_modes):
            self.Space_modes[i].SetBCs(0,0)

    def Freeze_Mesh(self):
        """Set the space coordinates as untrainable parameters"""
        for i in range(self.n_modes):
            self.Space_modes[i].Freeze_Mesh()

    def UnFreeze_Mesh(self):
        """Set the space coordinates as trainable parameters"""
        for i in range(self.n_modes):
            self.Space_modes[i].UnFreeze_Mesh()
    def Freeze_Space(self):
        """Set the spatial modes as untrainable """
        for i in range(self.n_modes):
            self.Space_modes[i].Freeze_FEM()   

    def UnFreeze_Space(self):
        """Set the spatial modes as trainable """
        for i in range(self.n_modes):
            self.Space_modes[i].UnFreeze_FEM()   

    def Freeze_MeshPara(self):
        """Set the para coordinates as untrainable parameters"""
        for i in range(self.n_modes):
            for j in range(self.n_para):
                self.Para_modes[i][j].Freeze_Mesh()

    def UnFreeze_MeshPara(self):
        """Set the para coordinates as trainable parameters"""
        for i in range(self.n_modes):
            self.Para_modes[i].UnFreeze_Mesh()

    def Freeze_Para(self):
        """Set the para modes as untrainable """
        for i in range(self.n_modes):
            self.Para_modes[i].Freeze_FEM()  

    def UnFreeze_Para(self):
        """Set the para modes as trainable """
        for i in range(self.n_modes):
            self.Para_modes[i].UnFreeze_FEM()  

    def forward(self,x,mu):
        Space_modes = [self.Space_modes[l](x) for l in range(self.n_modes)]
        Space_modes = torch.cat(Space_modes,dim=1)
        dimensions = mu.shape
        TensorParameters = torch.zeros(self.n_modes,)
        for mode in range(self.n_modes):
            Para_mode_List = [self.Para_modes[mode][l](mu[:,0].view(-1,1))[:,None] for l in range(self.n_para)]
            if mode == 0:
                Para_modes = torch.unsqueeze(torch.cat(Para_mode_List,dim=1), dim=0)
                # Para_modes = torch.unsqueeze(Para_modes, dim=0)
            else:
                New_mode = torch.unsqueeze(torch.cat(Para_mode_List,dim=1), dim=0)
                Para_modes = torch.vstack((Para_modes,New_mode))
        # Para_modes = torch.cat(Para_modes,dim=1)
        out = torch.matmul(Space_modes,Para_modes.view(self.n_modes,Para_modes.shape[1]))
        return out

        





        