#%% Libraries import
import time 
# Import pre-processing functions
from .src import Pre_processing as pre
# Import mechanical functions
from .src.PDE_Library import RHS, PotentialEnergyVectorised, \
        Derivative, AnalyticGradientSolution, AnalyticSolution, GramSchmidt
# Import torch librairies
import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)
#Import post processing libraries
from .Post import Plots as Pplot
import matplotlib.pyplot as plt
import numpy as np



import torch.jit as jit

#%% Define the model for a 1D linear Beam mesh
class LinearBlock(nn.Module):
    """This is the new implementation of the linear block 
     See [Zhang et al. 2021] Linear block. 
     Args:
        x (Tensor): Cordinate where the function is evaluated
        x_b (Tensor): If used for left part: x_b = x_i else if used right part x_b = x_ip1
        x_a (Tensor): If used for left part: x_a = x_im1 else if used right part x_a = x_i  
        y_a (Tensor): The value of the linear function at x=x_a
        y_b (Tensor): The value of the linear function at x=x_b
    Returns: 
        mid (Tensor): Linear funciton between x_a and x_b from y_a to y_b
    """
    def __init__(self):
        super(LinearBlock, self).__init__()
        self.relu = nn.ReLU()

    def forward(self,x, x_a, x_b, y_a, y_b):
        
        mid = self.relu(-x + x_b.T)
        mid = self.relu(1 - mid/(x_b.T-x_a.T))
        
        if y_a.dim() ==2:
            mid = mid*((y_b-y_a).T)  + y_a.T
        elif y_b.dim() ==2:
            mid = mid*((y_b.T)-y_a)  + y_a
        else:
            mid = mid*((y_b-y_a))  + y_a
        return mid

class ElementBlock_Bar_Quadr(nn.Module):
    """Bar 3 (quadratic 1D) element block
    Returns the N_i(x)'s for each nodes within the element"""
    def __init__(self, connectivity):
        super(ElementBlock_Bar_Quadr, self).__init__()
        self.LinearBlock = LinearBlock()
        self.connectivity = connectivity.astype(int)
        self.register_buffer('zero', torch.tensor([0]))

    def forward(self, x, coordinates, i):
        # i = self.i
        # For the outter Nodes, phantom elements are created 
        # to cancel out shape functions beyond the geometry of the structure in question, 
        # to prevent any form of extrapolation beyond its boundaries 

        if -1  in i:
            x_left_0    = coordinates[0]-coordinates[1]/100
            x_right_0   = coordinates[0]  
            x_left_2    = coordinates[1]
            x_right_2   = coordinates[1]*(1+1/100)
            x_left      = torch.cat([x_left_0,x_left_2])
            x_right     = torch.cat([x_right_0,x_right_2])
        else:
            x_left      = torch.cat([coordinates[row-1] for row in self.connectivity[i,0]])
            x_right     = torch.cat([coordinates[row-1] for row in self.connectivity[i,-2]])
            x_mid       = torch.cat([coordinates[row-1] for row in self.connectivity[i,-1]])
        sh_mid_1    = self.LinearBlock(x, x_left, x_right, self.zero, x_right - x_left)
        sh_mid_2    = self.LinearBlock(x, x_left, x_right, x_right - x_left, self.zero)    
        sh_mid      = -(sh_mid_1*sh_mid_2)/((x_mid -x_left)*(x_mid - x_right)).T

        sh_R_1      = self.LinearBlock(x, x_left, x_right, x_mid - x_left, x_mid - x_right)
        sh_R_2      = self.LinearBlock(x, x_left, x_right, x_right - x_left,  self.zero) 
        sh_R        = (sh_R_1*sh_R_2)/((x_left-x_mid)*(x_left - x_right)).T

        sh_L_1      = self.LinearBlock(x, x_left, x_right,  self.zero, x_right - x_left)
        sh_L_2      = self.LinearBlock(x, x_left, x_right, x_left - x_mid, x_right - x_mid)
        sh_L        = (sh_L_1*sh_L_2)/((x_right-x_left)*(x_right - x_mid)).T

        out = torch.stack((sh_L, sh_R, sh_mid),dim=2).view(sh_R.shape[0],-1) # Left | Right | Middle

        return out



class ElementBlock_Bar_Lin(nn.Module):
    """This is an implementation of the Bar 2 (linear 1D) element
    Args:
        x (Tensor): Cordinate where the function is evaluated
        coordinates (Parameters List): List of coordinates of nodes of the 1D mesh
        i (Integer): The indexes of the element for which an output is expected

    Returns:
         N_i(x)'s for each nodes within each element"""
    def __init__(self, connectivity):
        """ Initialise the Linear Bar element 
        Args:
            connectivity (Interger table): Connectivity matrix of the 1D mesh
        """
        super(ElementBlock_Bar_Lin, self).__init__()
        self.LinearBlock = LinearBlock()
        self.connectivity = connectivity.astype(int)
        self.register_buffer('zero', torch.tensor([0], dtype = torch.float32))
        self.register_buffer('one', torch.tensor([1], dtype = torch.float32))
        
    def forward(self, x, coordinates, i):
        """ This is the forward function of the Linear element block. Note that to prevent extrapolation outside of the structure's geometry, 
        phantom elements are used to cancel out the interpolation shape functions outside of the beam.
        Those phantom elements are flagged with index -1
        """
        try:
            if -1  in i:
                x_left_0    = coordinates[0]-coordinates[1]/100
                x_right_0   = coordinates[0]  
                x_left_2    = coordinates[1]
                x_right_2   = coordinates[1]*(1+1/100)
                x_left      = [x_left_0,x_left_2]
                x_right     = [x_right_0,x_right_2]
            else:
                x_left      = [coordinates[row-1] for row in self.connectivity[i,0]]
                x_right     = [coordinates[row-1] for row in self.connectivity[i,-1]]

            left    = self.LinearBlock(x, torch.cat(x_left), torch.cat(x_right), self.zero, self.one)
            right   = self.LinearBlock(x, torch.cat(x_left), torch.cat(x_right), self.one, self.zero)
            out     = torch.stack((left, right),dim=2).view(right.shape[0],-1) # Katka's left right implementation {[N2 N1] [N3 N2] [N4 N3]}
        except:
            self.register_buffer('zero', torch.tensor([0], dtype = torch.float32))
            self.register_buffer('one', torch.tensor([1], dtype = torch.float32))
            if -1  in i:
                x_left_0    = coordinates[0]-coordinates[1]/100
                x_right_0   = coordinates[0]  
                x_left_2    = coordinates[1]
                x_right_2   = coordinates[1]*(1+1/100)
                x_left      = [x_left_0,x_left_2]
                x_right     = [x_right_0,x_right_2]
            else:
                x_left = [coordinates[row-1] for row in self.connectivity[i,0]]
                x_right = [coordinates[row-1] for row in self.connectivity[i,-1]]

            left = self.LinearBlock(x, torch.cat(x_left), torch.cat(x_right), self.zero, self.one)
            right = self.LinearBlock(x, torch.cat(x_left), torch.cat(x_right), self.one, self.zero)
            out = torch.stack((left, right),dim=2).view(right.shape[0],-1) # Katka's left right implementation {[N2 N1] [N3 N2] [N4 N3]}
        return out


class MeshNN(nn.Module):
    """ This class is a space HiDeNN building a Finite Element (FE) interpolation over the space domain. 
    The coordinates of the nodes of the underlying mesh are trainable. Those coordinates are passed as a List of Parameters to the subsequent sub-neural networks
    Updating those parameters correspond to r-adaptativity
    The Interpolation layer weights correspond to the nodal values. Updating them 
    is equivqlent to solving the PDE. """
    def __init__(self, mesh):
        super(MeshNN, self).__init__()
        self.register_buffer('float_config',torch.tensor([0.0])  )                                                     # Keep track of device and dtype used throughout the model
        self.version    = "Trapezoidal"
        self.coordinates = nn.ParameterList([nn.Parameter(torch.tensor([[mesh.Nodes[i][1]]])) \
                                             for i in range(len(mesh.Nodes))])
        self.dofs       = mesh.NNodes*mesh.dim # Number of Dofs
        self.NElem      = mesh.NElem
        self.connectivity = mesh.Connectivity

        if mesh.NoBC==False:
            self.NBCs = len(mesh.ListOfDirichletsBCsIds) # Number of prescribed Dofs
        else:
            self.NBCs = 0

        if mesh.order =='1':
            self.ElementBlock = ElementBlock_Bar_Lin(mesh.Connectivity)
        elif mesh.order =='2':
            self.ElementBlock = ElementBlock_Bar_Quadr(mesh.Connectivity)


        # Phantom elements always use LinearBlock
        self.ElementBlock_BC = ElementBlock_Bar_Lin(mesh.Connectivity)
        self.InterpoLayer_uu = nn.Linear(self.dofs-self.NBCs,1,bias=False)
        # self.NodalValues_uu = nn.Parameter(data=0.1*torch.ones(self.dofs-self.NBCs), requires_grad=False)
        #TODO: trace why mesh.borders_nodes is not populated in 1D NeuROM
        # self.InterpoLayer_uu = nn.Linear(self.dofs-len(mesh.borders_nodes),1,bias=False)

        # Changed to border_nodes instead of NBCs for mixed formulation without BCs (TODO: confirm with NeuROM)
        self.NodalValues_uu = nn.Parameter(data=0.1*torch.ones(self.dofs-len(mesh.borders_nodes)), requires_grad=False)
        self.InterpoLayer_uu.weight.data = self.NodalValues_uu
        # self.InterpoLayer_uu.weight.data = self.NodalValues_uu*torch.randn_like(self.NodalValues_uu)
 
        self.AssemblyLayer = nn.Linear(2*(self.NElem+2),self.dofs)
        self.AssemblyLayer.weight.data = torch.tensor(mesh.weights_assembly_total,dtype=torch.float64).clone().detach()
        self.AssemblyLayer.weight.requires_grad=False
        # self.AssemblyLayer.bias.data =  torch.tensor(mesh.assembly_vector,dtype=torch.float32).clone().detach()
        self.AssemblyLayer.bias.data =  mesh.assembly_vector.clone().detach() # Remove warning, assembly_vector is already a tensor
        self.AssemblyLayer.bias.requires_grad=False

        self.InterpoLayer_dd = nn.Linear(2,1,bias=False)
        #self.InterpoLayer_dd.weight.data = 0.1*torch.ones(2)
        self.InterpoLayer_dd.weight.data = 0.0001*torch.randint(low=-100, high=100, size=(2,))

        self.SumLayer = nn.Linear(2,1,bias=False)
        self.SumLayer.weight.data.fill_(1)
        self.SumLayer.weight.requires_grad = False
        self.ElemList = torch.arange(self.NElem)

        if self.NBCs>0:
            self.SetBCs(mesh.ListOfDirichletsBCsValues)

    def forward(self,x):
        # Compute shape functions 
        out_uu = self.ElementBlock(x,self.coordinates,self.ElemList)
        out_dd = self.ElementBlock_BC(x,self.coordinates,torch.tensor(-1))

        joined_vector = torch.cat((out_uu,out_dd),dim=1)      
        recomposed_vector_u = self.AssemblyLayer(joined_vector) #-1
        u_u = self.InterpoLayer_uu(recomposed_vector_u[:,2:])
        u_d = self.InterpoLayer_dd(recomposed_vector_u[:,:2])

        u = torch.stack((u_u,u_d), dim=1)
 
        return self.SumLayer(u)
    
    def Init_from_previous(self,previous_model):
        newcoordinates = [coord for coord in self.coordinates]
        try:
             previous_model.float_config.dtype
        except:
            previous_model.float_config = torch.tensor([0],dtype = torch.float64)
        newcoordinates = torch.cat(newcoordinates,dim=0).to(previous_model.float_config.dtype)
        NewNodalValues = previous_model(newcoordinates).to(self.float_config.dtype)
        self.InterpoLayer_uu.weight.data = NewNodalValues[2:,0]

    def ZeroOut(self):
        """ This functions cancels out every nodal values associated with the MeshNN interpolation"""
        self.InterpoLayer_uu.weight.data = 0*self.NodalValues_uu


    def SetBCs(self,u_d):
        """Set the two Dirichlet boundary conditions
        Args:
            u_d (Float list): The left and right BCs"""
        self.register_buffer('u_0', torch.tensor(u_d[0], dtype=torch.float64))
        self.register_buffer('u_L', torch.tensor(u_d[1], dtype=torch.float64))


        self.InterpoLayer_dd.weight.data = torch.tensor([self.u_0,self.u_L], requires_grad=False)
        self.InterpoLayer_dd.weight.requires_grad = False

    def Freeze_Mesh(self):
        """Set the coordinates as untrainable parameters"""
        for param in self.coordinates:
            param.requires_grad = False

    def UnFreeze_Mesh(self):
        """Set the coordinates as trainable parameters """
        self.original_coordinates = [self.coordinates[i].data.item() for i in range(len(self.coordinates))]

        for param in self.coordinates:
            param.requires_grad = True
        #Freeze external coorinates to keep geometry    
        self.coordinates[0].requires_grad = False
        self.coordinates[1].requires_grad = False

    def UnFreeze_FEM(self):
        """Set the nodale values as trainable parameters """
        self.InterpoLayer_uu.weight.requires_grad = True

    def UnFreeze_BC(self):
        """Set the nodale values as trainable parameters """
        self.InterpoLayer_dd.weight.requires_grad = True

    def Freeze_FEM(self):
        """Set the nodale values as untrainable parameters """
        self.InterpoLayer_uu.weight.requires_grad = False

class InterpPara(nn.Module):
    """This class acts as the 1D interplation in the parametric space and therefore output a parameter mode in the Tensor Decomposition (TD) sens """
    def __init__(self, mu_min, mu_max,N_mu):
        super(InterpPara, self).__init__()
        import numpy as np
        # super(InterpPara, self).__init__()
        self.register_buffer('float_config',torch.tensor([0.0])  )                                                     # Keep track of device and dtype used throughout the model
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
        # Interpolation (nodal values) layer
        # self.NodalValues_para = nn.Parameter(data=torch.linspace(self.mu_min,self.mu_max,self.N_mu).pow(-1), requires_grad=False)
        self.NodalValues_para = nn.Parameter(data=torch.ones(self.N_mu), requires_grad=False)  
        self.InterpoLayer = nn.Linear(self.N_mu,1,bias=False)
        # Initialise with linear mode
        self.InterpoLayer.weight.data = 0.1*self.NodalValues_para
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

    def UnFreeze_FEM(self):
        """Set the nodale values as trainable parameters """
        self.InterpoLayer.weight.requires_grad = True
            
    def Freeze_FEM(self):
        """Set the nodale values as untrainable parameters """
        self.InterpoLayer.weight.requires_grad = False

    def Init_from_previous(self,previous_model):
        newparacoordinates = [coord for coord in self.coordinates]
        newparacoordinates = torch.cat(newparacoordinates,dim=0)
        try:
             previous_model.float_config.dtype
        except:
            previous_model.register_buffer('float_config',torch.tensor([0.0],dtype = torch.float64)  )                                                     # Keep track of device and dtype used throughout the model
        self.InterpoLayer.weight.data = (previous_model(newparacoordinates).to(previous_model.float_config.dtype).T).to(self.float_config.dtype)

#%% Parametric NeuROM Tensor decomposition

class NeuROM(nn.Module):
    """This class builds the Reduced-order model from the interpolation NN for space and parameters space"""
    def __init__(self, mesh, ParametersList, config, n_modes_ini = 1, n_modes_max = 100):
        """ The ROM model is built on a tensor decomposition between space and the given parameters.
        agrs:
            - mesh (Object): The mesh object parsed from the gmsh .geo file
            - ParametersList (tensor): The parameter hypercube: tensor of size Mx3 with M parameters mu. Each row is mu_min, mu_max, N_mu 
            - config (Dictionnary): the config file
            - n_modes_ini (integer): the initial number of modes of the ROM
            - n_modes_max (integer): the maximum number of modes of the ROM """

        super(NeuROM, self).__init__()
        self.register_buffer('float_config',torch.tensor([0.0])  )                                                     # Keep track of device and dtype used throughout the model
        IndexesNon0BCs = [i for i, BC in enumerate(mesh.ListOfDirichletsBCsValues) if BC != 0]
        if IndexesNon0BCs and n_modes_max==1: #If non homogeneous BCs, add mode for relevement
            n_modes_max+=1
        self.IndexesNon0BCs = IndexesNon0BCs
        self.n_modes = n_modes_max
        self.n_modes_truncated = torch.min(torch.tensor(self.n_modes),torch.tensor(n_modes_ini))
        self.dimension = mesh.dimension
        if IndexesNon0BCs and self.n_modes_truncated==1: #If non homogeneous BCs, add mode for relevement
            self.n_modes_truncated+=1
        self.config = config
        self.n_para = len(ParametersList)
        match mesh.dimension:
            case '1':
                match config["solver"]["IntegralMethod"]:   
                    case  "Trapezoidal":
                        self.Space_modes = nn.ModuleList([MeshNN(mesh) for i in range(self.n_modes)])
                    case "Gaussian_quad":
                        self.Space_modes = nn.ModuleList([MeshNN_1D(mesh, config["interpolation"]["n_integr_points"]) for i in range(self.n_modes)])
            case '2':
                self.Space_modes = nn.ModuleList([MeshNN_2D(mesh, n_components= 2) for i in range(self.n_modes)])
        self.Para_modes = nn.ModuleList([nn.ModuleList([InterpPara(Para[0], Para[1], Para[2]) for Para in ParametersList]) for i in range(self.n_modes)])
        # Set BCs 

        if IndexesNon0BCs:
            # First modes get the Boundary conditions
            self.Space_modes[0].SetBCs(mesh.ListOfDirichletsBCsValues)
            for para in range(self.n_para):
                self.Para_modes[0][para].InterpoLayer.weight.data.fill_(1) # Mode for parameters not trained and set to 1 to get correct space BCs
                self.Para_modes[0][para].Freeze_FEM()
            # Following modes are homogeneous (admissible to 0)
            for i in range(1,self.n_modes):
                self.Space_modes[i].SetBCs([0]*len(mesh.ListOfDirichletsBCsValues))
        else:
            for i in range(self.n_modes):
                self.Space_modes[i].SetBCs([0]*len(mesh.ListOfDirichletsBCsValues))

        self.FreezeAll()
        self.UnfreezeTruncated()
    def train(self):
        """Enable the training mode of NeurROM"""
        self.training = True
        for i in range(self.n_modes_truncated):
            self.Space_modes[i].train() 

    def eval(self):
        """Enable the evaluation mode of NeurROM"""
        self.training = False
        for i in range(self.n_modes_truncated):
            self.Space_modes[i].eval()  

    def TrainingParameters(self, loss_decrease_c = 1e-7,Max_epochs = 1000, learning_rate = 0.001):
        """Initialise the default training parameters used to train the ROM"""
        self.loss_decrease_c = loss_decrease_c
        self.Max_epochs = Max_epochs
        self.learning_rate = learning_rate


    def FreezeAll(self):
        """This method allows to freeze all sub neural networks"""
        self.Freeze_Mesh()
        self.Freeze_Space()
        self.Freeze_MeshPara()
        self.Freeze_Para()

    def SaveCoordinates(self):
        for m in range(self.n_modes_truncated):
            self.Space_modes[m].coord_old = [self.Space_modes[m].coordinates[i].data.item() for i in range(len(self.Space_modes[m].coordinates))]

    
    def AddMode(self):
        """This method allows to freeze the already computed modes and free the new mode when a new mode is required"""
        self.n_modes_truncated += 1     # Increment the number of modes used in the truncated tensor decomposition
        Mesh_status = self.Mesh_status  # Remember Mesh status
        self.FreezeAll()
        self.Mesh_status = Mesh_status  # Revert tocorrect  Mesh status
        self.Space_modes[self.n_modes_truncated-1].ZeroOut()
        self.Space_modes[self.n_modes_truncated-1].UnFreeze_FEM()

        if self.Mesh_status == 'Free':
            self.UnFreeze_Mesh()
        for j in range(self.n_para):
            self.Para_modes[self.n_modes_truncated-1][j].UnFreeze_FEM()  

    def AddMode2Optimizer(self,optim):
        "This method adds the newly freed parameters to the optimizer"
        New_mode_index = self.n_modes_truncated-1
        Space = self.Space_modes[self.n_modes_truncated-1].parameters()
        Para = self.Para_modes[self.n_modes_truncated-1][:].parameters()
        optim.add_param_group({'params': Space})
        optim.add_param_group({'params': Para})
    
    def Freeze_N_1(self):
        """Freezes N-1 first space modes """ 
        for i in range(self.n_modes_truncated-1):
            self.Space_modes[i].Freeze_FEM() 

    def UnfreezeTruncated(self):
        """Une freezes the used modes of the ROM """ 
        for i in range(self.n_modes_truncated):
            self.Space_modes[i].UnFreeze_FEM()  
        
        if self.IndexesNon0BCs:
            "Keep first modes frozen so that BCs are well accounted for if non-homogeneous"
            for i in range(1,self.n_modes_truncated):
                for j in range(self.n_para):
                    self.Para_modes[i][j].UnFreeze_FEM()
        else:
            for i in range(self.n_modes_truncated):  
                for j in range(self.n_para): 
                    self.Para_modes[i][j].UnFreeze_FEM()

    def Freeze_Mesh(self):
        """Set the space coordinates as untrainable parameters"""
        self.Mesh_status = 'Frozen'
        for i in range(self.n_modes):
            self.Space_modes[i].Freeze_Mesh()

    def UnFreeze_Mesh(self):
        """Set the space coordinates as trainable parameters"""
        self.Mesh_status = 'Free'
        for i in range(self.n_modes_truncated):
            self.Space_modes[i].UnFreeze_Mesh()

    def Freeze_Space(self):
        """Set the spatial modes as untrainable """
        for i in range(self.n_modes):
            self.Space_modes[i].Freeze_FEM()   

    def UnFreeze_Space(self):
        """Set the spatial modes as trainable """
        for i in range(self.n_modes_truncated):
            self.Space_modes[i].UnFreeze_FEM()   

    def Freeze_MeshPara(self):
        """Set the para coordinates as untrainable parameters"""
        for i in range(self.n_modes):
            for j in range(self.n_para):
                self.Para_modes[i][j].Freeze_Mesh()

    def UnFreeze_MeshPara(self):
        """Set the para coordinates as trainable parameters"""
        for i in range(self.n_modes):
            for j in range(self.n_para):
                self.Para_modes[i][j].UnFreeze_Mesh()

    def Freeze_Para(self):
        """Set the para modes as untrainable """
        for i in range(self.n_modes):
            for j in range(self.n_para):
                self.Para_modes[i][j].Freeze_FEM()  

    def UnFreeze_Para(self):
        """Set the para modes as trainable """
        for i in range(self.n_modes):
            for j in range(self.n_para):
                self.Para_modes[i][j].UnFreeze_FEM()  

    def forward(self,x,mu):
        # Use list comprehension to create Para_modes
        Para_mode_Lists = [
        [self.Para_modes[mode][l](mu[l][:,0].view(-1,1))[:,None] for l in range(self.n_para)]
        for mode in range(self.n_modes_truncated)
        ]

        Para_modes = [
            torch.cat([torch.unsqueeze(Para_mode_Lists[m][l],dim=0) for m in range(self.n_modes_truncated)], dim=0)
            for l in range(self.n_para)
        ]

        match self.dimension:
            case '1':
                match self.config["solver"]["IntegralMethod"]:
                    case "Trapezoidal":
                        Space_modes = [self.Space_modes[l](x) for l in range(self.n_modes_truncated)]
                        Space_modes = torch.cat(Space_modes,dim=1)
                        if len(mu)==1:
                            out = torch.einsum('ik,kj->ij',Space_modes,Para_modes[0].view(self.n_modes_truncated,Para_modes[0].shape[1]))
                        elif len(mu)==2:
                            out = torch.einsum('ik,kj,kl->ijl',Space_modes,Para_modes[0].view(self.n_modes_truncated,Para_modes[0].shape[1]),
                                            Para_modes[1].view(self.n_modes_truncated,Para_modes[1].shape[1]))
            
                    case "Gaussian_quad":
                        Space_modes = []
                        for i in range(self.n_modes_truncated):
                            IDs_elems = torch.tensor(self.Space_modes[i].mesh.GetCellIds(x),dtype=torch.int)
                            u_k = self.Space_modes[i](x,IDs_elems)
                            Space_modes.append(u_k)
                        u_i = torch.stack(Space_modes,dim=1)
                        P1 = (Para_modes[0].view(self.n_modes_truncated,Para_modes[0].shape[1])).to(torch.float64)
                        out = torch.einsum('em...,mp->ep',u_i,P1)


            case '2':
                match self.n_para:
                    case 1:
                        Space_modes = []
                        for i in range(self.n_modes_truncated):
                            IDs_elems = torch.tensor(self.Space_modes[i].mesh.GetCellIds(x),dtype=torch.int)
                            u_k = self.Space_modes[i](torch.tensor(x),IDs_elems)
                            Space_modes.append(u_k)
                        u_i = torch.stack(Space_modes,dim=2)
                        P1 = (Para_modes[0].view(self.n_modes_truncated,Para_modes[0].shape[1])).to(torch.float64)
                        out = torch.einsum('xyk,kj->xyj',u_i,P1)
                    case 2:
                        Space_modes = []
                        for i in range(self.n_modes_truncated):
                            if self.Space_modes[i].IdStored:
                                    if not False in (x == self.Space_modes[i].Stored_ID["coordinates"]):
                                        IDs_elems = self.Space_modes[i].Stored_ID["Ids"]
                                        u_k = self.Space_modes[i](self.Space_modes[i].Stored_ID["coordinates"],IDs_elems)
                                    else:
                                        self.Space_modes[i].StoreIdList(x)
                                        IDs_elems = self.Space_modes[i].Stored_ID["Ids"]
                                        u_k = self.Space_modes[i](self.Space_modes[i].Stored_ID["coordinates"],IDs_elems)
                            else:
                                self.Space_modes[i].StoreIdList(x)
                                IDs_elems = self.Space_modes[i].Stored_ID["Ids"]
                                u_k = self.Space_modes[i](self.Space_modes[i].Stored_ID["coordinates"],IDs_elems)
                            # IDs_elems = torch.tensor(self.Space_modes[i].mesh.GetCellIds(x),dtype=torch.int)
                            # u_k = self.Space_modes[i](torch.tensor(x),IDs_elems)
                            Space_modes.append(u_k)
                        u_i = torch.stack(Space_modes,dim=2)
                        P1 = (Para_modes[0].view(self.n_modes_truncated,Para_modes[0].shape[1])).to(torch.float64)
                        P2 = (Para_modes[1].view(self.n_modes_truncated,Para_modes[1].shape[1])).to(torch.float64)
                        out = torch.einsum('xyk,kj,kp->xyjp',u_i,P1,P2)
                    case 3:
                        Space_modes = []
                        for i in range(self.n_modes_truncated):
                            IDs_elems = torch.tensor(self.Space_modes[i].mesh.GetCellIds(x),dtype=torch.int)
                            u_k = self.Space_modes[i](torch.tensor(x),IDs_elems)
                            Space_modes.append(u_k)
                        u_i = torch.stack(Space_modes,dim=2)
                        P1 = (Para_modes[0].view(self.n_modes_truncated,Para_modes[0].shape[1])).to(torch.float64)
                        P2 = (Para_modes[1].view(self.n_modes_truncated,Para_modes[1].shape[1])).to(torch.float64)
                        P3 = (Para_modes[1].view(self.n_modes_truncated,Para_modes[1].shape[1])).to(torch.float64)
                        out = torch.einsum('xyk,kj,kp,kl->xyjpl',u_i,P1,P2,P3)
        return out

    def Init_from_previous(self,PreviousFullModel,Model_provided = False):
        """Initialise the model by projection PreviousFullModel onto self 
        args:
            - PreviousFullModel (ROM model or string): model used as initial point
            - Model_provided (Boolean): says if PreviousFullModel is the actual model or a its name (a string)"""
        import os

        if Model_provided:
            BeamROM_coarse = PreviousFullModel
        else:
            if os.path.isfile(PreviousFullModel):
                BeamROM_coarse = torch.load(PreviousFullModel) # To load a full coarse model
            elif not os.path.isfile(PreviousFullModel):
                print('******** WARNING LEARNING FROM SCRATCH ********\n')
                return

        if self.config["training"]["RemoveLastMode"]:
            self.n_modes_truncated_coarse = min(BeamROM_coarse.n_modes_truncated-1,self.n_modes)
        else:
            self.n_modes_truncated_coarse = min(BeamROM_coarse.n_modes_truncated,self.n_modes)
        if self.n_modes_truncated_coarse > self.n_modes_truncated:
            self.n_modes_truncated = self.n_modes_truncated_coarse
        Nb_modes_coarse = BeamROM_coarse.n_modes_truncated
        Nb_parameters_fine = len(self.Para_modes[0])
        Nb_parameters_coarse = len(BeamROM_coarse.Para_modes[0])
        if self.n_modes_truncated_coarse<self.n_modes_truncated:
            for mode in range(self.n_modes_truncated):
                self.Space_modes[mode].ZeroOut()
                self.Space_modes[mode].UnFreeze_FEM()
        for mode in range(self.n_modes_truncated_coarse):
            self.Space_modes[mode].Init_from_previous(BeamROM_coarse.Space_modes[mode])
            for para in range(min(Nb_parameters_fine,Nb_parameters_coarse)):
                self.Para_modes[mode][para].Init_from_previous(BeamROM_coarse.Para_modes[mode][para])

#%% 2D interpolation

class InterpolationBlock2D_Lin(nn.Module):
    """This class performs the FEM (linear) interpolation based on 2D shape functions and nodal values"""
    def __init__(self, connectivity):
       
        super(InterpolationBlock2D_Lin, self).__init__()
        self.connectivity = connectivity.astype(int)
        self.updated_connectivity = True
    def UpdateConnectivity(self,connectivity):
        """This function updates the connectivity tables of the interpolatin class
        args: 
            connectivity (numpy array): The new connectivty table"""
        self.connectivity = connectivity.astype(int)
        self.updated_connectivity = True
    def forward(self, x, cell_id, nodal_values, shape_functions, relation_BC_node_IDs, relation_BC_normals, relation_BC_values,node_mask_x = 'nan', node_mask_y= 'nan', nodal_values_tensor= 'nan', flag_training = 'True'):
        '''Performs the 2D linear interpolation
        args:
            - x (tensor): space coordinate where to do the evaluation
            - cell id (integer array): Corresponding element(s)
            - shape_functions corresponding N_i(x)
        '''
        vers = 'new_V2'                                                             # Enables 'old' slow implementation or 'New_V2' more efficient implementation
        if flag_training:
            if vers == 'old':
                cell_nodes_IDs = self.connectivity[cell_id,:] - 1
                if cell_nodes_IDs.ndim == 1:
                    cell_nodes_IDs = np.expand_dims(cell_nodes_IDs,0)

            if self.updated_connectivity:
                cell_nodes_IDs = self.connectivity[cell_id,:] - 1
                if cell_nodes_IDs.ndim == 1:
                    cell_nodes_IDs = np.expand_dims(cell_nodes_IDs,0)
                self.updated_connectivity = False
                if vers == 'new_V2':
                    self.Ids = torch.as_tensor(cell_nodes_IDs).to(nodal_values['x_free'].device).t()[:,:,None]
                else:
                    self.Ids = torch.as_tensor(cell_nodes_IDs).to(nodal_values[0][0].device).t()[:,None,:]
            match vers:
                case 'old':
                    node1_value =  torch.stack([torch.cat([val[row] for row in cell_nodes_IDs[:,0]]) for val in nodal_values], dim=0)
                    node2_value =  torch.stack([torch.cat([val[row] for row in cell_nodes_IDs[:,1]]) for val in nodal_values], dim=0)
                    node3_value =  torch.stack([torch.cat([val[row] for row in cell_nodes_IDs[:,2]]) for val in nodal_values], dim=0)
                    self.nodes_values = torch.stack([node1_value,node2_value,node3_value])                
                    u = shape_functions[:,0]*node1_value + shape_functions[:,1]*node2_value + shape_functions[:,2]*node3_value
                case 'new':
                    nodal_values_tensor = torch.stack([torch.cat(tuple(val)) for val in nodal_values], dim=0)
                    self.nodes_values =  torch.gather(nodal_values_tensor[None,:,:].repeat(3,1,1),2, self.Ids.repeat(1,2,1))
                    u = torch.einsum('ixg,gi->xg',self.nodes_values,shape_functions)
                case 'new_V2':
                    nodal_values_tensor = torch.ones_like(nodal_values_tensor)
                    nodal_values_tensor[node_mask_x,0] = nodal_values['x_free']
                    nodal_values_tensor[node_mask_y,1] = nodal_values['y_free']
                    nodal_values_tensor[~node_mask_x,0] = nodal_values['x_imposed']                    
                    nodal_values_tensor[~node_mask_y,1] = nodal_values['y_imposed']                    
                    self.nodes_values =  torch.gather(nodal_values_tensor[None,:,:].repeat(3,1,1),1, self.Ids.repeat(1,1,2))
                    u = torch.einsum('igx,gi->xg',self.nodes_values,shape_functions)
            return u

        else:
            cell_nodes_IDs = self.connectivity[cell_id,:] - 1
            if cell_nodes_IDs.ndim == 1:
                cell_nodes_IDs = np.expand_dims(cell_nodes_IDs,0)
            if vers == 'old':
                values = torch.ones_like(torch.tensor(nodal_values, dtype=nodal_values[0][0].dtype,  device=nodal_values[0][0].device))
                for j in range(values.shape[0]):
                    for k in range(values.shape[1]):
                        values[j,k] = nodal_values[j][k]
            for i in range(len(relation_BC_node_IDs)):
                nodes = relation_BC_node_IDs[i]
                normals = relation_BC_normals[i]
                value = relation_BC_values[i]

                if len(value)>1:
                    for j in range(nodes.shape[0]):

                        ID = nodes[j]
                        normal = normals[j]

                        if np.isclose(normal[0],0.0, atol=1.0e-8):
                            values[2,ID] = value[0]/normal[1]
                            values[1,ID] = value[1]/normal[1]
                        elif np.isclose(normal[1],0.0, atol=1.0e-8):
                            values[0,ID] = value[0]/normal[0]
                            values[2,ID] = value[1]/normal[0]
                        else:
                            values[0,ID] = (value[0] - nodal_values[2][ID]*normal[1])/normal[0]
                            values[1,ID] = (value[1] - nodal_values[2][ID]*normal[0])/normal[1]

                elif len(value)==1:
                    for j in range(nodes.shape[0]):

                        ID = nodes[j]
                        normal = normals[j]

                        if np.isclose(normal[0],0.0, atol=1.0e-8):
                            values[2,ID] = torch.nn.Parameter(torch.tensor([0*value[0]]))
                            values[1,ID] = torch.nn.Parameter(torch.tensor([value[0]]))
                        elif np.isclose(normal[1],0.0, atol=1.0e-8):
                            values[0,ID] = torch.nn.Parameter(torch.tensor([value[0]]))
                            values[2,ID] = torch.nn.Parameter(torch.tensor([0*value[0]]))
                        else:
                            values[0,ID] = (value[0]*normal[0] - nodal_values[2][ID]*normal[1])/normal[0]
                            values[1,ID] = (value[0]*normal[1] - nodal_values[2][ID]*normal[0])/normal[1]
            match vers:
                case 'old':
                    node1_value =  torch.stack([values[:,row] for row in cell_nodes_IDs[:,0]], dim=1)
                    node2_value =  torch.stack([values[:,row] for row in cell_nodes_IDs[:,1]], dim=1)
                    node3_value =  torch.stack([values[:,row] for row in cell_nodes_IDs[:,2]], dim=1)
                    u = shape_functions[:,0]*node1_value + shape_functions[:,1]*node2_value + shape_functions[:,2]*node3_value
                case 'new_V2':
                    nodal_values_tensor = torch.ones_like(nodal_values_tensor)
                    nodal_values_tensor[node_mask_x,0] = nodal_values['x_free']
                    nodal_values_tensor[node_mask_y,1] = nodal_values['y_free']
                    nodal_values_tensor[~node_mask_x,0] = nodal_values['x_imposed']                    
                    nodal_values_tensor[~node_mask_y,1] = nodal_values['y_imposed']
                    Ids = torch.as_tensor(cell_nodes_IDs).to(nodal_values['x_free'].device).t()[:,:,None]
                    nodes_values =  torch.gather(nodal_values_tensor[None,:,:].repeat(3,1,1),1, Ids.repeat(1,1,2))
                    u = torch.einsum('igx,gi->xg',nodes_values,shape_functions)
            return u


class InterpolationBlock2D_Quad(nn.Module):
    
    def __init__(self, connectivity):
       
        super(InterpolationBlock2D_Quad, self).__init__()
        self.connectivity = connectivity.astype(int)

    def forward(self, x, cell_id, nodal_values, shape_functions, relation_BC_node_IDs, relation_BC_normals, relation_BC_values, flag_training):
        """ This is the forward function of the Linear element block. Note that to prevent extrapolation outside of the structure's geometry, 
        phantom elements are used to cancel out the interpolation shape functions outside of the beam.
        Those phantom elements are flagged with index -1
        """

        cell_nodes_IDs = self.connectivity[cell_id,:] - 1
        if cell_nodes_IDs.ndim == 1:
            cell_nodes_IDs = np.expand_dims(cell_nodes_IDs,0)

        node1_value =  torch.stack([torch.cat([val[row] for row in cell_nodes_IDs[:,0]]) for val in nodal_values], dim=0)
        node2_value =  torch.stack([torch.cat([val[row] for row in cell_nodes_IDs[:,1]]) for val in nodal_values], dim=0)
        node3_value =  torch.stack([torch.cat([val[row] for row in cell_nodes_IDs[:,2]]) for val in nodal_values], dim=0)

        node4_value =  torch.stack([torch.cat([val[row] for row in cell_nodes_IDs[:,3]]) for val in nodal_values], dim=0)
        node5_value =  torch.stack([torch.cat([val[row] for row in cell_nodes_IDs[:,4]]) for val in nodal_values], dim=0)
        node6_value =  torch.stack([torch.cat([val[row] for row in cell_nodes_IDs[:,5]]) for val in nodal_values], dim=0)
        #out = torch.cat(shape_functions[:,0]*node1_value[:,0] + shape_functions[:,1]*node2_value[:,0] + shape_functions[:,2]*node3_value[:,0], shape_functions[:,0]*node1_value[:,1] + shape_functions[:,1]*node2_value[:,1] + shape_functions[:,2]*node3_value[:,1])

        prod = [shape_functions[:,0,k]*node1_value + shape_functions[:,1,k]*node2_value + shape_functions[:,2,k]*node3_value+\
            shape_functions[:,3,k]*node4_value + shape_functions[:,4,k]*node5_value + shape_functions[:,5,k]*node6_value for k in range(shape_functions.shape[2])]
        prod = torch.stack(prod, dim=2)
        return prod



class ElementBlock2D_Lin(nn.Module):
    """
    Returns:
         N_i(x)'s for each nodes within each element"""
    def __init__(self, connectivity):
        """ Initialise the Linear Bar element 
        Args:
            connectivity (Interger table): Connectivity matrix of the 1D mesh
        """
        super(ElementBlock2D_Lin, self).__init__()
        self.connectivity = connectivity.astype(int)
        self.register_buffer('GaussPoint',self.GP())
    def UpdateConnectivity(self,connectivity):
        self.connectivity = connectivity.astype(int)

    def GP(self):
        return torch.tensor([[1/3, 1/3, 1/3]],dtype=torch.float64, requires_grad=True) # a1, a2, a3 the 3 area coordinates

    def forward(self, x, cell_id, coordinates, nodal_values,coord_mask,coordinates_all,flag_training):
        """ This is the forward function of the Linear element block that outputs 2D linear shape functions based on
        args:
            - x (tensor) : position where to evalutate the shape functions
            - cell_id (interger) : Associated element
            - coordinates (np array): nodal coordinates array
            - coord_mask (boolean tensor) : mask for free nodes in the mesh
            - coordinates_all prealocated tensor of all coordinates
        """

        cell_nodes_IDs = self.connectivity[cell_id,:]
        if cell_nodes_IDs.ndim == 1:
            cell_nodes_IDs = np.expand_dims(cell_nodes_IDs,0)
        vers = 'new_V2'
        match vers:
            case 'old':
                node1_coord =  torch.cat([coordinates[row-1] for row in cell_nodes_IDs[:,0]])
                node2_coord =  torch.cat([coordinates[row-1] for row in cell_nodes_IDs[:,1]])
                node3_coord =  torch.cat([coordinates[row-1] for row in cell_nodes_IDs[:,2]])
            case 'new':
                nodal_coord_tensor =torch.cat(tuple(coordinates),dim = 0)
                Ids = torch.as_tensor(cell_nodes_IDs-1).to(nodal_coord_tensor.device).t()[:,:,None]
                nodes_coord =  torch.gather(nodal_coord_tensor[None,:,:].repeat(3,1,1),1, Ids.repeat(1,1,2))
            case 'new_V2':
                coordinates_all = torch.ones_like(coordinates_all)
                coordinates_all[coord_mask] = coordinates['free']
                coordinates_all[~coord_mask] = coordinates['imposed']
                Ids = torch.as_tensor(cell_nodes_IDs-1).to(coordinates_all.device).t()[:,:,None]
                nodes_coord =  torch.gather(coordinates_all[None,:,:].repeat(3,1,1),1, Ids.repeat(1,1,2))
        if flag_training:

            refCoordg = self.GaussPoint.repeat(cell_id.shape[0],1)

            w_g = 0.5                           # Gauss weight
            match vers:
                case 'old':
                    Ng = torch.stack((refCoordg[:,0], refCoordg[:,1], refCoordg[:,2]),dim=1) #.view(sh_R.shape[0],-1) # Left | Right | Middle
                case 'new' | 'new_V2':
                    Ng = refCoordg

            match vers:
                case 'old':
                    x_g = torch.stack([Ng[:,0]*node1_coord[:,0] + Ng[:,1]*node2_coord[:,0] + Ng[:,2]*node3_coord[:,0],Ng[:,0]*node1_coord[:,1] + Ng[:,1]*node2_coord[:,1] + Ng[:,2]*node3_coord[:,1]],dim=1)
                case 'new' | 'new_V2':
                    x_g = torch.einsum('nex,en->ex',nodes_coord,Ng)

            match vers:
                case 'old':
                    refCoord = GetRefCoord(x_g[:,0],x_g[:,1],node1_coord[:,0],node2_coord[:,0],node3_coord[:,0],node1_coord[:,1],node2_coord[:,1],node3_coord[:,1])
                case 'new' | 'new_V2':
                    refCoord = GetRefCoord(x_g[:,0],x_g[:,1],nodes_coord[0,:,0],nodes_coord[1,:,0],nodes_coord[2,:,0],nodes_coord[0,:,1],nodes_coord[1,:,1],nodes_coord[2,:,1])

            match vers:
                case 'old':
                    N = torch.stack((refCoord[:,0], refCoord[:,1], refCoord[:,2]),dim=1) #.view(sh_R.shape[0],-1) # Left | Right | Middle
                case 'new' | 'new_V2':
                    N = refCoord

            match vers:
                case 'old':
                    detJ = (node1_coord[:,0] - node3_coord[:,0])*(node2_coord[:,1] - node3_coord[:,1]) - (node2_coord[:,0] - node3_coord[:,0])*(node1_coord[:,1] - node3_coord[:,1])
                case 'new' | 'new_V2':
                    detJ = (nodes_coord[0,:,0] - nodes_coord[2,:,0])*(nodes_coord[1,:,1] - nodes_coord[2,:,1]) - (nodes_coord[1,:,0] - nodes_coord[2,:,0])*(nodes_coord[0,:,1] - nodes_coord[2,:,1])
            return N,x_g, detJ*w_g

        else:
            match vers:
                case 'old':
                    refCoord = GetRefCoord(x[:,0],x[:,1],node1_coord[:,0],node2_coord[:,0],node3_coord[:,0],node1_coord[:,1],node2_coord[:,1],node3_coord[:,1])
                case 'new'| 'new_V2':
                    refCoord = GetRefCoord(x[:,0],x[:,1],nodes_coord[0,:,0],nodes_coord[1,:,0],nodes_coord[2,:,0],nodes_coord[0,:,1],nodes_coord[1,:,1],nodes_coord[2,:,1])
            out = torch.stack((refCoord[:,0], refCoord[:,1], refCoord[:,2]),dim=1) #.view(sh_R.shape[0],-1) # Left | Right | Middle
            return out

class ElementBlock2D_Quad(nn.Module):
    """
    Returns:
         N_i(x)'s for each nodes within each element"""
    def __init__(self, connectivity):
        """ Initialise the Linear Bar element 
        Args:
            connectivity (Interger table): Connectivity matrix of the 1D mesh
        """
        super(ElementBlock2D_Quad, self).__init__()
        self.connectivity = connectivity.astype(int)
    
    def GP(self):
        return torch.tensor([[1/6,1/6, 1-2/6],[2/3,1/6, 1-2/3-1/6],[1/6,2/3, 1-2/3-1/6]], dtype=torch.float64, requires_grad=True)

    def forward(self, x, cell_id, coordinates, nodal_values, flag_training):
        """ This is the forward function of the Linear element block. Note that to prevent extrapolation outside of the structure's geometry, 
        phantom elements are used to cancel out the interpolation shape functions outside of the beam.
        Those phantom elements are flagged with index -1
        """

        cell_nodes_IDs = self.connectivity[cell_id,:]
        if cell_nodes_IDs.ndim == 1:
            cell_nodes_IDs = np.expand_dims(cell_nodes_IDs,0)

        node1_coord =  torch.cat([coordinates[row-1] for row in cell_nodes_IDs[:,0]])
        node2_coord =  torch.cat([coordinates[row-1] for row in cell_nodes_IDs[:,1]])
        node3_coord =  torch.cat([coordinates[row-1] for row in cell_nodes_IDs[:,2]])

        if flag_training:
            refCoordg = self.GP().unsqueeze(0).repeat(cell_id.shape[0],1,1)

            w_g = 1/6

            x_g = torch.stack([refCoordg[:,:,0]*node1_coord[:,0].unsqueeze(1) + refCoordg[:,:,1]*node2_coord[:,0].unsqueeze(1) + refCoordg[:,:,2]*node3_coord[:,0].unsqueeze(1),\
                refCoordg[:,:,0]*node1_coord[:,1].unsqueeze(1) + refCoordg[:,:,1]*node2_coord[:,1].unsqueeze(1) + refCoordg[:,:,2]*node3_coord[:,1].unsqueeze(1)],dim=2)

            refCoord = [GetRefCoord(x_g[:,k,0],x_g[:,k,1],node1_coord[:,0],node2_coord[:,0],node3_coord[:,0],node1_coord[:,1],node2_coord[:,1],node3_coord[:,1]) for k in range(x_g.shape[1])]
            refCoord = torch.stack(refCoord, dim=1)

            N1 = refCoord[:,:,0]*(2*refCoord[:,:,0]-1)
            N2 = refCoord[:,:,1]*(2*refCoord[:,:,1]-1)
            N3 = refCoord[:,:,2]*(2*refCoord[:,:,2]-1)

            N4 = 4*refCoord[:,:,0]*refCoord[:,:,1]
            N5 = 4*refCoord[:,:,1]*refCoord[:,:,2]
            N6 = 4*refCoord[:,:,2]*refCoord[:,:,0]


            N = torch.stack((N1,N2,N3,N4,N5,N6),dim=1)

            detJ = (node1_coord[:,0] - node3_coord[:,0])*(node2_coord[:,1] - node3_coord[:,1]) - (node2_coord[:,0] - node3_coord[:,0])*(node1_coord[:,1] - node3_coord[:,1])
            
            return N, x_g, detJ*w_g


        else:
            if len(x.shape)<3:
                x = x.unsqueeze(1)
            
            refCoord = [GetRefCoord(x[:,k,0],x[:,k,1],node1_coord[:,0],node2_coord[:,0],node3_coord[:,0],node1_coord[:,1],node2_coord[:,1],node3_coord[:,1]) for k in range(x.shape[1])]
            refCoord = torch.stack(refCoord, dim=1)

            N1 = refCoord[:,:,0]*(2*refCoord[:,:,0]-1)
            N2 = refCoord[:,:,1]*(2*refCoord[:,:,1]-1)
            N3 = refCoord[:,:,2]*(2*refCoord[:,:,2]-1)

            N4 = 4*refCoord[:,:,0]*refCoord[:,:,1]
            N5 = 4*refCoord[:,:,1]*refCoord[:,:,2]
            N6 = 4*refCoord[:,:,2]*refCoord[:,:,0]

            N = torch.stack((N1,N2,N3,N4,N5,N6),dim=1)

            out = torch.stack((N1,N2,N3,N4,N5,N6),dim=1) #.view(sh_R.shape[0],-1) # Left | Right | Middle
            return out

class MeshNN_2D(nn.Module):
    """ This class is a space HiDeNN building a Finite Element (FE) interpolation over the space domain. 
    The coordinates of the nodes of the underlying mesh are trainable. Those coordinates are passed as a List of Parameters to the subsequent sub-neural networks
    Updating those parameters correspond to r-adaptativity
    The Interpolation layer weights correspond to the nodal values. Updating them 
    is equivqlent to solving the PDE. """


    def __init__(self, mesh, n_components):
        super(MeshNN_2D, self).__init__()
        self.register_buffer('float_config',torch.tensor([0.0])  )                                                     # Keep track of device and dtype used throughout the model

        vers = 'new_V2'
        if vers =='new_V2': 
            self.register_buffer('coordinates_all', torch.cat(tuple([(torch.tensor([mesh.Nodes[i][1:int(mesh.dimension)+1]],dtype=torch.float64)) \
                                                                        for i in range(len(mesh.Nodes))])))                                                     # Keep track of device and dtype used throughout the model
            # self.coordinates_all = torch.cat(tuple([(torch.tensor([mesh.Nodes[i][1:int(mesh.dimension)+1]],dtype=torch.float64)) \
            #                                                             for i in range(len(mesh.Nodes))]))
            self.coordinates =nn.ParameterDict({
                                                'free': self.coordinates_all,
                                                'imposed': [],
                                                'mask':[]
                                                })

        else:
            self.coordinates = nn.ParameterList([nn.Parameter(torch.tensor([mesh.Nodes[i][1:int(mesh.dimension)+1]],dtype=torch.float64)) \
                                             for i in range(len(mesh.Nodes))])

        # self.values = 0.0001*torch.randint(low=-1000, high=1000, size=(mesh.NNodes,n_components))
        # self.values =0.5*torch.ones((mesh.NNodes,n_components))
        self.register_buffer('values',0.5*torch.ones((mesh.NNodes,n_components)))

        self.frozen_BC_node_IDs = []
        self.frozen_BC_node_IDs_x = []             
        self.frozen_BC_node_IDs_y = []             
        self.frozen_BC_component_IDs = []
        self.relation_BC_node_IDs = []
        self.relation_BC_values = []
        self.relation_BC_normals = []
        self.constit_BC_node_IDs = []
        self.relation_BC_lines = []


        self.connectivity = mesh.Connectivity
        self.ExcludeFromDirichlet = mesh.ExcludedPoints
        self.borders_nodes = mesh.borders_nodes
        self.elements_generation = np.ones(self.connectivity.shape[0])
        self.ListOfDirichletsBCsRelation = mesh.ListOfDirichletsBCsRelation
        self.ListOfDirichletsBCsConstit = mesh.ListOfDirichletsBCsConstit
        self.DirichletBoundaryNodes = mesh.DirichletBoundaryNodes
        self.ListOfDirichletsBCsNormals = mesh.ListOfDirichletsBCsNormals
        # self.normals = mesh.normals
        self.dofs = mesh.NNodes*mesh.dim # Number of Dofs
        self.NElem = mesh.NElem
        self.n_components = n_components
        self.ListOfDirichletsBCsValues = mesh.ListOfDirichletsBCsValues
        self.mesh = mesh
        self.IdStored = False

        if mesh.NoBC==False:
            self.SetBCs(mesh.ListOfDirichletsBCsValues)
            self.NBCs = len(mesh.ListOfDirichletsBCsIds) # Number of prescribed Dofs
        else:
            self.NBCs = 0

        self.order = mesh.order
        if mesh.order =='1':
            self.ElementBlock = ElementBlock2D_Lin(mesh.Connectivity)
            self.Interpolation = InterpolationBlock2D_Lin(mesh.Connectivity)
        elif mesh.order == '2':
            self.ElementBlock = ElementBlock2D_Quad(mesh.Connectivity)
            self.Interpolation = InterpolationBlock2D_Quad(mesh.Connectivity)

        # set parameters 
        self.RefinementParameters()
        self.TrainingParameters()
        self.UnFreeze_FEM()

    def ZeroOut(self):
        """
            Sets the nodal values of the model to zero
        """
        vers = 'New_V2'
        match vers:
            case 'old':
                # self.nodal_values = nn.ParameterList([nn.Parameter(0*torch.tensor([i[0]])) for i in self.values])
                self.nodal_values_x = nn.ParameterList([nn.Parameter((0*torch.tensor([i[0]], dtype = self.float_config.dtype, device = self.float_config.device))) for i in self.nodal_values_x])
                self.nodal_values_y = nn.ParameterList([nn.Parameter((0*torch.tensor([i[0]], dtype = self.float_config.dtype, device = self.float_config.device))) for i in self.nodal_values_y])
                self.nodal_values = [self.nodal_values_x,self.nodal_values_y]
            case 'New_V2':
                self.nodal_values['x_free'] = 0*self.nodal_values['x_free']
                self.nodal_values['y_free'] = 0*self.nodal_values['y_free']

                self.nodal_values['x_imposed'] = 0*self.nodal_values['x_imposed']
                self.nodal_values['y_imposed'] = 0*self.nodal_values['y_imposed']


    def StoreIdList(self,x):
        if torch.is_tensor(x):
            self.Stored_ID = {"coordinates": x, 
                                "Ids": self.mesh.GetCellIds(x)}
        else:
            self.Stored_ID = {"coordinates": torch.tensor(x), 
                                "Ids": self.mesh.GetCellIds(x)}
        self.IdStored = True

    def Init_from_previous(self,CoarseModel):
        """"
            Initialise the current model based on a previous "CoarseModel"
        """
        try:
             CoarseModel.float_config.dtype
        except:
            CoarseModel.float_config = torch.tensor([0],dtype = torch.float64)
        vers = 'New_V2'
        match vers:
            case 'old':
                newcoordinates = [coord for coord in self.coordinates]
                newcoordinates = torch.cat(newcoordinates,dim=0)
            case 'New_V2':
                newcoordinates = torch.ones_like(self.coordinates_all)
                newcoordinates[self.coord_free] = self.coordinates['free']
                newcoordinates[~self.coord_free] = self.coordinates['imposed']
        IDs_newcoord = torch.tensor(CoarseModel.mesh.GetCellIds(newcoordinates),dtype=torch.int)
        NewNodalValues = CoarseModel(newcoordinates.to(CoarseModel.float_config.dtype),IDs_newcoord).to(self.float_config.dtype).t()
        # check if a cell ID was not found for some new nodes 
        if -1 in IDs_newcoord:
            index_neg = (IDs_newcoord == -1).nonzero(as_tuple=False)
            match vers:
                case 'old':
                    oldcoordinates = [coord for coord in CoarseModel.coordinates]
                    oldcoordinates = torch.cat(oldcoordinates,dim=0)
                case 'New_V2':
                    oldcoordinates = torch.ones_like(CoarseModel.coordinates_all)
                    oldcoordinates[CoarseModel.coord_free] = CoarseModel.coordinates['free']
                    oldcoordinates[~CoarseModel.coord_free] = CoarseModel.coordinates['imposed']
            for ind_neg in index_neg:
                not_found_coordinates = newcoordinates[ind_neg]
                dist_vect = not_found_coordinates - oldcoordinates
                dist = torch.norm(dist_vect, dim=1)
                closest_old_nodal_value = dist.topk(1, largest=False)[1]
                match vers:
                    case 'old':
                        NewNodalValues[0][ind_neg] = CoarseModel.nodal_values_x[closest_old_nodal_value].to(self.float_config.dtype)
                        NewNodalValues[1][ind_neg] = CoarseModel.nodal_values_y[closest_old_nodal_value].to(self.float_config.dtype)
                    case 'New_V2':
                        old_values = CoarseModel.values
                        old_values[CoarseModel.dofs_free_x,0] = CoarseModel.nodal_values['x_free']
                        old_values[CoarseModel.dofs_free_y,1] = CoarseModel.nodal_values['y_free']
                        old_values[~CoarseModel.dofs_free_x,0] = CoarseModel.nodal_values['x_imposed']
                        old_values[~CoarseModel.dofs_free_y,1] = CoarseModel.nodal_values['y_imposed']
                        NewNodalValues[ind_neg,:] =  old_values[closest_old_nodal_value,:].to(self.float_config.dtype).to(self.float_config.device)
        vers =  'New_V2'
        match vers:
            case 'old':
                new_nodal_values_x = nn.ParameterList([nn.Parameter((torch.tensor([i[0]]))) for i in NewNodalValues.t()]).to(self.float_config.dtype).to(self.float_config.device)
                new_nodal_values_y = nn.ParameterList([nn.Parameter(torch.tensor([i[1]])) for i in NewNodalValues.t()]).to(self.float_config.dtype).to(self.float_config.device)
                new_nodal_values = [new_nodal_values_x,new_nodal_values_y]
                self.nodal_values_x = new_nodal_values_x
                self.nodal_values_y = new_nodal_values_y
                self.nodal_values = new_nodal_values
            case 'New_V2':
                NewNodalValues = NewNodalValues
                self.nodal_values['x_free'] = NewNodalValues[self.dofs_free_x,0]
                self.nodal_values['x_imposed'] = NewNodalValues[~self.dofs_free_x,0]
                self.nodal_values['y_free'] = NewNodalValues[self.dofs_free_y,1]
                self.nodal_values['y_imposed'] = NewNodalValues[~self.dofs_free_y,1]
 

    def SetBCs(self, ListOfDirichletsBCsValues):
        """
            Sets the Boundary conditions and defines which parameters should be frozen based on the BCs
        """
        for i in range(len(ListOfDirichletsBCsValues)):
            if self.ListOfDirichletsBCsRelation[i] == False:
                if self.ListOfDirichletsBCsConstit[i] == False:
                    vers = 'new_V2'
                    if vers == 'new_V2':
                        IDs = torch.tensor(self.DirichletBoundaryNodes[i], dtype=torch.int)
                        IDs = torch.unique(IDs.reshape(IDs.shape[0],-1))-1
                        match self.ListOfDirichletsBCsNormals[i]:
                            case 0:
                                self.frozen_BC_node_IDs_x.append(IDs)
                            case 1:
                                self.frozen_BC_node_IDs_y.append(IDs)   
                        self.values[IDs,self.ListOfDirichletsBCsNormals[i]] = ListOfDirichletsBCsValues[i]
                        
                    else:
                        IDs = torch.tensor(self.DirichletBoundaryNodes[i], dtype=torch.int)
                        IDs = torch.unique(IDs.reshape(IDs.shape[0],-1))-1
                        self.frozen_BC_node_IDs.append(IDs)
                        self.frozen_BC_component_IDs.append(self.ListOfDirichletsBCsNormals[i])
                        self.values[IDs,self.ListOfDirichletsBCsNormals[i]] = ListOfDirichletsBCsValues[i]

                
            else:
                IDs = torch.tensor(self.DirichletBoundaryNodes[i], dtype=torch.int)
                IDs = torch.unique(IDs.reshape(IDs.shape[0],-1))-1
                self.relation_BC_node_IDs.append(IDs)
                self.relation_BC_values.append(ListOfDirichletsBCsValues[i])
                # self.relation_BC_normals.append(self.normals[IDs])

        for i in range(len(ListOfDirichletsBCsValues)):
            if self.ListOfDirichletsBCsConstit[i] == True:
                IDs = torch.tensor(self.DirichletBoundaryNodes[i], dtype=torch.int)
                IDs = torch.unique(IDs.reshape(IDs.shape[0],-1))-1

                if len(self.relation_BC_node_IDs)>0:
                    delete_relation = torch.cat(self.relation_BC_node_IDs)

                    for elem in IDs:
                        if elem in delete_relation:
                            IDs = IDs[IDs!=elem]
                if len(self.frozen_BC_node_IDs)>0:
                    delete_simple = torch.cat(self.frozen_BC_node_IDs)
                    for elem in IDs:
                        if elem in delete_simple:
                            IDs = IDs[IDs!=elem]

                self.constit_BC_node_IDs.append(IDs)

        if self.n_components ==2:
            if vers == 'new_V2':
                self.IDs_frozen_BC_node_y = torch.unique(torch.stack(self.frozen_BC_node_IDs_y))
                self.IDs_frozen_BC_node_x = torch.unique(torch.stack(self.frozen_BC_node_IDs_x))
                self.dofs_free_x =( torch.ones_like(self.values[:,0])==1)
                self.dofs_free_x[self.IDs_frozen_BC_node_x] = False
                self.dofs_free_y =( torch.ones_like(self.values[:,0])==1)
                self.dofs_free_y[self.IDs_frozen_BC_node_y] = False

                nodal_values_x_imposed = self.values[~self.dofs_free_x,0]
                nodal_values_y_imposed = self.values[~self.dofs_free_y,1]
                nodal_values_x_free = self.values[self.dofs_free_x,0]
                nodal_values_y_free = self.values[self.dofs_free_y,1]
                self.nodal_values = nn.ParameterDict({
                                                    'x_free': nodal_values_x_free,
                                                    'y_free': nodal_values_y_free,
                                                    'x_imposed': nodal_values_x_imposed,
                                                    'y_imposed': nodal_values_y_imposed
                                                    })
                border_nodes = torch.unique(torch.tensor(self.borders_nodes, dtype=torch.int))-1
                Fixed_Ids = torch.unique(torch.cat([self.IDs_frozen_BC_node_x,self.IDs_frozen_BC_node_y,border_nodes]))
                self.coord_free =(torch.ones_like(self.values[:,0])==1)
                self.coord_free[Fixed_Ids] = False
                self.coordinates['free'] = self.coordinates_all[self.coord_free,:]
                self.coordinates['imposed'] = self.coordinates_all[~self.coord_free,:]

            else:
            # nn.ParameterList is supposed to hold a single list of nn.Parameter and cannot contain other nn.ParameterLists
                self.nodal_values_x = nn.ParameterList([nn.Parameter(torch.tensor([i[0]])) for i in self.values])
                self.nodal_values_y = nn.ParameterList([nn.Parameter(torch.tensor([i[1]])) for i in self.values])
                self.nodal_values = [self.nodal_values_x,self.nodal_values_y]

        elif self.n_components ==3:
            # nn.ParameterList is supposed to hold a single list of nn.Parameter and cannot contain other nn.ParameterLists
            self.nodal_values_x = nn.ParameterList([nn.Parameter(torch.tensor([i[0]])) for i in self.values])
            self.nodal_values_y = nn.ParameterList([nn.Parameter(torch.tensor([i[1]])) for i in self.values])
            self.nodal_values_xy = nn.ParameterList([nn.Parameter(torch.tensor([i[2]])) for i in self.values])
            self.nodal_values = [self.nodal_values_x,self.nodal_values_y, self.nodal_values_xy]





    def SplitElem(self, el_id,point,value):
        nodes = self.connectivity[el_id]
        # Coord = [self.coordinates[int(i-1)] for i in nodes]
        # new_point = (Coord[0] + Coord[1] + Coord[2])/3
        # NewNodalValues = self(new_point,torch.tensor([el_id],dtype=torch.int)) 
        self.coordinates.append(point)
        new_connectivity = self.connectivity
        new_connectivity = np.delete(new_connectivity,(el_id),axis = 0)
        new_elem = np.array([[np.max(self.connectivity)+1, nodes[0], nodes[1]],
                    [np.max(self.connectivity)+1, nodes[1], nodes[2]],
                    [np.max(self.connectivity)+1, nodes[2], nodes[0]]])
        new_connectivity = np.vstack((new_connectivity,new_elem))
        self.connectivity = new_connectivity
        if self.order =='1':
            self.ElementBlock.UpdateConnectivity(self.connectivity)
            self.Interpolation.UpdateConnectivity(self.connectivity)
        elif self.order == '2':
            self.ElementBlock.UpdateConnectivity(self.connectivity)
            self.Interpolation.UpdateConnectivity(self.connectivity)
        self.nodal_values[0].append(value[0])
        self.nodal_values[1].append(value[1])
        self.NElem +=2

    def Split_hangingNodes(self,edge_id,edge_nodes,new_node):
        self.NElem +=1
        vers = 'New_V2'
        match vers:
            case 'old':
                nodes = self.connectivity[edge_id][0]
                Third_node = np.delete(nodes,np.where(nodes == edge_nodes[0]))
                Third_node = np.delete(Third_node,np.where(Third_node == edge_nodes[1]))
            case 'New_V2':
                nodes = self.connectivity[edge_id]
                Third_node = np.delete(nodes,np.where(nodes == edge_nodes[0]))
                Third_node = np.delete(Third_node,np.where(Third_node == edge_nodes[1]))


        new_connectivity = self.connectivity
        new_generation = self.elements_generation
        new_det = self.detJ_0.numpy()
        new_current_det = self.detJ.numpy()
        curren_det = new_current_det[edge_id]

        curren_gen = new_generation[edge_id]

        new_connectivity = np.delete(new_connectivity,(edge_id),axis = 0)
        new_generation = np.delete(new_generation,(edge_id),axis = 0)
        new_det = np.delete(new_det,(edge_id),axis = 0)
        new_current_det = np.delete(new_current_det,(edge_id),axis = 0)

        new_elem = np.array([   [edge_nodes[0], new_node, Third_node[0]],
                                [edge_nodes[1], new_node, Third_node[0]]])
        new_connectivity = np.vstack((new_connectivity,new_elem))
        new_generation = np.hstack((new_generation,np.repeat(np.array(curren_gen), 2, axis=None)))
        new_det = np.hstack((new_det,np.repeat(np.array(curren_det/2), 2, axis=None)))
        new_current_det = np.hstack((new_current_det,np.repeat(np.array(curren_det/2), 2, axis=None)))

        self.connectivity = new_connectivity
        self.elements_generation = new_generation
        self.detJ_0 = torch.tensor( new_det, 
                                    dtype = self.float_config.dtype, 
                                    device= self.float_config.device)
        self.detJ = torch.tensor( new_current_det, 
                                    dtype = self.float_config.dtype, 
                                    device= self.float_config.device)
        if self.order =='1':
            self.ElementBlock.UpdateConnectivity(self.connectivity)
            self.Interpolation.UpdateConnectivity(self.connectivity)
        elif self.order == '2':
            self.ElementBlock.UpdateConnectivity(self.connectivity)
            self.Interpolation.UpdateConnectivity(self.connectivity)

    def TrainingParameters(self, loss_decrease_c = 1e-7,Max_epochs = 1000, learning_rate = 0.001):
        self.loss_decrease_c = loss_decrease_c
        self.Max_epochs = Max_epochs
        self.learning_rate = learning_rate

    def Initresults(self):
        self.U_interm = []
        self.X_interm = []
        self.G_interm = []
        self.Connectivity_interm = []
        self.Jacobian_interm = []
        self.Jacobian_current_interm = []

    def StoreResults(self):
        vers = "new_V2"
        if vers == 'new_V2':
            u = self.values
            u[self.dofs_free_x,0] = self.nodal_values['x_free']
            u[self.dofs_free_y,1] = self.nodal_values['y_free']
            u[~self.dofs_free_x,0] = self.nodal_values['x_imposed']                    
            u[~self.dofs_free_y,1] = self.nodal_values['y_imposed']

        else:
            u_x = [u for u in self.nodal_values_x]
            u_y = [u for u in self.nodal_values_y]
            u = torch.stack([torch.cat(u_x),torch.cat(u_y)],dim=1)
        self.U_interm.append(u.detach().clone())

        if vers == 'new_V2':
            new_coord = self.coordinates_all
            new_coord[self.coord_free] = self.coordinates['free']
            new_coord[~self.coord_free] = self.coordinates['imposed']


        else:
            new_coord = [coord for coord in self.coordinates]
            new_coord = torch.cat(new_coord,dim=0).detach().clone()
        self.X_interm.append(new_coord.detach().clone())
        self.G_interm.append(self.elements_generation)
        self.Connectivity_interm.append(self.connectivity-1)
        self.Jacobian_interm.append(self.detJ_0.detach().clone())
        self.Jacobian_current_interm.append(self.detJ.detach().clone())

    def RefinementParameters(self,MaxGeneration = 2, Jacobian_threshold = 0.4):
        self.MaxGeneration = MaxGeneration
        self.Jacobian_threshold = Jacobian_threshold
        self.MaxGeneration_elements = 0
    def GetCoordIndex(idx):
        match Free:
            case True:
                idx_coord = torch.sum(coord_free_mask[:int(idx)]) - 1
                return 'free', idx_coord
            case False:
                idx_coord = (idx - torch.sum(coord_free_mask[:int(idx)])) - 1
                return 'imposed', idx_coord

    def SplitElemNonLoc(self, el_id):
        nodes = self.connectivity[el_id]
        vers = 'New_V2'
        match vers:
            case 'old':
                # Find edges of the element
                node1_indices = (self.connectivity[:, 0] == nodes[0]) + (self.connectivity[:, 1] == nodes[0]) + (self.connectivity[:, 2] == nodes[0])
                node2_indices = (self.connectivity[:, 0] == nodes[1]) + (self.connectivity[:, 1] == nodes[1]) + (self.connectivity[:, 2] == nodes[1])
                node3_indices = (self.connectivity[:, 0] == nodes[2]) + (self.connectivity[:, 1] == nodes[2]) + (self.connectivity[:, 2] == nodes[2])
                # Find edges where there are the nodes
                edge_1_2 = np.transpose(np.vstack((node1_indices,node2_indices)))
                edge_1_3 = np.transpose(np.vstack((node1_indices,node3_indices)))
                edge_2_3 = np.transpose(np.vstack((node2_indices,node3_indices)))
                # Find element sharing the both nodes
                elem_edge_1 = np.where(np.all(edge_1_2 == [True,True],axis=1))[0]
                elem_edge_2 = np.where(np.all(edge_1_3 == [True,True],axis=1))[0]
                elem_edge_3 = np.where(np.all(edge_2_3 == [True,True],axis=1))[0]
                # Remove current element
                elem_edge_1 = np.delete(elem_edge_1,np.where(elem_edge_1 == el_id.item()))
                elem_edge_2 = np.delete(elem_edge_2,np.where(elem_edge_2 == el_id.item()))
                elem_edge_3 = np.delete(elem_edge_3,np.where(elem_edge_3 == el_id.item()))
                Coord = [self.coordinates[int(i-1)] for i in nodes]
                New_coordinates = torch.vstack([0.5*(Coord[0]+Coord[1]),
                                                0.5*(Coord[0]+Coord[2]),
                                                0.5*(Coord[1]+Coord[2])])
                for i in range(New_coordinates.shape[0]):
                    self.coordinates.append(New_coordinates[None,i])
                #Evaluate new nodale values:
                self.eval()
                newvalue = self(New_coordinates,torch.tensor([el_id,el_id,el_id]))
                self.train()
                # Initialise new nodale values
                for i in range(newvalue.shape[1]):
                    self.nodal_values[0].append(newvalue[None,0,i])
                    self.nodal_values[1].append(newvalue[None,1,i])

            case 'New_V2':
                connectivity_tensor = torch.tensor(self.connectivity, dtype = torch.int32)
                mask = (connectivity_tensor == nodes[0]) | (connectivity_tensor == nodes[1]) | (connectivity_tensor == nodes[2])
                row_counts = mask.sum(dim=1)
                elems_edges = torch.where(row_counts >= 2)[0]
                mask_el_id = elems_edges != el_id
                elems_edges = elems_edges[mask_el_id] 
                Coordinates_all = self.coordinates_all
                Coordinates_all[self.coord_free] = self.coordinates['free']
                Coordinates_all[~self.coord_free] = self.coordinates['imposed']
                Coord = Coordinates_all[nodes-1,:]

                # permutations = torch.tensor([np.where(np.isin(nodes,self.connectivity[elems_edges][i,:]))[0] for i in range(self.connectivity[elems_edges].shape[0])])
                permutations = torch.cat([torch.tensor(np.where(np.isin(nodes,self.connectivity[elems_edges][i,:]))[0])[None,:] for i in range(self.connectivity[elems_edges].shape[0])])
                if permutations.shape[0] < 3:
                    target_rows = torch.tensor([[0, 1], [1, 0]])
                    if not torch.any(torch.all(permutations == target_rows[:, None], dim=2)):
                        permutations = torch.cat([permutations, target_rows[0,:].unsqueeze(0)])
                    target_rows = torch.tensor([[0, 2], [2, 0]])
                    if not torch.any(torch.all(permutations == target_rows[:, None], dim=2)):
                        permutations = torch.cat([permutations, target_rows[0,:].unsqueeze(0)])
                    target_rows = torch.tensor([[1, 2], [2, 1]])
                    if not torch.any(torch.all(permutations == target_rows[:, None], dim=2)):
                        permutations = torch.cat([permutations, target_rows[0,:].unsqueeze(0)])

                # permutations = torch.tensor([[0, 1], [0, 2], [1, 2]])
                New_coordinates = 0.5 * (Coord[permutations[:, 0]] + Coord[permutations[:, 1]])
                #Evaluate new nodale values:
                self.eval()
                newvalue = self(New_coordinates,torch.tensor([el_id,el_id,el_id]))
                self.train()
                Coordinates_all_new = torch.cat([Coordinates_all,New_coordinates])
                new_coord_free = torch.ones(3,dtype = self.coord_free.dtype, device = self.coord_free.device) ==1
                if self.coord_free[nodes-1][permutations[0,0]] ==False and self.coord_free[nodes-1][permutations[0,1]] ==False:
                    new_coord_free[0] = False
                if self.coord_free[nodes-1][permutations[1,0]] ==False and self.coord_free[nodes-1][permutations[1,1]] ==False:
                    new_coord_free[1] = False
                if self.coord_free[nodes-1][permutations[2,0]] ==False and self.coord_free[nodes-1][permutations[2,1]] ==False:
                    new_coord_free[2] = False
                self.coord_free = torch.cat([self.coord_free,new_coord_free])


                d_coord_free =  Coordinates_all_new[self.coord_free,:].shape[0] - self.coordinates['free'].shape[0]
                d_coord_imposed =  Coordinates_all_new[~self.coord_free,:].shape[0] - self.coordinates['imposed'].shape[0]
                if d_coord_free> 0:
                    self.coordinates['free'] = torch.cat([self.coordinates['free'], Coordinates_all_new[self.coord_free,:][-d_coord_free:,:]])
                if d_coord_imposed> 0:
                    self.coordinates['imposed'] = torch.cat([self.coordinates['imposed'], New_coordinates[~new_coord_free,:]])
                    # self.coordinates['imposed'] = torch.cat([self.coordinates['imposed'], Coordinates_all_new[~self.coord_free,:][-d_coord_imposed:,:]])
                    self.coordinates['imposed'].requires_grad = False

                self.coordinates_all = Coordinates_all_new
                new_dofs_free_x = torch.ones(3,dtype = self.coord_free.dtype, device = self.coord_free.device) ==1
                new_dofs_free_y = torch.ones(3,dtype = self.coord_free.dtype, device = self.coord_free.device) ==1

                if self.dofs_free_x[nodes-1][permutations[0,0]] ==False and self.dofs_free_x[nodes-1][permutations[0,1]] ==False:
                    new_dofs_free_x[0] = False
                if self.dofs_free_x[nodes-1][permutations[1,0]] ==False and self.dofs_free_x[nodes-1][permutations[1,1]] ==False:
                    new_dofs_free_x[1] = False
                if self.dofs_free_x[nodes-1][permutations[2,0]] ==False and self.dofs_free_x[nodes-1][permutations[2,1]] ==False:
                    new_dofs_free_x[2] = False

                if self.dofs_free_y[nodes-1][permutations[0,0]] ==False and self.dofs_free_y[nodes-1][permutations[0,1]] ==False:
                    new_dofs_free_y[0] = False
                if self.dofs_free_y[nodes-1][permutations[1,0]] ==False and self.dofs_free_y[nodes-1][permutations[1,1]] ==False:
                    new_dofs_free_y[1] = False
                if self.dofs_free_y[nodes-1][permutations[2,0]] ==False and self.dofs_free_y[nodes-1][permutations[2,1]] ==False:
                    new_dofs_free_y[2] = False
                u = self.values
                u[self.dofs_free_x,0] = self.nodal_values['x_free']
                u[self.dofs_free_y,1] = self.nodal_values['y_free']
                u[~self.dofs_free_x,0] = self.nodal_values['x_imposed']                    
                u[~self.dofs_free_y,1] = self.nodal_values['y_imposed']
                self.dofs_free_x = torch.cat([self.dofs_free_x,new_dofs_free_x])
                self.dofs_free_y = torch.cat([self.dofs_free_y,new_dofs_free_y])
                self.values = torch.cat([u, newvalue.t()])
                dx_free = self.values[self.dofs_free_x,0].shape[0] - self.nodal_values['x_free'].shape[0]
                dy_free = self.values[self.dofs_free_y,0].shape[0] - self.nodal_values['y_free'].shape[0]
                dx_imposed = self.values[~self.dofs_free_x,0].shape[0] - self.nodal_values['x_imposed'].shape[0]
                dy_imposed = self.values[~self.dofs_free_y,0].shape[0] - self.nodal_values['y_imposed'].shape[0]
                if dx_free>0:
                    self.nodal_values['x_free'] = torch.cat([self.nodal_values['x_free'], self.values[self.dofs_free_x,0][-dx_free:]])
                if dy_free>0:
                    self.nodal_values['y_free'] = torch.cat([self.nodal_values['y_free'], self.values[self.dofs_free_y,1][-dy_free:]])
                if dx_imposed>0:
                    self.nodal_values['x_imposed'] = torch.cat([self.nodal_values['x_imposed'], self.values[~self.dofs_free_x,0][-dx_imposed:]])
                    self.nodal_values['x_imposed'].requires_grad = False

                if dy_imposed>0:
                    self.nodal_values['y_imposed'] = torch.cat([self.nodal_values['y_imposed'], self.values[~self.dofs_free_y,1][-dy_imposed:]])
                    self.nodal_values['y_imposed'].requires_grad = False

        NewNodes_indexes = np.max(self.connectivity) + np.array([1,2,3])
        match vers:
            case 'old':
                new_elem = np.array([   [NewNodes_indexes[0], NewNodes_indexes[1], nodes[0]],
                                        [nodes[1], NewNodes_indexes[2], NewNodes_indexes[0]],
                                        [NewNodes_indexes[2], nodes[2], NewNodes_indexes[1]],
                                        [NewNodes_indexes[2], NewNodes_indexes[1], NewNodes_indexes[0]]])
            case 'New_V2':

                permutations[torch.where((permutations == permutations[0,0])[1:,:])[0]+1,torch.where((permutations == permutations[0,0])[1:,:])[1]]



                Third_1 = torch.where((permutations == permutations[0,0])[1:,:])[0]+1
                Third_2 = torch.where((permutations == permutations[0,1])[1:,:])[0]+1
                Initial_3 = permutations[Third_2,torch.where(permutations[Third_2,:] != permutations[0,1])[1]]
                Third_3 = torch.where(torch.logical_and(torch.tensor([0,1,2]) != Third_1 , torch.tensor([0,1,2]) != Third_2 ))[0]
                new_elem = np.array([   [nodes[permutations[0,0]], NewNodes_indexes[0], NewNodes_indexes[Third_1]],
                                        [nodes[permutations[0,1]], NewNodes_indexes[0], NewNodes_indexes[Third_2]],
                                        [NewNodes_indexes[Third_2], nodes[Initial_3], NewNodes_indexes[Third_1]],
                                        [NewNodes_indexes[0], NewNodes_indexes[1], NewNodes_indexes[2]]])


        new_connectivity = self.connectivity
        new_generation = self.elements_generation
        new_det = self.detJ_0.numpy()
        new_current_det = self.detJ.numpy()
        # Remove splitted element
        new_connectivity = np.delete(new_connectivity,(el_id),axis = 0)
        curren_gen = new_generation[el_id]
        curren_det = new_current_det[el_id]
        new_generation = np.delete(new_generation,(el_id),axis = 0)
        new_det = np.delete(new_det,(el_id),axis = 0)
        new_current_det = np.delete(new_current_det,(el_id),axis = 0)

        # Update connectivity
        new_connectivity = np.vstack((new_connectivity,new_elem))
        new_generation = np.hstack((new_generation,np.repeat(np.array(curren_gen+1), 4, axis=None)))
        new_det = np.hstack((new_det,np.repeat(np.array(curren_det/4), 4, axis=None)))
        new_current_det = np.hstack((new_current_det,np.repeat(np.array(curren_det/4), 4, axis=None)))

        self.connectivity = new_connectivity
        self.elements_generation = new_generation
        self.detJ_0 = torch.tensor( new_det, 
                                    dtype = self.float_config.dtype, 
                                    device= self.float_config.device)
        self.detJ = torch.tensor( new_current_det, 
                                    dtype = self.float_config.dtype, 
                                    device= self.float_config.device)

        if self.order =='1':
            self.ElementBlock.UpdateConnectivity(self.connectivity)
            self.Interpolation.UpdateConnectivity(self.connectivity)
        elif self.order == '2':
            self.ElementBlock.UpdateConnectivity(self.connectivity)
            self.Interpolation.UpdateConnectivity(self.connectivity)
        self.NElem +=3
        Removed_elem_list = [el_id]
        match vers:
            case 'old':
                Edges = [elem_edge_1,elem_edge_2,elem_edge_3]
                nodes_edge = [[nodes[0],nodes[1]],[nodes[0],nodes[2]],[nodes[1],nodes[2]]]
                for i in range(len(Edges)):
                    edge = Edges[i]
                    # Need to be updated to new connectivity
                    node_edge = nodes_edge[i]
                    if edge.shape[0] == 1:
                        edge_new = edge - np.where(np.array(Removed_elem_list)<edge)[0].shape[0]
                        self.Split_hangingNodes(edge_new,node_edge,NewNodes_indexes[i])
                        Removed_elem_list.append(edge)
                    else:
                        self.coordinates[-(3-i)].requires_grad = False
                        if not (self.nodal_values[0][int(node_edge[0])-1].requires_grad and self.nodal_values[0][int(node_edge[1])-1].requires_grad):
                            self.nodal_values[0][-(3-i)].requires_grad = False
                        if not (self.nodal_values[1][int(node_edge[0])-1].requires_grad and self.nodal_values[1][int(node_edge[1])-1].requires_grad):
                            self.nodal_values[1][-(3-i)].requires_grad = False
            case 'New_V2':
                nodes_edge = nodes[permutations]
                for i in range(elems_edges.shape[0]):
                    edge = elems_edges[i]
                    node_edge = nodes_edge[i,:]
                    edge_new = edge.cpu().numpy() - np.where(np.array(Removed_elem_list)<edge.cpu().numpy())[0].shape[0]
                    self.Split_hangingNodes(edge_new,node_edge,NewNodes_indexes[i])
                    Removed_elem_list.append(edge[None])

        return Removed_elem_list

    def forward(self, x = 'NaN', el_id = 'NaN'):
        """
        The main forward pass of the mesh object.

        This function computes the interpolation based on the current mesh state. The behavior differs depending on the training mode (`self.training`).

        Args:
            self (object): The object itself.
            x (torch.Tensor, optional): Input tensor (defaults to 'NaN' and not required in training mode as the interpolation is performed in all elements).
            el_id (torch.Tensor, optional): Element ID tensor (defaults to 'NaN' and not required in training mode as the interpolation is performed in all elements).

        Returns:
            tuple: A tuple containing:
                - interpol (torch.Tensor): The interpolated values at the integration points.
                - x_g (torch.Tensor): The coordinates of the integration points. (Only returned during training)
                - detJ (torch.Tensor): The determinant of the Jacobian matrix. (Only returned during training)

        Notes:
            * During training (`self.training` is True):
                * `el_id` is generated internally if not provided.
                * Additional calculations are performed for `shape_functions`, `x_g`, and `detJ` using the `ElementBlock` function.
            * During evaluation (`self.training` is False):
                * Only `shape_functions` are calculated using `ElementBlock`.
        """
        if self.training:
            el_id = torch.arange(0,self.NElem,dtype=torch.int)
            shape_functions,x_g, detJ = self.ElementBlock(x, el_id, self.coordinates, self.nodal_values, self.coord_free,self.coordinates_all, self.training)
            interpol = self.Interpolation(x_g, el_id, self.nodal_values, shape_functions, self.relation_BC_node_IDs, self.relation_BC_normals, self.relation_BC_values, self.dofs_free_x,self.dofs_free_y,self.values,self.training)
        
            return interpol, x_g, detJ
        else:
            shape_functions = self.ElementBlock(x, el_id, self.coordinates, self.nodal_values, self.coord_free,self.coordinates_all, self.training)
            # shape_functions = self.ElementBlock(x, el_id, self.coordinates, self.nodal_values, False)
            # interpol = self.Interpolation(x, el_id, self.nodal_values, shape_functions, self.relation_BC_node_IDs, self.relation_BC_normals, self.relation_BC_values, False)
            interpol = self.Interpolation(x, el_id, self.nodal_values, shape_functions, self.relation_BC_node_IDs, self.relation_BC_normals, self.relation_BC_values, self.dofs_free_x,self.dofs_free_y,self.values, False)

            return interpol

    def UnFreeze_FEM(self):
        """This function unfreezes the nodal values that will be trainable during optimization. It uses the version string (`vers`) to switch the between the 'old'implementation
        and the more efficient 'New_V2'one.

            Args:
                self (object): The 2D space interpolation model.

            Returns:
                None (the function modifies the trainable flags of `self.nodal_values` in-place).

            Modifies:
                self.nodal_values (dict): A dictionary containing tensors of nodal values. The `requires_grad` attribute of specific tensors within the dictionary is modified.
        """        
        vers = 'new'
        if vers == 'new': 
            self.nodal_values['x_free'].requires_grad = True
            self.nodal_values['y_free'].requires_grad = True
            self.nodal_values['x_imposed'].requires_grad = False
            self.nodal_values['y_imposed'].requires_grad = False
        else:
            for dim in self.nodal_values:
                for val in dim:
                    val.requires_grad = True
    
            for j in range(len(self.frozen_BC_node_IDs)):
                # print("component ", self.frozen_BC_component_IDs[j], " : ", self.frozen_BC_node_IDs[j][0:5])
                # print("excluded : ", self.ExcludeFromDirichlet)
                values = self.nodal_values[self.frozen_BC_component_IDs[j]]
                frozen = self.frozen_BC_node_IDs[j]
                for idf in frozen:
                    if idf not in self.ExcludeFromDirichlet:
                        values[idf].requires_grad = False
                    else:
                        values[idf].requires_grad = True

    def Freeze_FEM(self):
        """
        This function prevents any modification of nodal values during optimisation. It uses the version string (`vers`) to switch the between the 'old'implementation
        and the more efficient 'New_V2'one.

        Args:
            self (object): The 2D space interpolation model.

        Returns:
            None (the function modifies the trainable flags of `self.nodal_values` in-place).

        Modifies:
            self.nodal_values (dict): A dictionary containing tensors of node values. The `requires_grad` attribute of all tensors within the dictionary is set to False.
        """
        vers = 'New_V2'
        match vers:
            case 'old':
                for dim in self.nodal_values:
                    for val in dim:
                        val.requires_grad = False
            case 'New_V2':
                self.nodal_values['x_free'].requires_grad = False
                self.nodal_values['y_free'].requires_grad = False

      
    def Freeze_Mesh(self):
        """
        This function prevents any modification of node coordinates during optimisation. It uses the version string (`vers`) to switch the between the 'old'implementation
        and the more efficient 'New_V2'one.

        Args:
            self (object): The 2D space interpolation model.

        Returns:
            None (the function modifies the trainable flags of `self.coordinates` in-place).

        Modifies:
            self.coordinates (dict): A dictionary containing tensors of node coordinates. The `requires_grad` attribute of all tensors within the dictionary is set to False.
        """
        vers = 'new_V2'
        if vers == 'new_V2':
            self.coordinates['free'].requires_grad = False
            self.coordinates['imposed'].requires_grad = False
        else:
            for param in self.coordinates:
                param.requires_grad = False
    
    def UnFreeze_Mesh(self):
        """This function unfreezes the nodes in the mesh that will be trainable during optimization. It uses the version string (`vers`) to switch the between the 'old'implementation
        and the more efficient 'New_V2'one.

            Args:
                self (object): The 2D space interpolation model.

            Returns:
                None (the function modifies the trainable flags of `self.coordinates` in-place).

            Modifies:
                self.coordinates (dict): A dictionary containing tensors of node coordinates. The `requires_grad` attribute of specific tensors within the dictionary is modified.
        """
        vers = 'new_V2'
        if vers == 'new_V2':
            self.coordinates['free'].requires_grad = True
            self.coordinates['imposed'].requires_grad = False

        else:
            for param in self.coordinates:
                param.requires_grad = True
            border_nodes = torch.unique(torch.tensor(self.borders_nodes, dtype=torch.int))-1
            for node in border_nodes:
                self.coordinates[node].requires_grad = False

    def CheckBCValues(self):
        """Set the coordinates as trainable parameters """
        print("Unfreeze values")

        for j in range(len(self.frozen_BC_node_IDs)):
            # print(j, self.frozen_BC_component_IDs[j], self.frozen_BC_node_IDs[j])
            values = self.nodal_values[self.frozen_BC_component_IDs[j]]
            frozen = self.frozen_BC_node_IDs[j]
            # for idf in frozen:
                # print(values[idf])


    def Update_Middle_Nodes(self, mesh):
        """
        Updates the coordinates of the middle nodes in a mesh structure.

        This function takes a mesh object as input and modifies the coordinates of the middle nodes (nodes 4, 5, and 6) of each cell. The new coordinates are calculated as the average of the coordinates of the two neighboring nodes.

        Args:
            mesh (object): A mesh object containing connectivity information and node coordinates.

        Returns:
            None

        Modifies:
            self.coordinates (torch.Tensor): The tensor containing the coordinates of all nodes in the mesh. The coordinates of the middle nodes are updated in-place.
        """
        cell_nodes_IDs = mesh.Connectivity
        node1_coord =  torch.cat([self.coordinates[int(row)-1] for row in cell_nodes_IDs[:,0]])
        node2_coord =  torch.cat([self.coordinates[int(row)-1] for row in cell_nodes_IDs[:,1]])
        node3_coord =  torch.cat([self.coordinates[int(row)-1] for row in cell_nodes_IDs[:,2]])

        T6_Coord1 = torch.nn.Parameter(node1_coord*0.5 + node2_coord*0.5)
        T6_Coord2 = node2_coord*0.5 + node3_coord*0.5
        T6_Coord3 = node1_coord*0.5 + node3_coord*0.5

        for j in range(len(cell_nodes_IDs)):
            self.coordinates[int(cell_nodes_IDs[j,3])-1] = T6_Coord1[j].unsqueeze(0)
            self.coordinates[int(cell_nodes_IDs[j,4])-1] = T6_Coord2[j].unsqueeze(0)
            self.coordinates[int(cell_nodes_IDs[j,5])-1] = T6_Coord3[j].unsqueeze(0)

    def ComputeNormalVectors(self):
        print()
        print(" * ComputeNormalVectors")
        line_normals = []

        if len(self.relation_BC_node_IDs)==0:
            return []
        else:

            for line in self.relation_BC_lines[0]:
                point_a = line[0]
                point_b = line[1]
                coord_a = self.coordinates[point_a-1]
                coord_b = self.coordinates[point_b-1]
                vect = coord_b - coord_a
                vect = vect[0,[1,0]]
                vect[0] = -vect[0]
                vect = vect/torch.norm(vect)
                line_normals.append(vect)

            normals = [[] for i in range(len(self.relation_BC_node_IDs[0])) ]

            for i in range(len(self.relation_BC_node_IDs[0])):
                node_id = self.relation_BC_node_IDs[0][i]

                for j in range(len(self.relation_BC_lines[0])): 
                    if node_id+1 in self.relation_BC_lines[0][j]:
                        normals[i].append(line_normals[j])

            normals = [(x[0]+x[1])/2 for x in normals]
            self.relation_BC_normals.append(normals)

    # def UnFreeze_Mesh(self):
    #     """Set the coordinates as trainable parameters"""
    #     for param in self.coordinates:
    #         param.requires_grad = True
    #     border_nodes = torch.unique(torch.tensor(self.borders_nodes, dtype=torch.int))-1
    #     for node in border_nodes:
    #         self.coordinates[node].requires_grad = False




def GetRefCoord(x,y,x1,x2,x3,y1,y2,y3):
    inverse_matrix2 = torch.ones([int(y.shape[0]), 3, 3], dtype=x.dtype, device=x.device)
    denominator = (x1*y2 - x1*y3 - x2*y1 + x2*y3 + x3*y1 - x3*y2)
    inverse_matrix2[:,0,0] = (y2 - y3)/denominator
    inverse_matrix2[:,1,0] = (x3 - x2)/denominator
    inverse_matrix2[:,2,0] = (-x3*y2 + x2*y3)/denominator

    inverse_matrix2[:,0,1] = (-y1 + y3)/denominator
    inverse_matrix2[:,1,1] = (x1 - x3)/denominator
    inverse_matrix2[:,2,1]= (x3*y1 - x1*y3)/denominator

    inverse_matrix2[:,0,2] = (y1 - y2)/denominator
    inverse_matrix2[:,1,2] = (-x1 + x2)/denominator
    inverse_matrix2[:,2,2] = (-x2*y1 + x1*y2)/denominator
    x_extended = torch.stack((x,y, torch.ones_like(y)),dim=1)
    if len(x_extended.shape)<3:
        x_extended = x_extended.unsqueeze(1)
    return torch.einsum('eij,ei->ej',inverse_matrix2,x_extended.squeeze(1))



def GetRefCoord_1D(x,x1,x2):

    return 2*(x-x1)/(x2-x1)-1


def GetRealCoord_1D(xi,x1,x2):

    return x1 + (xi+1)*(x2-x1)/2


class InterpolationBlock1D_Lin(nn.Module):
    
    def __init__(self, connectivity):
       
        super(InterpolationBlock1D_Lin, self).__init__()
        self.connectivity = connectivity.astype(int)
    
    def UpdateConnectivity(self,connectivity):
        self.connectivity = connectivity.astype(int)

    def forward(self, x, cell_id, nodal_values, shape_functions):

        cell_nodes_IDs = self.connectivity[cell_id,:] - 1
        if cell_nodes_IDs.ndim == 1:
            cell_nodes_IDs = np.expand_dims(cell_nodes_IDs,0)

        node1_value = torch.cat([nodal_values[row] for row in cell_nodes_IDs[:,0]]) 
        node2_value = torch.cat([nodal_values[row] for row in cell_nodes_IDs[:,1]]) 

        prod = [shape_functions[:,0,k]*node1_value + shape_functions[:,1,k]*node2_value for k in range(shape_functions.shape[2])]
        prod = torch.stack(prod, dim=1)

        # print("shape_functions = ", shape_functions.shape)
        # print("prod = ", prod.shape)
        # print("__________________________________")
        # print()

        return prod



class ElementBlock1D_Lin(nn.Module):
    """
    Returns:
         N_i(x)'s for each nodes within each element"""
    def __init__(self, connectivity, n_integr_points):
        """ Initialise the Linear Bar element 
        Args:
            connectivity (Interger table): Connectivity matrix of the 1D mesh
        """
        super(ElementBlock1D_Lin, self).__init__()
        self.connectivity = connectivity.astype(int)
        self.n_integr_points = n_integr_points

    def UpdateConnectivity(self,connectivity):
        self.connectivity = connectivity.astype(int)

    def GP(self, n_integr_points):
        if n_integr_points ==1:
            return torch.tensor([[0.0]],dtype=torch.float64, requires_grad=True) # coordinates in ref. element
        elif n_integr_points ==2:
            return torch.tensor([[-1/np.sqrt(3.0),1/np.sqrt(3.0)]],dtype=torch.float64, requires_grad=True)
        elif n_integr_points ==3:
            return torch.tensor([[-np.sqrt(3/5), 0, np.sqrt(3/5)]],dtype=torch.float64, requires_grad=True)
        elif n_integr_points ==4:
            return torch.tensor([[  -np.sqrt((3+2*np.sqrt(6/5))/7), 
                                    -np.sqrt((3-2*np.sqrt(6/5))/7), 
                                    np.sqrt((3-2*np.sqrt(6/5))/7),
                                    np.sqrt((3+2*np.sqrt(6/5))/7)]],dtype=torch.float64, requires_grad=True)
        elif n_integr_points ==5:
            return torch.tensor([[  0,
                                    -1/3*np.sqrt(5-2*np.sqrt(10/7)), 
                                    1/3*np.sqrt(5-2*np.sqrt(10/7)), 
                                    -1/3*np.sqrt(5+2*np.sqrt(10/7)),
                                    1/3*np.sqrt(5+2*np.sqrt(10/7))]],dtype=torch.float64, requires_grad=True)


    def forward(self, x, cell_id, coordinates, nodal_values,flag_training):
        """ This is the forward function of the Linear element block. Note that to prevent extrapolation outside of the structure's geometry, 
        phantom elements are used to cancel out the interpolation shape functions outside of the beam.
        Those phantom elements are flagged with index -1
        """

        cell_nodes_IDs = self.connectivity[cell_id,:]
        if cell_nodes_IDs.ndim == 1:
            cell_nodes_IDs = np.expand_dims(cell_nodes_IDs,0)

        node1_coord =  torch.cat([coordinates[row-1] for row in cell_nodes_IDs[:,0]])
        node2_coord =  torch.cat([coordinates[row-1] for row in cell_nodes_IDs[:,1]])
        
        if flag_training:
            refCoordg = self.GP(self.n_integr_points).repeat(cell_id.shape[0],1)
                                   # Gauss weight
            if self.n_integr_points ==1:
                w_g = torch.tensor([2]).repeat(cell_id.shape[0],1)
            elif self.n_integr_points ==2:   
                w_g = torch.tensor([1,1]).repeat(cell_id.shape[0],1) 
            elif self.n_integr_points ==3:   
                w_g = torch.tensor([5/9,8/9,5/9]).repeat(cell_id.shape[0],1) 
            elif self.n_integr_points ==4:   
                w_g = torch.tensor([(18-np.sqrt(30))/36, (18+np.sqrt(30))/36, (18+np.sqrt(30))/36, (18-np.sqrt(30))/36]).repeat(cell_id.shape[0],1) 
            elif self.n_integr_points ==5:    
                w_g = torch.tensor([128/225,
                                    (322+13*np.sqrt(70))/900, (322+13*np.sqrt(70))/900, 
                                    (322-13*np.sqrt(70))/900,(322-13*np.sqrt(70))/900]).repeat(cell_id.shape[0],1)                                                                                          

            x_g = GetRealCoord_1D(refCoordg,node1_coord,node2_coord)
            refCoord = GetRefCoord_1D(x_g, node1_coord,node2_coord)
            N = torch.stack((-0.5*refCoord + 0.5, 0.5*refCoord + 0.5),dim=1) #.view(sh_R.shape[0],-1) # Left | Right | Middle
            detJ = (node2_coord - node1_coord)/2

            # print("refCoordg = ", refCoordg.shape)
            # print("x_g = ", x_g.shape, x_g[2])
            # print("nodes = ", node1_coord[-1], node2_coord[-1])
            # print("refCoord = ", refCoord[-1])
            # print("N = ", N.shape, N[2,0,:], N[2,1,:])
            # print("detJ = ", detJ[-1])
            # print()
            # print("detJ*w_g = ",  (detJ*w_g).shape)

            # print()

            return N, x_g, detJ*w_g

        else:
            if len(x.shape)==1:
                refCoord = GetRefCoord_1D(x.unsqueeze(1), node1_coord,node2_coord)
            else: 
                refCoord = GetRefCoord_1D(x, node1_coord,node2_coord)

            N = torch.stack((-0.5*refCoord + 0.5, 0.5*refCoord + 0.5),dim=1) #.view(sh_R.shape[0],-1) # Left | Right | Middle
            
            # print("cell_id = ", cell_id[-2])
            # print("x = ", x.shape, x[-2])
            # print("nodes = ", node1_coord[-2], node2_coord[-2])
            # print("refCoord = ", refCoord.shape, refCoord[-2])
            # print("N = ", N.shape, N[2,0,:], N[2,1,:])
            # print()
            return N

class MeshNN_1D(nn.Module):
    """ This class is a space HiDeNN building a Finite Element (FE) interpolation over the space domain. 
    The coordinates of the nodes of the underlying mesh are trainable. Those coordinates are passed as a List of Parameters to the subsequent sub-neural networks
    Updating those parameters correspond to r-adaptativity
    The Interpolation layer weights correspond to the nodal values. Updating them 
    is equivqlent to solving the PDE. """


    def __init__(self, mesh, n_integr_points):
        super(MeshNN_1D, self).__init__()
        self.register_buffer('float_config',torch.tensor([0.0])  )                                                     # Keep track of device and dtype used throughout the model

        self.version = "Gauss_quadrature"
        self.mesh = mesh
        self.n_integr_points = n_integr_points
        # if self.n_integr_points == 0:
        #     self.Mixed = True
        n_components = 1

        self.coordinates = nn.ParameterList([nn.Parameter(torch.tensor([mesh.Nodes[i][1:int(mesh.dimension)+1]],dtype=torch.float64)) \
                                             for i in range(len(mesh.Nodes))])

        print("mesh.NNodes = ", mesh.NNodes)
        # self.values = 0.001*torch.randint(low=-100, high=100, size=(mesh.NNodes,1))

        self.values =0.1*torch.ones((mesh.NNodes,1))

        self.connectivity = mesh.Connectivity
        self.borders_nodes = mesh.borders_nodes
        print("border_nodes : ", self.borders_nodes)

        self.dofs = mesh.NNodes*mesh.dim # Number of Dofs
        self.NElem = mesh.NElem
        self.order = mesh.order
        print("self.order = ", self.order)
        self.frozen_BC_node_IDs = []
        self.frozen_BC_component_IDs = []
        self.DirichletBoundaryNodes = mesh.DirichletBoundaryNodes
        self.ListOfDirichletsBCsValues = mesh.ListOfDirichletsBCsValues

        self.ListOfDirichletsBCsNormals = mesh.ListOfDirichletsBCsNormals
        print("mesh.ListOfDirichletsBCsValues = ", mesh.ListOfDirichletsBCsValues)

        if mesh.NoBC==False:
            for i in range(len(mesh.ListOfDirichletsBCsValues)):
                IDs = torch.tensor(mesh.DirichletBoundaryNodes[i], dtype=torch.int)
                print("IDs = ", IDs)
                IDs = torch.unique(IDs.reshape(IDs.shape[0],-1))-1
                print("IDs = ", IDs)
                self.frozen_BC_node_IDs.append(IDs)
                self.frozen_BC_component_IDs.append(mesh.ListOfDirichletsBCsNormals[i])
                self.values[IDs,0] = mesh.ListOfDirichletsBCsValues[i]

        self.nodal_values = nn.ParameterList([nn.Parameter(torch.tensor([i[0]])) for i in self.values])

        if self.order == '1':
            self.ElementBlock = ElementBlock1D_Lin(mesh.Connectivity, self.n_integr_points)
            self.Interpolation = InterpolationBlock1D_Lin(mesh.Connectivity)

    def SetBCs(self, ListOfDirichletsBCsValues):
        for i in range(len(ListOfDirichletsBCsValues)):
            IDs = torch.tensor(self.DirichletBoundaryNodes[i], dtype=torch.int)
            IDs = torch.unique(IDs.reshape(IDs.shape[0],-1))-1
            self.frozen_BC_node_IDs.append(IDs)
            self.frozen_BC_component_IDs.append(self.ListOfDirichletsBCsNormals[i])
            self.values[IDs,self.ListOfDirichletsBCsNormals[i]] = ListOfDirichletsBCsValues[i]

    def ZeroOut(self):
        self.nodal_values = nn.ParameterList([nn.Parameter(0*torch.tensor([i[0]])) for i in self.values])


    def UnFreeze_FEM(self):
        """Set the coordinates as trainable parameters """
        # print("Unfreeze values")

        for val in self.nodal_values:
            val.requires_grad = True

        for j in range(len(self.frozen_BC_node_IDs)):
            frozen = self.frozen_BC_node_IDs[j]
            print("frozen : ", frozen)
            self.nodal_values[frozen].requires_grad = False

    def SetFixedValues(self, node, val_inter):
        """Set the coordinates as trainable parameters """
        # print("Unfreeze values")


        for val in self.nodal_values:
            val.data = torch.tensor([0])
        ids = []
        for i in range(self.NElem+1):
            if i not in self.frozen_BC_node_IDs:
                ids.append(i)
        self.nodal_values[ids[node]].data = torch.tensor([val_inter])


    def Freeze_FEM(self):
        """Set the coordinates as untrainable parameters"""
        for val in self.nodal_values:
            val.requires_grad = False

    def Freeze_Mesh(self):
        """Set the coordinates as untrainable parameters"""
        for param in self.coordinates:
            param.requires_grad = False
    
    def UnFreeze_Mesh(self):
        """Set the coordinates as trainable parameters"""

        self.original_coordinates = [self.coordinates[i].data.item() for i in range(len(self.coordinates))]
        for param in self.coordinates:
            param.requires_grad = True

        border_nodes = torch.unique(torch.tensor(self.borders_nodes, dtype=torch.int))-1
        for node in border_nodes:
            self.coordinates[node].requires_grad = False

    def Init_from_previous(self,CoarseModel):
        try:
             CoarseModel.float_config.dtype
        except:
            CoarseModel.register_buffer('float_config',torch.tensor([0.0],dtype = torch.float64)  )                                                     # Keep track of device and dtype used throughout the model
            # CoarseModel.float_config = torch.tensor([0],dtype = torch.float64)
        newcoordinates = [coord for coord in self.coordinates]
        newcoordinates = torch.cat(newcoordinates,dim=0)
        IDs_newcoord = torch.tensor(CoarseModel.mesh.GetCellIds(newcoordinates),dtype=torch.int)
        NewNodalValues = CoarseModel(newcoordinates.to(CoarseModel.float_config.dtype),IDs_newcoord).to(self.float_config.dtype)
        # check if a cell ID was not found for some new nodes 
        if -1 in IDs_newcoord:
            index_neg = (IDs_newcoord == -1).nonzero(as_tuple=False)
            oldcoordinates = [coord for coord in CoarseModel.coordinates]
            oldcoordinates = torch.cat(oldcoordinates,dim=0)
            for ind_neg in index_neg:
                not_found_coordinates = newcoordinates[ind_neg]
                dist_vect = not_found_coordinates - oldcoordinates
                dist = torch.norm(dist_vect, dim=1)
                closest_old_nodal_value = dist.topk(1, largest=False)[1]
                NewNodalValues[0][ind_neg] = CoarseModel.nodal_values_x[closest_old_nodal_value].to(self.float_config.dtype)
                # NewNodalValues[1][ind_neg] = CoarseModel.nodal_values_y[closest_old_nodal_value].type(torch.float64)
        new_nodal_values_x = nn.ParameterList([nn.Parameter((torch.tensor([i[0]]))) for i in NewNodalValues])
        # new_nodal_values_y = nn.ParameterList([nn.Parameter(torch.tensor([i[1]])) for i in NewNodalValues.t()])
        # new_nodal_values = [new_nodal_values_x,new_nodal_values_y]
        self.nodal_values = new_nodal_values_x
        # self.nodal_values_y = new_nodal_values_y
        # self.nodal_values = new_nodal_values
    
    def forward(self,x = 'NaN', el_id = 'NaN'):
        if self.training:
            el_id = torch.arange(0,self.NElem,dtype=torch.int)
            shape_functions,x_g, detJ = self.ElementBlock(x, el_id, self.coordinates, self.nodal_values, self.training)
            interpol = self.Interpolation(x_g, el_id, self.nodal_values, shape_functions)

            return interpol, x_g, detJ
        else:
            shape_functions = self.ElementBlock(x, el_id, self.coordinates, self.nodal_values, False)
            interpol = self.Interpolation(x, el_id, self.nodal_values, shape_functions)

            return interpol



#%% 3D interpolation

class ElementBlock3D_Lin(nn.Module):
    """
    Returns:
         N_i(x)'s for each nodes within each element"""
    def __init__(self, connectivity):
        """ Initialise the Linear Bar element 
        Args:
            connectivity (Interger table): Connectivity matrix of the 3D mesh
        """
        super(ElementBlock2D_Lin, self).__init__()
        self.connectivity = connectivity.astype(int)
        self.register_buffer('GaussPoint',self.GP())
    def UpdateConnectivity(self,connectivity):
        self.connectivity = connectivity.astype(int)

    def GP(self):
        gp =  torch.tensor([  [0.5854101966249685, 0.1381966011250105, 0.1381966011250105], 
                [0.1381966011250105, 0.5854101966249685, 0.1381966011250105], 
                [0.1381966011250105, 0.1381966011250105, 0.5854101966249685], 
                [0.1381966011250105, 0.1381966011250105, 0.1381966011250105]],dtype=torch.float64, requires_grad=True) # a1, a2, a3  the 3 volume coordinates
        # Add 4th volume coordinate from first 3
        gp = torch.hstack([gp, 1-torch.sum(gp, dim=1)[:,None]]) # shape [gxn]
        return gp

    def forward(self, x, cell_id, coordinates, nodal_values,coord_mask,coordinates_all,flag_training):
        """ This is the forward function of the Linear element block that outputs 3D linear shape functions based on
        args:
            - x (tensor) : position where to evalutate the shape functions
            - cell_id (interger) : Associated element
            - coordinates (np array): nodal coordinates array
            - coord_mask (boolean tensor) : mask for free nodes in the mesh
            - coordinates_all prealocated tensor of all coordinates
        """

        cell_nodes_IDs = self.connectivity[cell_id,:]
        if cell_nodes_IDs.ndim == 1:
            cell_nodes_IDs = np.expand_dims(cell_nodes_IDs,0)

        coordinates_all = torch.ones_like(coordinates_all)
        coordinates_all[coord_mask] = coordinates['free']
        coordinates_all[~coord_mask] = coordinates['imposed']
        Ids = torch.as_tensor(cell_nodes_IDs-1).to(coordinates_all.device).t()[:,:,None]
        nodes_coord =  torch.gather(coordinates_all[None,:,:].repeat(4,1,1),1, Ids.repeat(1,1,3)) # for each element, 4 nodes with 3 coordonates 

        if flag_training:

            refCoordg = self.GaussPoint.repeat(cell_id.shape[0],1) # Computes N_i(xg local) for each gp for each element, shape [exgxn]

            w_g = 1/24                                              # Gauss weight

            Ng = refCoordg

            x_g = torch.einsum('nex,egn->egx',nodes_coord,Ng)

            refCoord = GetRefCoord_3D(x_g[:,0],x_g[:,1],nodes_coord[0,:,0],nodes_coord[1,:,0],nodes_coord[2,:,0],nodes_coord[0,:,1],nodes_coord[1,:,1],nodes_coord[2,:,1])

            N = refCoord

            ##### I stopped here, compute det of tet using broadcast 
        #     detJ = (nodes_coord[0,:,0] - nodes_coord[2,:,0])*(nodes_coord[1,:,1] - nodes_coord[2,:,1]) - (nodes_coord[1,:,0] - nodes_coord[2,:,0])*(nodes_coord[0,:,1] - nodes_coord[2,:,1])
            
        #     return N,x_g, detJ*w_g

        # else:
        #     match vers:
        #         case 'old':
        #             refCoord = GetRefCoord(x[:,0],x[:,1],node1_coord[:,0],node2_coord[:,0],node3_coord[:,0],node1_coord[:,1],node2_coord[:,1],node3_coord[:,1])
        #         case 'new'| 'new_V2':
        #             refCoord = GetRefCoord(x[:,0],x[:,1],nodes_coord[0,:,0],nodes_coord[1,:,0],nodes_coord[2,:,0],nodes_coord[0,:,1],nodes_coord[1,:,1],nodes_coord[2,:,1])
        #     out = torch.stack((refCoord[:,0], refCoord[:,1], refCoord[:,2]),dim=1) #.view(sh_R.shape[0],-1) # Left | Right | Middle
        #     return out