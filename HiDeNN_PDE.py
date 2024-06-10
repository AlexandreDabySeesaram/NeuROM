#%% Libraries import
import time 
# Import pre-processing functions
import Bin.Pre_processing as pre
# Import mechanical functions
from Bin.PDE_Library import RHS, PotentialEnergyVectorised, \
        Derivative, AnalyticGradientSolution, AnalyticSolution, GramSchmidt
# Import torch librairies
import torch
import torch.nn as nn
torch.set_default_dtype(torch.float32)
#Import post processing libraries
import Post.Plots as Pplot
import matplotlib.pyplot as plt
mps_device = torch.device("mps")
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
            x_left = torch.cat([x_left_0,x_left_2])
            x_right = torch.cat([x_right_0,x_right_2])
        else:
            x_left = torch.cat([coordinates[row-1] for row in self.connectivity[i,0]])
            x_right = torch.cat([coordinates[row-1] for row in self.connectivity[i,-2]])
            x_mid = torch.cat([coordinates[row-1] for row in self.connectivity[i,-1]])

        sh_mid_1 = self.LinearBlock(x, x_left, x_right, torch.tensor([0]), x_right - x_left)
        sh_mid_2 = self.LinearBlock(x, x_left, x_right, x_right - x_left, torch.tensor([0]))    
        sh_mid = -(sh_mid_1*sh_mid_2)/((x_mid -x_left)*(x_mid - x_right)).T

        sh_R_1 = self.LinearBlock(x, x_left, x_right, x_mid - x_left, x_mid - x_right)
        sh_R_2 = self.LinearBlock(x, x_left, x_right, x_right - x_left,  torch.tensor([0])) 
        sh_R = (sh_R_1*sh_R_2)/((x_left-x_mid)*(x_left - x_right)).T

        sh_L_1 = self.LinearBlock(x, x_left, x_right,  torch.tensor([0]), x_right - x_left)
        sh_L_2 = self.LinearBlock(x, x_left, x_right, x_left - x_mid, x_right - x_mid)
        sh_L = (sh_L_1*sh_L_2)/((x_right-x_left)*(x_right - x_mid)).T

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
        
    def forward(self, x, coordinates, i):
        """ This is the forward function of the Linear element block. Note that to prevent extrapolation outside of the structure's geometry, 
        phantom elements are used to cancel out the interpolation shape functions outside of the beam.
        Those phantom elements are flagged with index -1
        """

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
    """ This class is a space HiDeNN building a Finite Element (FE) interpolation over the space domain. 
    The coordinates of the nodes of the underlying mesh are trainable. Those coordinates are passed as a List of Parameters to the subsequent sub-neural networks
    Updating those parameters correspond to r-adaptativity
    The Interpolation layer weights correspond to the nodal values. Updating them 
    is equivqlent to solving the PDE. """
    def __init__(self, mesh):
        super(MeshNN, self).__init__()
        self.version = "Trapezoidal"

        self.coordinates = nn.ParameterList([nn.Parameter(torch.tensor([[mesh.Nodes[i][1]]])) \
                                             for i in range(len(mesh.Nodes))])
        self.dofs = mesh.NNodes*mesh.dim # Number of Dofs
        self.NElem = mesh.NElem

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
        self.NodalValues_uu = nn.Parameter(data=0.1*torch.ones(self.dofs-self.NBCs), requires_grad=False)
        self.InterpoLayer_uu.weight.data = self.NodalValues_uu
        # self.InterpoLayer_uu.weight.data = self.NodalValues_uu*torch.randn_like(self.NodalValues_uu)
 
        self.AssemblyLayer = nn.Linear(2*(self.NElem+2),self.dofs)
        self.AssemblyLayer.weight.data = torch.tensor(mesh.weights_assembly_total,dtype=torch.float32).clone().detach()
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
        newcoordinates = torch.cat(newcoordinates,dim=0)
        NewNodalValues = previous_model(newcoordinates)
        self.InterpoLayer_uu.weight.data = NewNodalValues[2:,0]

    def SetBCs(self,u_d):
        """Set the two Dirichlet boundary conditions
        Args:
            u_d (Float list): The left and right BCs"""

        self.u_0 = torch.tensor(u_d[0], dtype=torch.float32)
        self.u_L = torch.tensor(u_d[1], dtype=torch.float32)
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
        # Interpolation (nodal values) layer
        # self.NodalValues_para = nn.Parameter(data=torch.linspace(self.mu_min,self.mu_max,self.N_mu).pow(-1), requires_grad=False)
        self.NodalValues_para = nn.Parameter(data=torch.ones(self.N_mu), requires_grad=False)  
        self.InterpoLayer = nn.Linear(self.N_mu,1,bias=False)
        # Initialise with linear mode
        # self.InterpoLayer.weight.data = self.NodalValues_para
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

    def Init_from_previous(self,previous_model):
        newparacoordinates = [coord for coord in self.coordinates]
        newparacoordinates = torch.cat(newparacoordinates,dim=0)
        self.InterpoLayer.weight.data = previous_model(newparacoordinates).T

class NeuROM(nn.Module):
    """This class builds the Reduced-order model from the interpolation NN for space and parameters space"""
    def __init__(self, mesh, ParametersList, config, n_modes_ini = 1, n_modes_max = 100):
        super(NeuROM, self).__init__()
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
        self.training = True
        for i in range(self.n_modes_truncated):
            self.Space_modes[i].train() 

    def eval(self):
        self.training = False
        for i in range(self.n_modes_truncated):
            self.Space_modes[i].eval()  

    def TrainingParameters(self, loss_decrease_c = 1e-7,Max_epochs = 1000, learning_rate = 0.001):
        self.loss_decrease_c = loss_decrease_c
        self.Max_epochs = Max_epochs
        self.learning_rate = learning_rate


    def FreezeAll(self):
        """This method allows to freeze all sub neural networks"""
        self.Freeze_Mesh()
        self.Freeze_Space()
        self.Freeze_MeshPara()
        self.Freeze_Para()
    
    def AddMode(self):
        """This method allows to freeze the already computed modes and free the new mode when a new mode is required"""
        self.n_modes_truncated += 1  # Increment the number of modes used in the truncated tensor decomposition
        self.FreezeAll()
        self.Space_modes[self.n_modes_truncated-1].UnFreeze_FEM()
        self.Space_modes[self.n_modes_truncated-1].InterpoLayer_uu.weight.data = 0*self.Space_modes[self.n_modes_truncated-1].NodalValues_uu

        for j in range(self.n_para):
            self.Para_modes[self.n_modes_truncated-1][j].UnFreeze_FEM()  

    def AddMode2Optimizer(self,optim):
        "This method adds the newly freed parameters to the optimizer"
        New_mode_index = self.n_modes_truncated-1
        Space = self.Space_modes[self.n_modes_truncated-1].parameters()
        Para = self.Para_modes[self.n_modes_truncated-1][:].parameters()
        optim.add_param_group({'params': Space})
        optim.add_param_group({'params': Para})

    def UnfreezeTruncated(self):
        for i in range(self.n_modes_truncated):
            self.Space_modes[i].UnFreeze_FEM()  
        
        if self.IndexesNon0BCs:
            "Keep first modes frozen so that BCs are well accounted for if non-homogeneous"
            for i in range(1,self.n_modes_truncated):
                for j in range(self.n_para):
                    self.Para_modes[i][j].UnFreeze_FEM()
        else:
            for i in range(1,self.n_modes_truncated):
                for j in range(self.n_para):
                    self.Para_modes[i][j].UnFreeze_FEM()

    def Freeze_Mesh(self):
        """Set the space coordinates as untrainable parameters"""
        for i in range(self.n_modes):
            self.Space_modes[i].Freeze_Mesh()

    def UnFreeze_Mesh(self):
        """Set the space coordinates as trainable parameters"""
        for i in range(self.n_modes_truncated):
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
                            u_k = self.Space_modes[i](torch.tensor(x),IDs_elems)
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
                            IDs_elems = torch.tensor(self.Space_modes[i].mesh.GetCellIds(x),dtype=torch.int)
                            u_k = self.Space_modes[i](torch.tensor(x),IDs_elems)
                            Space_modes.append(u_k)
                        u_i = torch.stack(Space_modes,dim=2)
                        P1 = (Para_modes[0].view(self.n_modes_truncated,Para_modes[0].shape[1])).to(torch.float64)
                        P2 = (Para_modes[1].view(self.n_modes_truncated,Para_modes[1].shape[1])).to(torch.float64)
                        out = torch.einsum('xyk,kj,kp->xyjp',u_i,P1,P2)
        return out
    def Init_from_previous(self,PreviousFullModel):
        import os
        if os.path.isfile(PreviousFullModel):
            BeamROM_coarse = torch.load(PreviousFullModel) # To load a full coarse model
            self.n_modes_truncated = min(BeamROM_coarse.n_modes_truncated-1,self.n_modes)
            Nb_modes_coarse = BeamROM_coarse.n_modes_truncated
            Nb_parameters_fine = len(self.Para_modes[0])
            Nb_parameters_coarse = len(BeamROM_coarse.Para_modes[0])
            self.n_modes_truncated
            for mode in range(self.n_modes_truncated):
                self.Space_modes[mode].Init_from_previous(BeamROM_coarse.Space_modes[mode])
                # newcoordinates = [coord for coord in self.Space_modes[mode].coordinates]
                # newcoordinates = torch.cat(newcoordinates,dim=0)
                # NewNodalValues = BeamROM_coarse.Space_modes[mode](newcoordinates)
                # self.Space_modes[mode].InterpoLayer_uu.weight.data = NewNodalValues[2:,0]
                for para in range(min(Nb_parameters_fine,Nb_parameters_coarse)):
                    self.Para_modes[mode][para].Init_from_previous(BeamROM_coarse.Para_modes[mode][para])
                    # newparacoordinates = [coord for coord in self.Para_modes[mode][para].coordinates]
                    # newparacoordinates = torch.cat(newparacoordinates,dim=0)
                    # self.Para_modes[mode][para].InterpoLayer.weight.data = BeamROM_coarse.Para_modes[mode][para](newparacoordinates).T
        elif not os.path.isfile(PreviousFullModel):
            print('******** WARNING LEARNING FROM SCRATCH ********\n')

class InterpolationBlock2D_Lin(nn.Module):
    
    def __init__(self, connectivity):
       
        super(InterpolationBlock2D_Lin, self).__init__()
        self.connectivity = connectivity.astype(int)
    
    def UpdateConnectivity(self,connectivity):
        self.connectivity = connectivity.astype(int)

    def forward(self, x, cell_id, nodal_values, shape_functions,flag_training):
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

        #out = torch.cat(shape_functions[:,0]*node1_value[:,0] + shape_functions[:,1]*node2_value[:,0] + shape_functions[:,2]*node3_value[:,0], shape_functions[:,0]*node1_value[:,1] + shape_functions[:,1]*node2_value[:,1] + shape_functions[:,2]*node3_value[:,1])
        if flag_training:

            u = shape_functions[:,0]*node1_value + shape_functions[:,1]*node2_value + shape_functions[:,2]*node3_value

            return u
        else:
            return shape_functions[:,0]*node1_value + shape_functions[:,1]*node2_value + shape_functions[:,2]*node3_value

class InterpolationBlock2D_Quad(nn.Module):
    
    def __init__(self, connectivity):
       
        super(InterpolationBlock2D_Quad, self).__init__()
        self.connectivity = connectivity.astype(int)

    def forward(self, x, cell_id, nodal_values, shape_functions):
        """ This is the forward function of the Linear element block. Note that to prevent extrapolation outside of the structure's geometry, 
        phantom elements are used to cancel out the interpolation shape functions outside of the beam.
        Those phantom elements are flagged with index -1
        """
        cell_nodes_IDs = self.connectivity[cell_id,:] - 1

        node1_value =  torch.stack([torch.cat([val[row] for row in cell_nodes_IDs[:,0]]) for val in nodal_values], dim=0)
        node2_value =  torch.stack([torch.cat([val[row] for row in cell_nodes_IDs[:,1]]) for val in nodal_values], dim=0)
        node3_value =  torch.stack([torch.cat([val[row] for row in cell_nodes_IDs[:,2]]) for val in nodal_values], dim=0)

        node4_value =  torch.stack([torch.cat([val[row] for row in cell_nodes_IDs[:,3]]) for val in nodal_values], dim=0)
        node5_value =  torch.stack([torch.cat([val[row] for row in cell_nodes_IDs[:,4]]) for val in nodal_values], dim=0)
        node6_value =  torch.stack([torch.cat([val[row] for row in cell_nodes_IDs[:,5]]) for val in nodal_values], dim=0)
        #out = torch.cat(shape_functions[:,0]*node1_value[:,0] + shape_functions[:,1]*node2_value[:,0] + shape_functions[:,2]*node3_value[:,0], shape_functions[:,0]*node1_value[:,1] + shape_functions[:,1]*node2_value[:,1] + shape_functions[:,2]*node3_value[:,1])

        return shape_functions[:,0]*node1_value + shape_functions[:,1]*node2_value + shape_functions[:,2]*node3_value+\
                shape_functions[:,3]*node4_value + shape_functions[:,4]*node5_value + shape_functions[:,5]*node6_value

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
    def UpdateConnectivity(self,connectivity):
        self.connectivity = connectivity.astype(int)

    def GP(self):
        return torch.tensor([[1/3,1/3, 1/3]],dtype=torch.float64, requires_grad=True) # a1, a2, a3 the 3 area coordinates

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
        node3_coord =  torch.cat([coordinates[row-1] for row in cell_nodes_IDs[:,2]])

        if flag_training:
            refCoordg = self.GP().repeat(cell_id.shape[0],1)
            w_g = 0.5                           # Gauss weight
            Ng = torch.stack((refCoordg[:,0], refCoordg[:,1], refCoordg[:,2]),dim=1) #.view(sh_R.shape[0],-1) # Left | Right | Middle
            x_g = torch.stack([Ng[:,0]*node1_coord[:,0] + Ng[:,1]*node2_coord[:,0] + Ng[:,2]*node3_coord[:,0],Ng[:,0]*node1_coord[:,1] + Ng[:,1]*node2_coord[:,1] + Ng[:,2]*node3_coord[:,1]],dim=1)
            refCoord = GetRefCoord(x_g[:,0],x_g[:,1],node1_coord[:,0],node2_coord[:,0],node3_coord[:,0],node1_coord[:,1],node2_coord[:,1],node3_coord[:,1])
            N = torch.stack((refCoord[:,0], refCoord[:,1], refCoord[:,2]),dim=1) #.view(sh_R.shape[0],-1) # Left | Right | Middle
            detJ = (node1_coord[:,0] - node3_coord[:,0])*(node2_coord[:,1] - node3_coord[:,1]) - (node2_coord[:,0] - node3_coord[:,0])*(node1_coord[:,1] - node3_coord[:,1])
            return N,x_g, detJ*w_g

        else:
            refCoord = GetRefCoord(x[:,0],x[:,1],node1_coord[:,0],node2_coord[:,0],node3_coord[:,0],node1_coord[:,1],node2_coord[:,1],node3_coord[:,1])
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
        return torch.tensor([[1/6,1/6],[2/3,1/6],[1/6,2/3]])

    def forward(self, x, cell_id, coordinates, nodal_values,flag_training):
        """ This is the forward function of the Linear element block. Note that to prevent extrapolation outside of the structure's geometry, 
        phantom elements are used to cancel out the interpolation shape functions outside of the beam.
        Those phantom elements are flagged with index -1
        """

        cell_nodes_IDs = self.connectivity[cell_id,:]

        node1_coord =  torch.cat([coordinates[row-1] for row in cell_nodes_IDs[:,0]])
        node2_coord =  torch.cat([coordinates[row-1] for row in cell_nodes_IDs[:,1]])
        node3_coord =  torch.cat([coordinates[row-1] for row in cell_nodes_IDs[:,2]])

        # We dont need this.
        # node4_coord =  torch.cat([coordinates[row-1] for row in cell_nodes_IDs[:,3]])
        # node4_coord =  torch.cat([coordinates[row-1] for row in cell_nodes_IDs[:,4]])
        # node6_coord =  torch.cat([coordinates[row-1] for row in cell_nodes_IDs[:,5]])

        if flag_training:
            refCoord = self.GP()
        else:
            refCoord = GetRefCoord(x[:,0],x[:,1],node1_coord[:,0],node2_coord[:,0],node3_coord[:,0],node1_coord[:,1],node2_coord[:,1],node3_coord[:,1])
        
        N1 = refCoord[:,0]*(2*refCoord[:,0]-1)
        N2  = refCoord[:,1]*(2*refCoord[:,1]-1)
        N3  = refCoord[:,2]*(2*refCoord[:,2]-1)

        N4 = 4*refCoord[:,0]*refCoord[:,1]
        N5 = 4*refCoord[:,1]*refCoord[:,2]
        N6 = 4*refCoord[:,2]*refCoord[:,0]

        '''
        print("x = ", x[1])
        print("cell_id = ", cell_id[1])
        print("cell_nodes_IDs = ", cell_nodes_IDs[1])
        print((N1+N2+N3+N4+N5+N6)[1])
        print()
        '''

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
        self.coordinates = nn.ParameterList([nn.Parameter(torch.tensor([mesh.Nodes[i][1:int(mesh.dimension)+1]],dtype=torch.float64)) \
                                             for i in range(len(mesh.Nodes))])

        self.values = 0.0001*torch.randint(low=-1000, high=1000, size=(mesh.NNodes,n_components))
        # self.values =0.5*torch.ones((mesh.NNodes,n_components))
        self.frozen_BC_node_IDs = []
        self.frozen_BC_component_IDs = []
        self.relation_BC_node_IDs = []
        self.relation_BC_values = []
        self.relation_BC_normals = []
        self.constit_BC_node_IDs = []
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

    def Init_from_previous(self,CoarseModel):
        newcoordinates = [coord for coord in self.coordinates]
        newcoordinates = torch.cat(newcoordinates,dim=0)
        IDs_newcoord = torch.tensor(CoarseModel.mesh.GetCellIds(newcoordinates),dtype=torch.int)
        NewNodalValues = CoarseModel(newcoordinates,IDs_newcoord)
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
                NewNodalValues[0][ind_neg] = CoarseModel.nodal_values_x[closest_old_nodal_value].type(torch.float64)
                NewNodalValues[1][ind_neg] = CoarseModel.nodal_values_y[closest_old_nodal_value].type(torch.float64)
        new_nodal_values_x = nn.ParameterList([nn.Parameter((torch.tensor([i[0]]))) for i in NewNodalValues.t()])
        new_nodal_values_y = nn.ParameterList([nn.Parameter(torch.tensor([i[1]])) for i in NewNodalValues.t()])
        new_nodal_values = [new_nodal_values_x,new_nodal_values_y]
        self.nodal_values_x = new_nodal_values_x
        self.nodal_values_y = new_nodal_values_y
        self.nodal_values = new_nodal_values
 

    def SetBCs(self, ListOfDirichletsBCsValues):
        for i in range(len(ListOfDirichletsBCsValues)):
            if self.ListOfDirichletsBCsRelation[i] == False:
                if self.ListOfDirichletsBCsConstit[i] == False:
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
        nodes = self.connectivity[edge_id][0]
        Third_node = np.delete(nodes,np.where(nodes == edge_nodes[0]))
        Third_node = np.delete(Third_node,np.where(Third_node == edge_nodes[1]))


        new_connectivity = self.connectivity
        new_generation = self.elements_generation
        new_det = self.detJ_0.numpy()
        curren_det = new_det[edge_id]

        curren_gen = new_generation[edge_id]

        new_connectivity = np.delete(new_connectivity,(edge_id),axis = 0)
        new_generation = np.delete(new_generation,(edge_id),axis = 0)
        new_det = np.delete(new_det,(edge_id),axis = 0)

        new_elem = np.array([   [edge_nodes[0], new_node, Third_node[0]],
                                [edge_nodes[1], new_node, Third_node[0]]])
        new_connectivity = np.vstack((new_connectivity,new_elem))
        new_generation = np.hstack((new_generation,np.repeat(np.array(curren_gen+1), 2, axis=None)))
        new_det = np.hstack((new_det,np.repeat(np.array(curren_det/2), 2, axis=None)))

        self.connectivity = new_connectivity
        self.elements_generation = new_generation
        self.detJ_0 = torch.tensor(new_det)

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

    def StoreResults(self):
        u_x = [u for u in self.nodal_values_x]
        u_y = [u for u in self.nodal_values_y]
        u = torch.stack([torch.cat(u_x),torch.cat(u_y)],dim=1)
        self.U_interm.append(u.data)
        new_coord = [coord for coord in self.coordinates]
        new_coord = torch.cat(new_coord,dim=0)
        self.X_interm.append(new_coord)
        self.G_interm.append(self.elements_generation)
        self.Connectivity_interm.append(self.connectivity-1)
        self.Jacobian_interm.append(self.detJ_0)

    def RefinementParameters(self,MaxGeneration = 2, Jacobian_threshold = 0.4):
        self.MaxGeneration = MaxGeneration
        self.Jacobian_threshold = Jacobian_threshold
        self.MaxGeneration_elements = 0

    def SplitElemNonLoc(self, el_id):
        nodes = self.connectivity[el_id]
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

        NewNodes_indexes = np.max(self.connectivity) + np.array([1,2,3])
        new_elem = np.array([   [NewNodes_indexes[0], NewNodes_indexes[1], nodes[0]],
                                [nodes[1], NewNodes_indexes[2], NewNodes_indexes[0]],
                                [NewNodes_indexes[2], nodes[2], NewNodes_indexes[1]],
                                [NewNodes_indexes[2], NewNodes_indexes[1], NewNodes_indexes[0]]])


        for i in range(New_coordinates.shape[0]):
            self.coordinates.append(New_coordinates[None,i])
        new_connectivity = self.connectivity
        new_generation = self.elements_generation
        new_det = self.detJ_0.numpy()

        # Remove splitted element
        new_connectivity = np.delete(new_connectivity,(el_id),axis = 0)
        curren_gen = new_generation[el_id]
        curren_det = new_det[el_id]
        new_generation = np.delete(new_generation,(el_id),axis = 0)
        new_det = np.delete(new_det,(el_id),axis = 0)

        #Evaluate new nodale values:
        self.eval()
        newvalue = self(New_coordinates,torch.tensor([el_id,el_id,el_id]))
        self.train()
        # Initialise new nodale values
        for i in range(newvalue.shape[1]):
            self.nodal_values[0].append(newvalue[None,0,i])
            self.nodal_values[1].append(newvalue[None,1,i])
        # Update connectivity
        new_connectivity = np.vstack((new_connectivity,new_elem))
        new_generation = np.hstack((new_generation,np.repeat(np.array(curren_gen+1), 4, axis=None)))
        new_det = np.hstack((new_det,np.repeat(np.array(curren_det/4), 4, axis=None)))

        self.connectivity = new_connectivity
        self.elements_generation = new_generation
        self.detJ_0 = torch.tensor(new_det)

        if self.order =='1':
            self.ElementBlock.UpdateConnectivity(self.connectivity)
            self.Interpolation.UpdateConnectivity(self.connectivity)
        elif self.order == '2':
            self.ElementBlock.UpdateConnectivity(self.connectivity)
            self.Interpolation.UpdateConnectivity(self.connectivity)
        self.NElem +=3
        Removed_elem_list = [el_id]
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
        return Removed_elem_list

    def forward(self,x = 'NaN', el_id = 'NaN'):
        if self.training:
            el_id = torch.arange(0,self.NElem,dtype=torch.int)
            shape_functions,x_g, detJ = self.ElementBlock(x, el_id, self.coordinates, self.nodal_values, self.training)
            interpol = self.Interpolation(x_g, el_id, self.nodal_values, shape_functions, self.training)
            return interpol,x_g, detJ
        else:
            shape_functions = self.ElementBlock(x, el_id, self.coordinates, self.nodal_values, self.training)
            interpol = self.Interpolation(x, el_id, self.nodal_values, shape_functions, self.training)
            return interpol

    def UnFreeze_FEM(self):
        """Set the coordinates as trainable parameters """
        # print("Unfreeze values")

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
        """Set the coordinates as untrainable parameters """
        for dim in self.nodal_values:
            for val in dim:
                val.requires_grad = False
      
    def Freeze_Mesh(self):
        """Set the coordinates as untrainable parameters"""
        for param in self.coordinates:
            param.requires_grad = False
    
    def UnFreeze_Mesh(self):
        """Set the coordinates as trainable parameters"""
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
        
        print("     * Export current nodal coordinates")

        cell_nodes_IDs = mesh.Connectivity

        node1_coord =  torch.cat([self.coordinates[int(row)-1] for row in cell_nodes_IDs[:,0]])
        node2_coord =  torch.cat([self.coordinates[int(row)-1] for row in cell_nodes_IDs[:,1]])
        node3_coord =  torch.cat([self.coordinates[int(row)-1] for row in cell_nodes_IDs[:,2]])

        # node4_coord =  torch.cat([self.coordinates[int(row)-1] for row in cell_nodes_IDs[:,3]])
        # node5_coord =  torch.cat([self.coordinates[int(row)-1] for row in cell_nodes_IDs[:,4]])
        # node6_coord =  torch.cat([self.coordinates[int(row)-1] for row in cell_nodes_IDs[:,5]])

        T6_Coord1 = node1_coord*0.5 + node2_coord*0.5 
        T6_Coord2 = node2_coord*0.5 + node3_coord*0.5
        T6_Coord3 = node1_coord*0.5 + node3_coord*0.5

        for j in range(len(cell_nodes_IDs)):
            self.coordinates[int(cell_nodes_IDs[j,3])-1] = T6_Coord1[j]
            self.coordinates[int(cell_nodes_IDs[j,4])-1] = T6_Coord2[j]
            self.coordinates[int(cell_nodes_IDs[j,5])-1] = T6_Coord3[j]



def GetRefCoord(x,y,x1,x2,x3,y1,y2,y3):

    inverse_matrix = torch.ones([int(y.shape[0]), 3, 3], dtype=torch.float64)

    inverse_matrix[:,0,0] = (y3 - y2)/(x1*(y3 - y2) + x2*(y1 - y3) + x3*(y2 - y1)) 
    inverse_matrix[:,1,0] = (x2 - x3)/(-x1*y2 + x1*y3 + x2*y1 - x2*y3 - x3*y1 + x3*y2)
    inverse_matrix[:,2,0] = (x3*y2 - x2*y3)/(-x1*y2 + x1*y3 + x2*y1 - x2*y3 - x3*y1 + x3*y2)

    inverse_matrix[:,0,1] = (y1 - y3)/(-x1*y2 + x1*y3 + x2*y1 - x2*y3 - x3*y1 + x3*y2)
    inverse_matrix[:,1,1] = (x1 - x3)/(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    inverse_matrix[:,2,1]= (x3*y1 - x1*y3)/(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

    inverse_matrix[:,0,2] = (y1 - y2)/(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    inverse_matrix[:,1,2] = (x1 - x2)/(-x1*y2 + x1*y3 + x2*y1 - x2*y3 - x3*y1 + x3*y2)
    inverse_matrix[:,2,2] = (x2*y1 - x1*y2)/(-x1*y2 + x1*y3 + x2*y1 - x2*y3 - x3*y1 + x3*y2)

    x_extended = torch.stack((x,y, torch.ones_like(y)),dim=1)
    x_extended = x_extended.unsqueeze(1)

    return torch.matmul(x_extended, inverse_matrix).squeeze(1)


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
        self.version = "Gauss_quadrature"
        self.mesh = mesh
        self.n_integr_points = n_integr_points
        # if self.n_integr_points == 0:
        #     self.Mixed = True
        n_components = 1

        self.coordinates = nn.ParameterList([nn.Parameter(torch.tensor([mesh.Nodes[i][1:int(mesh.dimension)+1]],dtype=torch.float64)) \
                                             for i in range(len(mesh.Nodes))])

        print("mesh.NNodes = ", mesh.NNodes)
        # self.values = 0.001*torch.randint(low=-10, high=10, size=(mesh.NNodes,1))
        # self.values =0.01*torch.ones((mesh.NNodes,1))

        self.values =0.1*torch.ones((mesh.NNodes,1))

        self.connectivity = mesh.Connectivity
        self.borders_nodes = mesh.borders_nodes

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

    def UnFreeze_FEM(self):
        """Set the coordinates as trainable parameters """
        # print("Unfreeze values")

        for val in self.nodal_values:
            val.requires_grad = True

        for j in range(len(self.frozen_BC_node_IDs)):
            frozen = self.frozen_BC_node_IDs[j]
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
        for param in self.coordinates:
            param.requires_grad = True

        border_nodes = torch.unique(torch.tensor(self.borders_nodes, dtype=torch.int))-1
        for node in border_nodes:
            self.coordinates[node].requires_grad = False
    
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