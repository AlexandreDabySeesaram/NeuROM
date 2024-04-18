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
        #self.NodalValues_uu = nn.Parameter(data=0.1*torch.ones(self.dofs-self.NBCs), requires_grad=False)
        #self.InterpoLayer_uu.weight.data = self.NodalValues_uu
        self.InterpoLayer_uu.weight.data = 0.001*torch.randint(low=-100, high=100, size=(self.dofs-self.NBCs,))

 
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
    
    def SetBCs(self,u_0,u_L):
        """Set the two Dirichlet boundary conditions
        Args:
            u_0 (Float): The left BC
            u_L (Float): The right BC """
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

class NeuROM(nn.Module):
    """This class builds the Reduced-order model from the interpolation NN for space and parameters space"""
    def __init__(self, mesh, BCs, n_modes, ParametersList):
        super(NeuROM, self).__init__()
        IndexesNon0BCs = [i for i, BC in enumerate(BCs) if BC != 0]
        if IndexesNon0BCs and n_modes==1: #If non homogeneous BCs, add mode for relevement
            n_modes+=1
        self.n_modes = n_modes
        self.n_para = len(ParametersList)
        self.Space_modes = nn.ModuleList([MeshNN(mesh) for i in range(self.n_modes)])
        # self.Para_Nets = nn.ModuleList([InterpPara(Para[0], Para[1], Para[2]) for Para in ParametersList])
        self.Para_modes = nn.ModuleList([nn.ModuleList([InterpPara(Para[0], Para[1], Para[2]) for Para in ParametersList]) for i in range(self.n_modes)])
        # Set BCs 
        if IndexesNon0BCs:
            # First modes get the Boundary conditions
            self.Space_modes[0].SetBCs(BCs[0],BCs[1])
            for para in range(self.n_para):
                self.Para_modes[0][para].InterpoLayer.weight.data.fill_(1) # Mode for parameters not trained and set to 1 to get correct space BCs
                self.Para_modes[0][para].Freeze_FEM()
            # Following modes are homogeneous (admissible to 0)
            for i in range(1,self.n_modes):
                self.Space_modes[i].SetBCs(0,0)
        else:
            for i in range(self.n_modes):
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
        for mode in range(self.n_modes):
            Para_mode_List = [self.Para_modes[mode][l](mu[l][:,0].view(-1,1))[:,None] for l in range(self.n_para)]
            if mode == 0:
                # Para_modes = torch.unsqueeze(torch.cat(Para_mode_List,dim=1), dim=0)
                # Para_modes = [Para_mode_List[l] for l in range(self.n_para)]
                Para_modes = [torch.unsqueeze(Para_mode_List[l],dim=0) for l in range(self.n_para)]
            else:
                # New_mode = torch.unsqueeze(torch.cat(Para_mode_List,dim=1), dim=0)
                # Para_modes = torch.vstack((Para_modes,New_mode))
                New_mode = Para_mode_List
                Para_modes = [torch.vstack((Para_modes[l],torch.unsqueeze(New_mode[l],dim=0))) for l in range(self.n_para)]
        if len(mu)==1:
            # out = torch.matmul(Space_modes,Para_modes[0].view(self.n_modes,Para_modes[0].shape[1]))
            out = torch.einsum('ik,kj->ij',Space_modes,Para_modes[0].view(self.n_modes,Para_modes[0].shape[1]))
        elif len(mu)==2:
            out = torch.einsum('ik,kj,kl->ijl',Space_modes,Para_modes[0].view(self.n_modes,Para_modes[0].shape[1]),
                            Para_modes[1].view(self.n_modes,Para_modes[1].shape[1]))
        return out

class InterpolationBlock2D_Lin(nn.Module):
    
    def __init__(self, connectivity):
       
        super(InterpolationBlock2D_Lin, self).__init__()
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

        #out = torch.cat(shape_functions[:,0]*node1_value[:,0] + shape_functions[:,1]*node2_value[:,0] + shape_functions[:,2]*node3_value[:,0], shape_functions[:,0]*node1_value[:,1] + shape_functions[:,1]*node2_value[:,1] + shape_functions[:,2]*node3_value[:,1])

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

    def forward(self, x, cell_id, coordinates, nodal_values):
        """ This is the forward function of the Linear element block. Note that to prevent extrapolation outside of the structure's geometry, 
        phantom elements are used to cancel out the interpolation shape functions outside of the beam.
        Those phantom elements are flagged with index -1
        """

        cell_nodes_IDs = self.connectivity[cell_id,:]
        node1_coord =  torch.cat([coordinates[row-1] for row in cell_nodes_IDs[:,0]])
        node2_coord =  torch.cat([coordinates[row-1] for row in cell_nodes_IDs[:,1]])
        node3_coord =  torch.cat([coordinates[row-1] for row in cell_nodes_IDs[:,2]])

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

    def forward(self, x, cell_id, coordinates, nodal_values):
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
        #self.values =0.5*torch.ones((mesh.NNodes,n_components))
        self.frozen_BC_node_IDs = []
        self.frozen_BC_component_IDs = []
        self.relation_BC_node_IDs = []
        self.relation_BC_values = []
        self.relation_BC_normals = []
        self.constit_BC_node_IDs = []
        self.relation_BC_lines = []

        self.ExcludeFromDirichlet = mesh.ExcludedPoints
        self.borders_nodes = mesh.borders_nodes

        if mesh.NoBC==False:
            for i in range(len(mesh.ListOfDirichletsBCsValues)):
                if mesh.ListOfDirichletsBCsRelation[i] == False:
                    if mesh.ListOfDirichletsBCsConstit[i] == False:
                        IDs = torch.tensor(mesh.DirichletBoundaryNodes[i], dtype=torch.int)
                        IDs = torch.unique(IDs.reshape(IDs.shape[0],-1))-1
                        self.frozen_BC_node_IDs.append(IDs)
                        self.frozen_BC_component_IDs.append(mesh.ListOfDirichletsBCsNormals[i])
                        self.values[IDs,mesh.ListOfDirichletsBCsNormals[i]] = mesh.ListOfDirichletsBCsValues[i]
                        print("  Entity : ", mesh.ListOfDirichletsBCsIds[i])
                        print("    Simple BC ")
                        print("      IDs : ", IDs)
                        print("      component : ", mesh.ListOfDirichletsBCsNormals[i])
                        print("      values : ", mesh.ListOfDirichletsBCsValues[i])
                        print()
                else:
                    IDs = torch.tensor(mesh.DirichletBoundaryNodes[i], dtype=torch.int)
                    self.relation_BC_lines.append(IDs)
                    IDs = torch.unique(IDs.reshape(IDs.shape[0],-1))-1
                    self.relation_BC_node_IDs.append(IDs)
                    self.relation_BC_values.append(mesh.ListOfDirichletsBCsValues[i])
                    #self.relation_BC_normals.append(mesh.normals[IDs])
                    print("  Entity : ", mesh.ListOfDirichletsBCsIds[i])
                    print("    Relation BC ")
                    print("      IDs : ", IDs)
                    print("      values : ", self.relation_BC_values)
                    print()

            for i in range(len(mesh.ListOfDirichletsBCsValues)):
                if mesh.ListOfDirichletsBCsConstit[i] == True:
                    IDs = torch.tensor(mesh.DirichletBoundaryNodes[i], dtype=torch.int)
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
                    print("  Entity : ", mesh.ListOfDirichletsBCsIds[i])
                    print("    Constitutive BC ")
                    print("      IDs : ", IDs)
                    print()


        if n_components ==2:
            # nn.ParameterList is supposed to hold a single list of nn.Parameter and cannot contain other nn.ParameterLists
            self.nodal_values_x = nn.ParameterList([nn.Parameter(torch.tensor([i[0]])) for i in self.values])
            self.nodal_values_y = nn.ParameterList([nn.Parameter(torch.tensor([i[1]])) for i in self.values])
            self.nodal_values = [self.nodal_values_x,self.nodal_values_y]


        elif n_components ==3:
            # nn.ParameterList is supposed to hold a single list of nn.Parameter and cannot contain other nn.ParameterLists
            self.nodal_values_x = nn.ParameterList([nn.Parameter(torch.tensor([i[0]])) for i in self.values])
            self.nodal_values_y = nn.ParameterList([nn.Parameter(torch.tensor([i[1]])) for i in self.values])
            self.nodal_values_xy = nn.ParameterList([nn.Parameter(torch.tensor([i[2]])) for i in self.values])
            self.nodal_values = [self.nodal_values_x,self.nodal_values_y, self.nodal_values_xy]


        print("self.nodal_values = ", len(self.nodal_values), (len(self.nodal_values[0])))

        self.dofs = mesh.NNodes*mesh.dim # Number of Dofs
        self.NElem = mesh.NElem
    
        if mesh.NoBC==False:
            self.NBCs = len(mesh.ListOfDirichletsBCsIds) # Number of prescribed Dofs
        else:
            self.NBCs = 0

        if mesh.order =='1':
            self.ElementBlock = ElementBlock2D_Lin(mesh.Connectivity)
            self.Interpolation = InterpolationBlock2D_Lin(mesh.Connectivity)
        elif mesh.order == '2':
            self.ElementBlock = ElementBlock2D_Quad(mesh.Connectivity)
            self.Interpolation = InterpolationBlock2D_Quad(mesh.Connectivity)



        print("----------------------------------------------")


        print("----------------------------------------------")
        print()


    def forward(self,x, el_id):
            
        shape_functions = self.ElementBlock(x, el_id, self.coordinates, self.nodal_values)
        interpol = self.Interpolation(x, el_id, self.nodal_values, shape_functions)

        return interpol

    def UnFreeze_Values(self):
        """Set the coordinates as trainable parameters """
        print("Unfreeze values")

        for dim in self.nodal_values:
            for val in dim:
                val.requires_grad = True
 
        for j in range(len(self.frozen_BC_node_IDs)):
            print("Component ", self.frozen_BC_component_IDs[j], " : ", self.frozen_BC_node_IDs[j][0:5])
            print("To be excluded : ", self.ExcludeFromDirichlet)
            values = self.nodal_values[self.frozen_BC_component_IDs[j]]
            frozen = self.frozen_BC_node_IDs[j]
            
            all_excluded =  [f in self.ExcludeFromDirichlet for f in frozen]

            if all(all_excluded) == False:
                print("Excluded")

                for idf in frozen:
                    if idf not in self.ExcludeFromDirichlet:
                        values[idf].requires_grad = False
                    else:
                        values[idf].requires_grad = True
            else:
                print("Not excluded")
                for idf in frozen:
                    values[idf].requires_grad = False


    def Freeze_Mesh(self):
        """Set the coordinates as untrainable parameters"""
        for param in self.coordinates:
            param.requires_grad = False

    def CheckBCValues(self):
        """Set the coordinates as trainable parameters """
        print("Unfreeze values")

        for j in range(len(self.frozen_BC_node_IDs)):
            print(j, self.frozen_BC_component_IDs[j], self.frozen_BC_node_IDs[j])
            values = self.nodal_values[self.frozen_BC_component_IDs[j]]
            frozen = self.frozen_BC_node_IDs[j]
            for idf in frozen:
                print(values[idf])


    def Update_Middle_Nodes(self, mesh):

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

    def UnFreeze_Mesh(self):
        """Set the coordinates as trainable parameters"""
        for param in self.coordinates:
            param.requires_grad = True
        border_nodes = torch.unique(torch.tensor(self.borders_nodes, dtype=torch.int))-1
        for node in border_nodes:
            self.coordinates[node].requires_grad = False



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

