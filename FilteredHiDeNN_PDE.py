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
import numpy

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

class InterpolationBlock(nn.Module):
    
    def __init__(self, connectivity):
       
        super(InterpolationBlock, self).__init__()
        self.LinearBlock = LinearBlock()
        self.connectivity = connectivity.astype(int)

    def forward(self, x, cell_id, nodal_values, shape_functions):
        """ This is the forward function of the Linear element block. Note that to prevent extrapolation outside of the structure's geometry, 
        phantom elements are used to cancel out the interpolation shape functions outside of the beam.
        Those phantom elements are flagged with index -1
        """

        cell_nodes_IDs = self.connectivity[cell_id,:]

        node1_value =  torch.cat([nodal_values[row-1] for row in cell_nodes_IDs[:,0]])
        node2_value =  torch.cat([nodal_values[row-1] for row in cell_nodes_IDs[:,1]])
        node3_value =  torch.cat([nodal_values[row-1] for row in cell_nodes_IDs[:,2]])

        return shape_functions[:,0]*node1_value + shape_functions[:,1]*node2_value + shape_functions[:,2]*node3_value

class ElementBlock2D_Lin(nn.Module):
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
        super(ElementBlock2D_Lin, self).__init__()
        self.LinearBlock = LinearBlock()
        self.connectivity = connectivity.astype(int)

    '''
    def LinSHQuad(self, x, node1, node2, node3, node4):
        pom11 = (node3[:,0]-node2[:,0])*(x[:,1]     -node2[:,1]) - (node3[:,1]-node2[:,1])*(x[:,0]      -node2[:,0])
        pom12 = (node3[:,0]-node2[:,0])*(node1[:,1] -node2[:,1]) - (node3[:,1]-node2[:,1])*(node1[:,0]  -node2[:,0])

        pom21 = (node3[:,0]-node4[:,0])*(x[:,1]     -node4[:,1]) - (node3[:,1]-node4[:,1])*(x[:,0]      -node4[:,0])
        pom22 = (node3[:,0]-node4[:,0])*(node1[:,1] -node4[:,1]) - (node3[:,1]-node4[:,1])*(node1[:,0]  -node4[:,0])    

        return (pom11/pom12)*(pom21/pom22)

    def forward_QUAD(self, x, cell_id, coordinates, nodal_values):
        """ This is the forward function of the Linear element block. Note that to prevent extrapolation outside of the structure's geometry, 
        phantom elements are used to cancel out the interpolation shape functions outside of the beam.
        Those phantom elements are flagged with index -1
        """

        cell_nodes_IDs = self.connectivity[cell_id,:]
        node1_coord =  torch.cat([coordinates[row-1] for row in cell_nodes_IDs[:,0]])
        node2_coord =  torch.cat([coordinates[row-1] for row in cell_nodes_IDs[:,1]])
        node3_coord =  torch.cat([coordinates[row-1] for row in cell_nodes_IDs[:,2]])
        node4_coord =  torch.cat([coordinates[row-1] for row in cell_nodes_IDs[:,3]])

        node1_value =  torch.cat([nodal_values[row-1] for row in cell_nodes_IDs[:,0]])
        node2_value =  torch.cat([nodal_values[row-1] for row in cell_nodes_IDs[:,1]])
        node3_value =  torch.cat([nodal_values[row-1] for row in cell_nodes_IDs[:,2]])
        node4_value =  torch.cat([nodal_values[row-1] for row in cell_nodes_IDs[:,3]])


        print("x = ", x[64])
        print("cell id = ", cell_id[64])
        print("nodes = ", cell_nodes_IDs[64])
        print("node1_coord = ", node1_coord[64])
        print("node2_coord = ", node2_coord[64])
        print("node3_coord = ", node3_coord[64])
        print("node4_coord = ", node4_coord[64])
        print()
        print("node1 value = ", node1_value[64])
        print("node2 value = ", node2_value[64])
        print("node3 value = ", node3_value[64])
        print("node4 value = ", node4_value[64])
        print()


        sf_node1 = self.LinSH(x, node1_coord, node2_coord, node3_coord, node4_coord)
        sf_node2 = self.LinSH(x, node2_coord, node3_coord, node4_coord, node1_coord)
        sf_node3 = self.LinSH(x, node3_coord, node4_coord, node1_coord, node2_coord)
        sf_node4 = self.LinSH(x, node4_coord, node1_coord, node2_coord, node3_coord)

        print(min(sf_node1).item(), max(sf_node1).item())
        print(min(sf_node2).item(), max(sf_node2).item())
        print(min(sf_node3).item(), max(sf_node3).item())
        print(min(sf_node4).item(), max(sf_node4).item())

        print()

        print(sf_node1[64])
        print(sf_node2[64])
        print(sf_node3[64])
        print(sf_node4[64])
        print()

        bad_id2 = torch.where(sf_node2>1)
        print("bad_id2 = ", bad_id2)
        print("bad point = ", x[bad_id2], "   id = ", cell_id[bad_id2])
        print()

        bad_id4 = torch.where(sf_node4>1)
        print("bad_id4 = ", bad_id4)
        print("bad point = ", x[bad_id4], "   id = ", cell_id[bad_id4])
        print()


        print("sum max = ", max(sf_node1 + sf_node2 + sf_node3 + sf_node4).item())
        print("sum min = ", min(sf_node1 + sf_node2 + sf_node3 + sf_node4).item())

        interpol = sf_node1*node1_value + sf_node2*node2_value + sf_node3*node3_value + sf_node4*node4_value
        #interpol =  sf_node3

        return interpol
    '''

    def forward(self, x, cell_id, coordinates, nodal_values):
        """ This is the forward function of the Linear element block. Note that to prevent extrapolation outside of the structure's geometry, 
        phantom elements are used to cancel out the interpolation shape functions outside of the beam.
        Those phantom elements are flagged with index -1
        """

        cell_nodes_IDs = self.connectivity[cell_id,:]
        node1_coord =  torch.cat([coordinates[row-1] for row in cell_nodes_IDs[:,0]])
        node2_coord =  torch.cat([coordinates[row-1] for row in cell_nodes_IDs[:,1]])
        node3_coord =  torch.cat([coordinates[row-1] for row in cell_nodes_IDs[:,2]])

        node1_value =  torch.cat([nodal_values[row-1] for row in cell_nodes_IDs[:,0]])
        node2_value =  torch.cat([nodal_values[row-1] for row in cell_nodes_IDs[:,1]])
        node3_value =  torch.cat([nodal_values[row-1] for row in cell_nodes_IDs[:,2]])

        refCoord = GetRefCoord(x[:,0],x[:,1],node1_coord[:,0],node2_coord[:,0],node3_coord[:,0],node1_coord[:,1],node2_coord[:,1],node3_coord[:,1])
        
        print("x = ", x[2])
        print("cell id = ", cell_id[2])
        print("nodes = ", cell_nodes_IDs[2])
        print("node1_coord = ", node1_coord[2])
        print("node2_coord = ", node2_coord[2])
        print("node3_coord = ", node3_coord[2])
        print()
        print("refCoord = ", refCoord[2])
        
        out = torch.stack((refCoord[:,0], refCoord[:,1], refCoord[:,2]),dim=1) #.view(sh_R.shape[0],-1) # Left | Right | Middle

        return out


class MeshNN(nn.Module):
    """ This class is a space HiDeNN building a Finite Element (FE) interpolation over the space domain. 
    The coordinates of the nodes of the underlying mesh are trainable. Those coordinates are passed as a List of Parameters to the subsequent sub-neural networks
    Updating those parameters correspond to r-adaptativity
    The Interpolation layer weights correspond to the nodal values. Updating them 
    is equivqlent to solving the PDE. """


    def __init__(self, mesh, alpha = 0.005):
        super(MeshNN, self).__init__()
        self.alpha = alpha                                      # set the weight for the Mesh regularisation 
        self.coordinates = nn.ParameterList([nn.Parameter(torch.tensor([mesh.Nodes[i][1:int(mesh.dimension)+1]],dtype=torch.float64)) \
                                             for i in range(len(mesh.Nodes))])

        #self.values = 0.0001*torch.randint(low=-100, high=100, size=(mesh.NNodes,))
        self.values =torch.ones((mesh.NNodes,))
        self.values[8]=5
        self.nodal_values = nn.ParameterList([nn.Parameter(torch.tensor([i])) for i in self.values])

        self.dofs = mesh.NNodes*mesh.dim # Number of Dofs
        self.NElem = mesh.NElem
        self.NBCs = len(mesh.ListOfDirichletsBCsIds) # Number of prescribed Dofs
        self.ElementBlock = ElementBlock2D_Lin(mesh.Connectivity)
        self.Interpolation = InterpolationBlock(mesh.Connectivity)

        # Maybe we don't need this anymore
        #self.ElemList = torch.arange(self.NElem)
                
    def forward(self,x, el_id):
            
        shape_functions = self.ElementBlock(x, el_id, self.coordinates, self.nodal_values)
        print("shape_functions = ", shape_functions.shape)

        interpol = self.Interpolation(x, el_id, self.nodal_values, shape_functions)
        print("interpol = ", interpol.shape )

        return interpol







L = 10                                              # Length of the Beam
np = 3                                             # Number of Nodes in the Mesh
A = 1                                               # Section of the beam
E = 175                                             # Young's Modulus (should be 175)
# User defines all boundary conditions 
DirichletDictionryList = [  {"Entity": 1, 
                             "Value": 0, 
                             "normal":1}, 
                            {"Entity": 2, 
                             "Value": 0.0, 
                             "normal":1}]

# Definition of the space discretisation
alpha =0.005                                       # Weight for the Mesh regularisation 
order = 1                                          # Order of the shape functions
dimension = 2

if order ==1:
    MaxElemSize = L/(np-1)                         # Compute element size
elif order ==2:
    n_elem = 0.5*(np-1)
    MaxElemSize = L/n_elem                         # Compute element size

if dimension ==1:
    Domain_mesh = pre.Mesh('Beam',MaxElemSize, order, dimension)    # Create the mesh object
if dimension ==2:
    Domain_mesh = pre.Mesh('Rectangle',MaxElemSize, order, dimension)    # Create the mesh object
Volume_element = 100                               # Volume element correspond to the 1D elem in 1D

Domain_mesh.AddBCs(Volume_element,
                 DirichletDictionryList)           # Include Boundary physical domains infos (BCs+volume)
Domain_mesh.MeshGeo()                                # Mesh the .geo file if .msh does not exist


Domain_mesh.ReadMesh()                               # Parse the .msh file
Domain_mesh.ReadMeshVtk()

n_train = 100

TrailCoord_1d = torch.tensor([i for i in torch.linspace(0,L,n_train)],dtype=torch.float64)
TrialCoordinates = torch.cartesian_prod(TrailCoord_1d,TrailCoord_1d)
print("TrialCoordinates = ", TrialCoordinates.shape) 
#print(TrialCoordinates) 

TrialIDs = torch.tensor(Domain_mesh.GetCellIds(TrialCoordinates),dtype=torch.int)
print("TrialIDs = ", TrialIDs.shape)
#print(TrialIDs)

DomainModel = MeshNN(Domain_mesh,alpha)                # Create the associated model

u_predicted = DomainModel(TrialCoordinates, TrialIDs) 


print("u_predicted min : max ", min(u_predicted), " : ", max(u_predicted))

img = numpy.reshape(u_predicted.detach(), (n_train, n_train), order='C') 
plt.imshow(img) #, vmin = 1-1.0e-7, vmax = 1+1.0e-7)
plt.colorbar()
plt.savefig('Results/2D_val.pdf', transparent=True)  
plt.clf()




