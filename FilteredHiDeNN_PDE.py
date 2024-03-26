#%% Libraries import
import time 
# Import pre-processing functions
import Bin.Pre_processing as pre
# Import mechanical functions
from Bin.PDE_Library import RHS, PotentialEnergyVectorised, \
        Derivative, AnalyticGradientSolution, AnalyticSolution
from Post.Evaluation_wrt_NumSolution import NumSol_eval

# Import torch librairies
import torch
import torch.nn as nn
torch.set_default_dtype(torch.float32)
#Import post processing libraries
import Post.Plots as Pplot
import matplotlib.pyplot as plt
mps_device = torch.device("mps")
import numpy
from scipy import ndimage


from Bin.Training import LBFGS_Stage2_2D, GradDescend_Stage1_2D
from Bin.PDE_Library import Mixed_2D_loss


class Dataset(torch.utils.data.Dataset):

    def __init__(self, X):
        self.X = X
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):    
        x = self.X[index]

        return x

def GetMaxElemSize(L, np, order):
    if order ==1:
        MaxElemSize = L/(np-1)                         # Compute element size
    elif order ==2:
        n_elem = 0.5*(np-1)
        MaxElemSize = L/n_elem                         # Compute element size

    return MaxElemSize

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

class InterpolationBlock_Lin(nn.Module):
    
    def __init__(self, connectivity):
       
        super(InterpolationBlock_Lin, self).__init__()
        self.LinearBlock = LinearBlock()
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

class InterpolationBlock_Quad(nn.Module):
    
    def __init__(self, connectivity):
       
        super(InterpolationBlock_Quad, self).__init__()
        self.LinearBlock = LinearBlock()
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

        node4_coord =  torch.cat([coordinates[row-1] for row in cell_nodes_IDs[:,3]])
        node4_coord =  torch.cat([coordinates[row-1] for row in cell_nodes_IDs[:,4]])
        node6_coord =  torch.cat([coordinates[row-1] for row in cell_nodes_IDs[:,5]])

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

class MeshNN(nn.Module):
    """ This class is a space HiDeNN building a Finite Element (FE) interpolation over the space domain. 
    The coordinates of the nodes of the underlying mesh are trainable. Those coordinates are passed as a List of Parameters to the subsequent sub-neural networks
    Updating those parameters correspond to r-adaptativity
    The Interpolation layer weights correspond to the nodal values. Updating them 
    is equivqlent to solving the PDE. """


    def __init__(self, mesh, n_components):
        super(MeshNN, self).__init__()
        self.coordinates = nn.ParameterList([nn.Parameter(torch.tensor([mesh.Nodes[i][1:int(mesh.dimension)+1]],dtype=torch.float64)) \
                                             for i in range(len(mesh.Nodes))])

        self.values = 0.001*torch.randint(low=-100, high=100, size=(mesh.NNodes,n_components))
        #self.values =0.5*torch.ones((mesh.NNodes,n_components))
        self.frozen_BC_values_IDs = []
        self.frozen_BC_component_IDs = []

        for i in range(len(mesh.ListOfDirichletsBCsValues)):

            IDs = torch.tensor(mesh.DirichletBoundaryNodes[i], dtype=torch.int)
            IDs = torch.unique(IDs.reshape(IDs.shape[0],-1))-1
            self.frozen_BC_values_IDs.append(IDs)
            self.frozen_BC_component_IDs.append(mesh.ListOfDirichletsBCsNormals[i])
            self.values[IDs,mesh.ListOfDirichletsBCsNormals[i]] = mesh.ListOfDirichletsBCsValues[i]

        #self.nodal_values = [ nn.ParameterList([nn.Parameter(torch.tensor([i[j]])) for i in self.values]) for j in range(n_components)]
        #print("self.nodal_values = ", len(self.nodal_values), len(self.nodal_values[0]))

        #self.nodal_values = []
        #for dim in range(n_components):
        #    self.nodal_values.append(nn.ParameterList([nn.Parameter(torch.tensor([i[dim]])) for i in self.values]))


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
        self.NBCs = len(mesh.ListOfDirichletsBCsIds) # Number of prescribed Dofs

        if mesh.order =='1':
            self.ElementBlock = ElementBlock2D_Lin(mesh.Connectivity)
            self.Interpolation = InterpolationBlock_Lin(mesh.Connectivity)
        elif mesh.order == '2':
            self.ElementBlock = ElementBlock2D_Quad(mesh.Connectivity)
            self.Interpolation = InterpolationBlock_Quad(mesh.Connectivity)

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
 
        for j in range(len(self.frozen_BC_values_IDs)):
            print(j, self.frozen_BC_component_IDs[j], self.frozen_BC_values_IDs[j])
            values = self.nodal_values[self.frozen_BC_component_IDs[j]]
            frozen = self.frozen_BC_values_IDs[j]
            for idf in frozen:
                values[idf].requires_grad = False

    def Freeze_Mesh(self):
        """Set the coordinates as untrainable parameters"""
        for param in self.coordinates:
            param.requires_grad = False

    def CheckBCValues(self):
        """Set the coordinates as trainable parameters """
        print("Unfreeze values")

        for j in range(len(self.frozen_BC_values_IDs)):
            print(j, self.frozen_BC_component_IDs[j], self.frozen_BC_values_IDs[j])
            values = self.nodal_values[self.frozen_BC_component_IDs[j]]
            frozen = self.frozen_BC_values_IDs[j]
            for idf in frozen:
                print(values[idf])



L = 10                                              # Length of the Beam
np = 5                                             # Number of Nodes in the Mesh
#A = 1                                               # Section of the beam
#E = 175                                             # Young's Modulus (should be 175)

loss = [[],[]]

lmbda = 1.25
mu = 1.0

# Definition of the space discretisation
order_u = 2                                          # Order of the shape functions
order_du = 1
dimension = 2

MaxElemSize = GetMaxElemSize(L, np, order_du)

if dimension ==1:
    Domain_mesh_u = pre.Mesh('Beam',MaxElemSize, order_u, dimension)    # Create the mesh object
    Domain_mesh_du = pre.Mesh('Beam',MaxElemSize, order_du, dimension)    # Create the mesh object
if dimension ==2:
    Domain_mesh_u = pre.Mesh('Rectangle',MaxElemSize, order_u, dimension)    # Create the mesh object
    Domain_mesh_du = pre.Mesh('Rectangle',MaxElemSize, order_du, dimension)    # Create the mesh object

Volume_element = 100                               # Volume element correspond to the 1D elem in 1D

####################################################################
# User defines all boundary conditions
DirichletDictionryList = [  {"Entity": 111, "Value": 0, "Normal": 0},
                            {"Entity": 111, "Value": 0, "Normal": 1},
                            {"Entity": 113, "Value": 0, "Normal": 0},
                            {"Entity": 113, "Value": 1, "Normal": 1},]

Domain_mesh_u.AddBCs(Volume_element, DirichletDictionryList)           # Include Boundary physical domains infos (BCs+volume)
Domain_mesh_u.MeshGeo()                                # Mesh the .geo file if .msh does not exist
Domain_mesh_u.ReadMesh()                               # Parse the .msh file
Domain_mesh_u.ExportMeshVtk()

####################################################################
# Normal: 0     component: x, du_dx
# Normal: 1     component: y, dv_dy
# Normal: 2     component: x, du_dy = dv_dx

DirichletDictionryList = [  {"Entity": 112, "Value": 0, "Normal": 0},
                            {"Entity": 112, "Value": 0, "Normal":  2},
                            {"Entity": 114, "Value": 0, "Normal": 0},
                            {"Entity": 114, "Value": 0, "Normal": 2},]

Domain_mesh_du.AddBCs(Volume_element, DirichletDictionryList)           # Include Boundary physical domains infos (BCs+volume)
Domain_mesh_du.MeshGeo()                                # Mesh the .geo file if .msh does not exist
Domain_mesh_du.ReadMesh()                               # Parse the .msh file
Domain_mesh_du.ExportMeshVtk()

####################################################################

n_train = 20

TrailCoord_1d_x = torch.tensor([i for i in torch.linspace(0,L,n_train)],dtype=torch.float64, requires_grad=True)
TrailCoord_1d_y = torch.tensor([i for i in torch.linspace(0,5*L,5*n_train)],dtype=torch.float64,  requires_grad=True)

PlotCoordinates = torch.cartesian_prod(TrailCoord_1d_x,TrailCoord_1d_y)
IDs_u = torch.tensor(Domain_mesh_u.GetCellIds(PlotCoordinates),dtype=torch.int)
IDs_du = torch.tensor(Domain_mesh_du.GetCellIds(PlotCoordinates),dtype=torch.int)

CoordinatesBatchSet = torch.utils.data.DataLoader([[PlotCoordinates[i], IDs_u[i], IDs_du[i]] for i in range((IDs_u.shape)[0])], batch_size=100, shuffle=True)

print("Number of training points = ", PlotCoordinates.shape[0])
print("Batch size = ", CoordinatesBatchSet.batch_size)
print("Number of batches per epoch = ", len(CoordinatesBatchSet))
print()


####################################################################
print("Model u")
Model_u = MeshNN(Domain_mesh_u, 2)                # Create the associated model
Model_u.UnFreeze_Values()
Model_u.Freeze_Mesh()

print()
print("Model du")
Model_du = MeshNN(Domain_mesh_du, 3)                # Create the associated model
Model_du.UnFreeze_Values()
Model_du.Freeze_Mesh()

#print("Model du param")
#for param in Model_du.parameters():
#    if param.requires_grad == True:
#        print(param)

####################################################


u_predicted = Model_u(PlotCoordinates, IDs_u) 
du_predicted = Model_du(PlotCoordinates, IDs_du) 

Pplot.Plot2Dresults(u_predicted, n_train, 5*n_train, "_u_Initial")
#Pplot.Plot2Dresults_Derivative(du_predicted, s11, s22, s12, n_train, 5*n_train, "_Stress_Initial")

optimizer = torch.optim.Adam(list(Model_u.parameters())+list(Model_du.parameters()))

w0 = numpy.sqrt(10*50)
w1 = 1

n_epochs = 50

print("**************** START TRAINING 1st stage ***************\n")

Model_u, Model_du, loss = GradDescend_Stage1_2D(Model_u, Model_du, IDs_u, IDs_du, PlotCoordinates, CoordinatesBatchSet, w0, w1, n_epochs, optimizer, n_train)


print("*************** 2nd stage LBFGS ***************\n")

Model_u, Model_du = LBFGS_Stage2_2D(Model_u, Model_du, IDs_u, IDs_du, PlotCoordinates, w0, w1, n_train)



num_sol_name = "Rectangle_order_1_2.5_order2_displacement.npy"

NumSol_eval(Domain_mesh_u, Domain_mesh_du, Model_u, Model_du, num_sol_name, L)