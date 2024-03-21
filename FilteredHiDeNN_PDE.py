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
from scipy import ndimage


from Bin.Training_2D import Mixed_Initial_Training
from Bin.PDE_Library import Mixed_2D_loss

class Dataset(torch.utils.data.Dataset):

    def __init__(self, X):
        self.X = X
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):    
        x = self.X[index]

        return x


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
        cell_nodes_IDs = self.connectivity[cell_id,:] - 1

        node1_value =  torch.stack([torch.cat([val[row] for row in cell_nodes_IDs[:,0]]) for val in nodal_values], dim=0)
        node2_value =  torch.stack([torch.cat([val[row] for row in cell_nodes_IDs[:,1]]) for val in nodal_values], dim=0)
        node3_value =  torch.stack([torch.cat([val[row] for row in cell_nodes_IDs[:,2]]) for val in nodal_values], dim=0)

        #out = torch.cat(shape_functions[:,0]*node1_value[:,0] + shape_functions[:,1]*node2_value[:,0] + shape_functions[:,2]*node3_value[:,0], shape_functions[:,0]*node1_value[:,1] + shape_functions[:,1]*node2_value[:,1] + shape_functions[:,2]*node3_value[:,1])

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

        refCoord = GetRefCoord(x[:,0],x[:,1],node1_coord[:,0],node2_coord[:,0],node3_coord[:,0],node1_coord[:,1],node2_coord[:,1],node3_coord[:,1])
        
        # print("x = ", x[2])
        # print("cell id = ", cell_id[2])
        # print("nodes = ", cell_nodes_IDs[2])
        # print("node1_coord = ", node1_coord[2])
        # print("node2_coord = ", node2_coord[2])
        # print("node3_coord = ", node3_coord[2])
        # print()
        # print("refCoord = ", refCoord[2])
        
        out = torch.stack((refCoord[:,0], refCoord[:,1], refCoord[:,2]),dim=1) #.view(sh_R.shape[0],-1) # Left | Right | Middle

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
        self.ElementBlock = ElementBlock2D_Lin(mesh.Connectivity)
        self.Interpolation = InterpolationBlock(mesh.Connectivity)

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
order = 1                                          # Order of the shape functions
dimension = 2

if order ==1:
    MaxElemSize = L/(np-1)                         # Compute element size
elif order ==2:
    n_elem = 0.5*(np-1)
    MaxElemSize = L/n_elem                         # Compute element size

if dimension ==1:
    Domain_mesh_u = pre.Mesh('Beam',MaxElemSize, order, dimension)    # Create the mesh object
    Domain_mesh_du = pre.Mesh('Beam',MaxElemSize, order, dimension)    # Create the mesh object
if dimension ==2:
    Domain_mesh_u = pre.Mesh('Rectangle',MaxElemSize, order, dimension)    # Create the mesh object
    Domain_mesh_du = pre.Mesh('Rectangle',MaxElemSize, order, dimension)    # Create the mesh object

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
Domain_mesh_u.ReadMeshVtk()

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
Domain_mesh_du.ReadMeshVtk()

####################################################################

n_train = 10

TrailCoord_1d_x = torch.tensor([i for i in torch.linspace(0,L,n_train)],dtype=torch.float64)
TrailCoord_1d_y = torch.tensor([i for i in torch.linspace(0,5*L,5*n_train)],dtype=torch.float64)

PlotCoordinates = torch.tensor(torch.cartesian_prod(TrailCoord_1d_x,TrailCoord_1d_y), requires_grad=True)
IDs_u = torch.tensor(Domain_mesh_u.GetCellIds(PlotCoordinates),dtype=torch.int)
IDs_du = torch.tensor(Domain_mesh_du.GetCellIds(PlotCoordinates),dtype=torch.int)

CoordinatesBatchSet = torch.utils.data.DataLoader([[PlotCoordinates[i], IDs_u[i], IDs_du[i]] for i in range((IDs_u.shape)[0])], batch_size=100, shuffle=True)

print("Number of training points = ", PlotCoordinates.shape[0])
print("Batch size = ", CoordinatesBatchSet.batch_size)
print("Number of batches per epoch = ", len(CoordinatesBatchSet))
print()



mesh_coord = torch.tensor(numpy.load("/Users/skardova/Dropbox/Lungs/HiDeNN_1D/Fenics_solution_2D/Rectangle_nodal_coord.npy"),requires_grad=True)
num_u = torch.tensor(numpy.load("/Users/skardova/Dropbox/Lungs/HiDeNN_1D/Fenics_solution_2D/Rectangle_order_1_2.5_displacement.npy"))
num_stress = torch.tensor(numpy.load("/Users/skardova/Dropbox/Lungs/HiDeNN_1D/Fenics_solution_2D/Rectangle_order_1_2.5_stress.npy"))

mesh_IDs_u = torch.tensor(Domain_mesh_u.GetCellIds(mesh_coord),dtype=torch.int)

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

w0 = 50
w1 = 1

n_epochs = 4000

evaluation_time = 0
loss_time = 0
optimizer_time = 0
backward_time = 0

stagnancy_counter = 0
loss_counter = 0

print("**************** START TRAINING ***************\n")
start_train_time = time.time()
loss_old = 1
loss_current = 1
loss_min = 1
epoch = 0


while epoch<n_epochs and (loss_counter<2 or loss_current > 1.0e-3): #and stagnancy_counter < 50 :

    for DataSet in CoordinatesBatchSet:

        TrialCoordinates =  DataSet[0]
        TrialIDs_u = DataSet[1]
        TrialIDs_du = DataSet[2]

        start_time = time.time()
        u_predicted = Model_u(TrialCoordinates, TrialIDs_u) 
        du_predicted = Model_du(TrialCoordinates, TrialIDs_du) 
        evaluation_time += time.time() - start_time

        start_time = time.time()
        l_pde, l_compat, _, _, _ =  Mixed_2D_loss(u_predicted[0,:], u_predicted[1,:],
                                                    du_predicted[0,:], du_predicted[1,:], du_predicted[2,:], 
                                                    TrialCoordinates, lmbda = 1.25, mu = 1.0)
        l =  w0*l_pde +w1*l_compat
        loss_time += time.time() - start_time

        start_time = time.time()
        l.backward()
        backward_time += time.time() - start_time

        start_time = time.time()
        optimizer.step()
        optimizer_time += time.time() - start_time

        optimizer.zero_grad()

        # loss_decrease = (loss_old - loss_current)/numpy.abs(loss_old)
        # loss_old = loss_current

        # if loss_decrease >= 0 and loss_decrease < 1.0e-7:
        #     stagnancy_counter = stagnancy_counter +1
        # else:
        #     stagnancy_counter = 0

    if (epoch+1) % 10 == 0:

        u_predicted = Model_u(PlotCoordinates, IDs_u) 
        du_predicted = Model_du(PlotCoordinates, IDs_du) 

        l_pde, l_compat, s11, s22, s12 =  Mixed_2D_loss(u_predicted[0,:], u_predicted[1,:],
                                                        du_predicted[0,:], du_predicted[1,:], du_predicted[2,:], 
                                                        PlotCoordinates, lmbda = 1.25, mu = 1.0)
        print(epoch+1)
        #print("    loss = ", l_pde.item()+l_compat.item())
        print("    loss_counter = ", loss_counter)
        print("     loss PDE = ", l_pde.item())
        print("     loss compatibility = ", l_compat.item())

        loss[0].append(l_pde.item())
        loss[1].append(l_compat.item())

        loss_current = l_pde.item()+l_compat.item()
        if loss_min > loss_current:
            loss_min = loss_current
            loss_counter = 0
        else:
            loss_counter += 1

        if (epoch+1) % 50 == 0:
            Pplot.Plot2Dresults(u_predicted, n_train, 5*n_train, "_u_Stage_1")
            Pplot.Plot2Dresults_Derivative(du_predicted, s11, s22, s12, n_train, 5*n_train, "_Stress_Stage_1")
            Pplot.Plot2DLoss(loss)


    epoch = epoch+1

stopt_train_time = time.time()

#print("loss_current = ", loss_current)
#print("loss_counter = ", loss_counter)

print("*************** END OF TRAINING ***************\n")
print(f'* Training time: {stopt_train_time-start_train_time}s\n\
    * Evaluation time: {evaluation_time}s\n\
    * Loss time: {loss_time}s\n\
    * Backward time: {backward_time}s\n\
    * Training time per epochs: {(stopt_train_time-start_train_time)/n_epochs}s\n\
    * Optimiser time: {optimizer_time}s\n')

u_predicted = Model_u(PlotCoordinates, IDs_u) 
du_predicted = Model_du(PlotCoordinates, IDs_du) 

l_pde, l_compat, s11, s22, s12 =  Mixed_2D_loss(u_predicted[0,:], u_predicted[1,:],
                                                du_predicted[0,:], du_predicted[1,:], du_predicted[2,:],
                                                PlotCoordinates, lmbda = 1.25, mu = 1.0)
l =  l_pde +l_compat
print(epoch)
print("     loss PDE = ", l_pde.item())
print("     loss compatibility = ", l_compat.item())
print()

Pplot.Plot2Dresults(u_predicted, n_train, 5*n_train, "_u_Stage_1")
Pplot.Plot1DSection(u_predicted, n_train, 5*n_train, "_Stage_1")
Pplot.Plot2Dresults_Derivative(du_predicted, s11, s22, s12, n_train, 5*n_train, "_Stress_Stage_1")
Pplot.Plot2DLoss(loss)

print("*************** 2nd stage LBFGS ***************\n")

optim = torch.optim.LBFGS(list(Model_u.parameters())+list(Model_du.parameters()),
                history_size=5, 
                max_iter=15, 
                tolerance_grad = 1.0e-9,
                line_search_fn="strong_wolfe")

while stagnancy_counter < 5:

    def closure():
        optim.zero_grad()

        u_predicted = Model_u(PlotCoordinates, IDs_u) 
        du_predicted = Model_du(PlotCoordinates, IDs_du) 

        l_pde, l_compat, s11, s22, s12 =  Mixed_2D_loss(u_predicted[0,:], u_predicted[1,:],
                                                        du_predicted[0,:], du_predicted[1,:], du_predicted[2,:], 
                                                        PlotCoordinates, lmbda = 1.25, mu = 1.0)
        l =  w0*l_pde +w1*l_compat

        l.backward()
        return l

    
    optim.step(closure)
    l = closure()
    loss_current = l.item()
    loss_decrease = (loss_old - loss_current)/numpy.abs(loss_old)
    loss_old = loss_current

    print("     Loss = ", l.item())

    if loss_decrease >= 0 and loss_decrease < 1.0e-7:
        stagnancy_counter = stagnancy_counter +1
    else:
        stagnancy_counter = 0


u_predicted = Model_u(PlotCoordinates, IDs_u) 
du_predicted = Model_du(PlotCoordinates, IDs_du) 

l_pde, l_compat, s11, s22, s12 =  Mixed_2D_loss(u_predicted[0,:], u_predicted[1,:],
                                                du_predicted[0,:], du_predicted[1,:], du_predicted[2,:],
                                                PlotCoordinates, lmbda = 1.25, mu = 1.0)
l =  l_pde +l_compat
print()
print("     loss PDE = ", l_pde.item())
print("     loss compatibility = ", l_compat.item())

#Model_u.CheckBCValues()
Pplot.Plot2Dresults(u_predicted, n_train, 5*n_train, "_u_Final")
Pplot.Plot1DSection(u_predicted, n_train, 5*n_train, "_Final")
Pplot.Plot2Dresults_Derivative(du_predicted, s11, s22, s12, n_train, 5*n_train, "_Stress_Final")
Pplot.Plot2DLoss(loss)

print()
print("*************** Evaluation wrt. Numerical solution ***************\n")

u_predicted_x = torch.tensor(Model_u.nodal_values[0])
u_predicted_y = torch.tensor(Model_u.nodal_values[1])

norm_num_ux = torch.norm(num_u[:,0])
norm_num_uy = torch.norm(num_u[:,1])

L2_diff_ux = torch.norm(u_predicted_x.detach() - num_u[:,0])
L2_diff_uy = torch.norm(u_predicted_y.detach() - num_u[:,1])

# print("ux: |NN - Num| = " , L2_diff_ux.item())
# print("uy: |NN - Num| = " , L2_diff_uy.item())
# print()
print("ux: |NN - Num|/|Num| = " , (L2_diff_ux/norm_num_ux).item())
print("uy: |NN - Num|/|Num| = " , (L2_diff_uy/norm_num_uy).item())
print()

MSE_ux = torch.mean((u_predicted_x.detach() - num_u[:,0])**2)
MSE_uy = torch.mean((u_predicted_y.detach() - num_u[:,1])**2)

print("ux: MSE(NN , Num) = " , MSE_ux.item())
print("uy: MSE(NN , Num) = " , MSE_uy.item())
print()


TrailCoord_1d_x = torch.tensor([i for i in torch.linspace(int(L/4),int(3*L/4),n_train)],dtype=torch.float64)
TrailCoord_1d_y = torch.tensor([i for i in torch.linspace(int(5*L/4),int(3*5*L/4),int(5*n_train/2))],dtype=torch.float64)

PlotCoordinates = torch.tensor(torch.cartesian_prod(TrailCoord_1d_x,TrailCoord_1d_y), requires_grad=True)
IDs_u = torch.tensor(Domain_mesh_u.GetCellIds(PlotCoordinates),dtype=torch.int)
IDs_du = torch.tensor(Domain_mesh_du.GetCellIds(PlotCoordinates),dtype=torch.int)

u_predicted = Model_u(PlotCoordinates, IDs_u) 
du_predicted = Model_du(PlotCoordinates, IDs_du) 

l_pde, l_compat, s11, s22, s12 =  Mixed_2D_loss(u_predicted[0,:], u_predicted[1,:],
                                                du_predicted[0,:], du_predicted[1,:], du_predicted[2,:],
                                                PlotCoordinates, lmbda = 1.25, mu = 1.0)




num_nodal_values_x = nn.ParameterList([nn.Parameter(torch.tensor([i[0]])) for i in num_u])
num_nodal_values_y = nn.ParameterList([nn.Parameter(torch.tensor([i[1]])) for i in num_u])

Model_u.nodal_values = [num_nodal_values_x, num_nodal_values_y]

u_predicted = Model_u(PlotCoordinates, IDs_u) 
du_predicted = Model_du(PlotCoordinates, IDs_du) 

l_pde, l_compat, num_s11, num_s22, num_s12 =  Mixed_2D_loss(u_predicted[0,:], u_predicted[1,:],
                                                du_predicted[0,:], du_predicted[1,:], du_predicted[2,:],
                                                PlotCoordinates, lmbda = 1.25, mu = 1.0)

norm_num_s11 = torch.norm(num_s11)
norm_num_s22 = torch.norm(num_s22)
norm_num_s12 = torch.norm(num_s12)

L2_diff_s11 = torch.norm(num_s11 - s11)
L2_diff_s22 = torch.norm(num_s22 - s22)
L2_diff_s12 = torch.norm(num_s12 - s12)

print("s11: |NN - Num| = " , L2_diff_s11.item())
print("s22: |NN - Num| = " , L2_diff_s22.item())
print("s12: |NN - Num| = " , L2_diff_s12.item())

# print()
# print("s11: |NN - Num|/|Num| = " , (L2_diff_s11/norm_num_s11).item())
# print("s22: |NN - Num|/|Num| = " , (L2_diff_s22/norm_num_s22).item())
# print("s12: |NN - Num|/|Num| = " , (L2_diff_s12/norm_num_s12).item())


print()

MSE_s11 = torch.mean((num_s11 - s11))
MSE_s22 = torch.mean((num_s22 - s22))
MSE_s12 = torch.mean((num_s12 - s12))

print("s11: mean(NN - Num) = " , MSE_s11.item())
print("s22: mean(NN - Num) = " , MSE_s22.item())
print("s12: mean(NN - Num) = " , MSE_s12.item())

print()





# img = numpy.reshape(s22.detach(), (n_train, 5*n_train), order='C') 
# plt.imshow(ndimage.rotate(img, 90)) # , vmin = 0, vmax = 1)
# plt.colorbar()
# plt.savefig('Results/pom1.pdf', transparent=True) 
# plt.close()

# img = numpy.reshape(num_s22.detach(), (n_train, 5*n_train), order='C') 
# plt.imshow(ndimage.rotate(img, 90)) # , vmin = 0, vmax = 1)
# plt.colorbar()
# plt.savefig('Results/pom2.pdf', transparent=True) 
# plt.close()