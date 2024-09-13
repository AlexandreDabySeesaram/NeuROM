import numpy as np 
import os
import subprocess
import meshio
import vtk

"""
Cheat sheet msh format 2.2
$MeshFormat
version-number file-type data-size
$EndMeshFormat
$Nodes
number-of-nodes
node-number x-coord y-coord z-coord
…
$EndNodes
$Elements
number-of-elements
elm-number elm-type number-of-tags < tag > … node-number-list
…
$EndElements
$PhysicalNames
number-of-names
phyical-number "physical-name"
…
$EndPhysicalNames
https://www.manpagez.com/info/gmsh/gmsh-2.2.6/gmsh_63.php
"""
def get_git_tag() -> str:
    try:
        return subprocess.check_output(['git', 'describe','--tag', '--abbrev=0']).decode('ascii').strip()
    except:
        return 'unknown version'

def PrintWelcome():
    version = get_git_tag()
    """Neural network reduced-order modelling for mechanics"""
#     # Ivrit NeuROMech
#     print(" \
# \n \
# \n \
#   _   _            ____   ___  __  __           _     \n \
#  | \ | | ___ _   _|  _ \ / _ \|  \/  | ___  ___| |__  \n \
#  |  \| |/ _ \ | | | |_) | | | | |\/| |/ _ \/ __| '_ \ \n \
#  | |\  |  __/ |_| |  _ <| |_| | |  | |  __/ (__| | | |\n \
#  |_| \_|\___|\__,_|_| \_\ ___/|_|  |_|\___|\___|_| |_|\n ")

     # Ivrit NeuROM
    print(" \
\n \
\n \
  _   _            ____   ___  __  __ \n \
 | \ | | ___ _   _|  _ \ / _ \|  \/  |\n \
 |  \| |/ _ \ | | | |_) | | | | |\/| |\n \
 | |\  |  __/ |_| |  _ <| |_| | |  | |\n \
 |_| \_|\___|\__,_|_| \_\ ___/|_|  |_|\n\n " + \
"                 "+version)

class Material():
    def __init__(self,flag_lame,coef1,coef2):
        if flag_lame:
            self.lmbda = coef1                                               # First Lamé's coefficient
            self.mu = coef2                                                  # Second Lamé's coefficient
            self.E = self.mu*(3*self.lmbda+2*self.mu)/(self.lmbda+self.mu)                            # Young's modulus
            self.nu = self.lmbda/(2*(self.lmbda+self.mu))                                   # Poisson's ratio
        else:
            self.E = coef1                                                   # Young's modulus (GPa)
            self.nu = coef2                                                  # Poisson's ratio
            self.lmbda = (self.E*self.nu)/((1+self.nu)*(1-2*self.nu))                            # First Lamé's coefficient
            self.mu = self.E/(2*(1+self.nu))                                           # Second Lamé's coefficient
                                                                                                                                    
def ElementSize(dimension, **kwargs):
    if dimension ==1:
        L = kwargs['L']
        order = kwargs['order']
        np = kwargs['np']
        if order ==1:
            MaxElemSize = L/(np-1)                         # Compute element size
        elif order ==2:
            n_elem = 0.5*(np-1)
            MaxElemSize = L/n_elem                         # Compute element size
    else:
        MaxElemSize = kwargs['MaxElemSize2D']
    return MaxElemSize
    
class Mesh:
    def __init__(self, name, h_max, order, dimension, welcome =True):
        """inputs the name of the geometry and the maximum size of the element"""
        if welcome:
            PrintWelcome()
        self.h_max_str = str(np.around(h_max, decimals=3))

        self.order = str(order)
        self.dimension = str(dimension)
        self.name = name
        self.name_mesh = self.name+'_order_'+self.order+'_'+self.h_max_str+'.msh'
        self.name_geo = self.name+'.geo'
        self.borders_exist = False
        if not os.path.isdir("Results"):
            subprocess.run(["mkdir", "Results"])
        if not os.path.isdir("Results/Paraview"):
            subprocess.run(["mkdir", "Results/Paraview"])
        if not os.path.isdir("Results/Paraview/TimeSeries/"):
            subprocess.run(["mkdir", "Results/Paraview/TimeSeries/"])
    
    def AddBorders(self,borders):
        self.borders = borders
        self.borders_nodes = []
        self.borders_exist = True

    def AddBCs(self,Volume,Exclude,Dirichlets):
        self.VolumeId = Volume
        self.ExcludeId = Exclude
        NumberOfBCs = len(Dirichlets)

        if NumberOfBCs == 1:
            if len(Dirichlets[0])==0:                        # Empty list : Dirichlets =  [{}] ... len = 1
                self.NoBC = True
        else:
            self.NoBC = False

        if self.NoBC == False:
            ListOfDirichletsBCsIds = [Dirichlets[i]["Entity"] for i in range(NumberOfBCs)]
            ListOfDirichletsBCsValues = [Dirichlets[i]["Value"] for i in range(NumberOfBCs)]
            ListOfDirichletsBCsNormals = [Dirichlets[i]["Normal"] for i in range(NumberOfBCs)]
            ListOfDirichletsBCsRelation = [Dirichlets[i]["Relation"] for i in range(NumberOfBCs)]
            ListOfDirichletsBCsConstit = [Dirichlets[i]["Constitutive"] for i in range(NumberOfBCs)]
        
            self.ListOfDirichletsBCsIds = ListOfDirichletsBCsIds
            self.ListOfDirichletsBCsValues = ListOfDirichletsBCsValues
            self.ListOfDirichletsBCsNormals = ListOfDirichletsBCsNormals
            self.ListOfDirichletsBCsRelation = ListOfDirichletsBCsRelation
            self.ListOfDirichletsBCsConstit = ListOfDirichletsBCsConstit
        else:
            self.ListOfDirichletsBCsIds = []
            self.ListOfDirichletsBCsValues = []
            self.ListOfDirichletsBCsNormals = []
            self.ListOfDirichletsBCsRelation = []
            self.ListOfDirichletsBCsConstit = []

        if len(self.ExcludeId)==0:
            self.NoExcl = True
        else:
            self.NoExcl = False

    def MeshGeo(self):
        path = 'Geometries/'+self.name_mesh
        if os.path.isfile(path):
            pass
        else:
            print('*************** Mesh Geometry  ****************\n' )
            # GMSH is in path but does not appear to be through python os.sytem
            # -1 = Perform 1D mesh generation
            try:
                mesh_command = 'gmsh Geometries/'+self.name_geo+ \
                        ' -'+self.dimension+' -order '+self.order+' -o '+'Geometries/'+self.name_mesh+  \
                        ' -clmax '+self.h_max_str  
                        ##' -clmin '+self.h_min_str + '- algo delquad'
                os.system(mesh_command)
            except:
                mesh_command = '/Applications/Gmsh.app/Contents/MacOS/gmsh Geometries/'+self.name_geo+ \
                        ' -'+self.dimension+' -order '+self.order+' -o '+'Geometries/'+self.name_mesh+  \
                        ' -clmax '+self.h_max_str  
                        ##' -clmin '+self.h_min_str + '- algo delquad'
                os.system(mesh_command)
         
    
    def ReadMesh(self):
        with open('Geometries/'+self.name_mesh) as mshfile:
            line = mshfile.readline()
            line = mshfile.readline()
            self.MshFormat = line
            line = mshfile.readline()
            line = mshfile.readline()
            line = mshfile.readline()
            self.NNodes = int(line.strip())
            self.Nodes = []
            for node in range(self.NNodes):
                line = mshfile.readline()
                coordinates = line.split()  # Split the line at each space
                NodesList = [float(coordinate) for coordinate in coordinates]
                self.Nodes.append(NodesList)
            line = mshfile.readline()
            line = mshfile.readline()
            line = mshfile.readline()
            self.NElem = int(line)
            self.Connectivity = []

            if self.NoBC == False:
                self.DirichletBoundaryNodes = [[] for id in self.ListOfDirichletsBCsValues]

            self.ExcludedPoints = []

            flagType = True
            for elem in range(self.NElem):
                line = mshfile.readline()
                Elems = line.split()  # Split the line at each space
                ElemList = [float(Elem_item) for Elem_item in Elems]
                if ElemList[3] == self.VolumeId:
                    if flagType:
                        match ElemList[1]:
                            case 1:
                                self.type = "2-node bar"
                                self.dim = 1
                                self.node_per_elem = 2
                            case 2:
                                self.type = "t3: 3-node triangle"
                                self.dim = 2
                                self.node_per_elem = 3
                            case 3:
                                self.type = "4-node quadrangle"
                                self.dim = 2
                                self.node_per_elem = 4
                            case 4:
                                self.type = "4-node tetrahedron"
                                self.dim = 3
                                self.node_per_elem = 4
                            case 8:
                                self.type = "3-node quadratic bar"
                                self.dim = 1
                                self.node_per_elem = 3
                            case 9:
                                self.type = "T6: 6-node quadratic traingle"
                                self.dim = 2
                                self.node_per_elem = 6
                        flagType = False  
                    self.Connectivity.append(ElemList[-self.node_per_elem:])  

                if ElemList[3] in self.borders: 
                    match ElemList[1]:
                        case 1:
                            node_per_elem = 2
                        case 2:
                            node_per_elem = 3
                        case 8:
                            node_per_elem = 3
                        case 9:
                            node_per_elem = 6
                        case 15:
                            node_per_elem = 1
                    self.borders_nodes.append(ElemList[-node_per_elem:])  

                if self.NoBC == False:
                    for ID_idx in range(len(self.ListOfDirichletsBCsIds)):
                        if ElemList[3] == self.ListOfDirichletsBCsIds[ID_idx]: 
                            match ElemList[1]:
                                case 1:
                                    BCs_type = "2-node bar"
                                    BCs_dim = 1
                                    BCs_node_per_elem = 2
                                case 2:
                                    BCs_node_per_elem = 3
                                case 8:
                                    BCs_type = "3-node quadratic bar"
                                    BCs_dim = 1
                                    BCs_node_per_elem = 3
                                case 9:
                                    BCs_node_per_elem = 6
                                case 15:
                                    BCs_type = "point"
                                    BCs_dim = 1
                                    BCs_node_per_elem = 1
                            self.DirichletBoundaryNodes[ID_idx].append(ElemList[-BCs_node_per_elem:])  

                if self.NoExcl == False:
                    for ID_idx in range(len(self.ExcludeId)):
                        if ElemList[3] == self.ExcludeId[ID_idx]: 
                            self.ExcludedPoints.append(ElemList[-1:][0]-1)


            self.Connectivity = np.array(self.Connectivity)
            self.NElem = self.Connectivity.shape[0] # Only count the volume elements
            np.save( 'Geometries/'+ self.name_mesh[:-4]+"_all_nodes.npy", np.array(self.Nodes))

            print(f'\n************ MESH READING COMPLETE ************\n\n \
* Dimension of the problem: {self.dim}D\n \
* Elements type:            {self.type}\n \
* Number of Dofs:           {self.NNodes*int(self.dimension)}')
# * No excluded points:          {self.NoExcl}')
#* Number of free Dofs:      {self.NNodes-len(self.ListOfDirichletsBCsIds)}\n')
# * Number of Elements:       {self.Connectivity.shape[0]}\n \



    def ExportMeshVtk(self,flag_update = False):
        msh_name = 'Geometries/'+self.name_mesh
        meshBeam = meshio.read(msh_name)

        if flag_update:
            points = np.array(self.Nodes)[:,1:]
            cells = (self.Connectivity-1).astype(np.int32)
        else:
            points = meshBeam.points
            match self.type:
                case 't3: 3-node triangle':
                    cells = meshBeam.cells_dict["triangle"]
                case '4-node tetrahedron':
                    cells = meshBeam.cells_dict["tetra"]

        # create meshio mesh based on points and cells from .msh file
        
        match self.type:
            case 't3: 3-node triangle':
                if self.order =='1':
                    mesh = meshio.Mesh(points, {"triangle":cells})
                    meshio.write(msh_name[0:-4]+".vtk", mesh, binary=True )
                    # mesh = meshio.Mesh(points[:,:2], {"triangle":cells})
                    # meshio.write(msh_name[0:-4]+".xml", mesh)

                elif self.order =='2':
                    mesh = meshio.Mesh(points, {"triangle6":meshBeam.cells_dict["triangle6"]})
                    meshio.write(msh_name[0:-4]+".vtk", mesh, binary=True )

                    # mesh = meshio.Mesh(points[:,:2], {"triangle":meshBeam.cells_dict["triangle6"][:,0:3]})
                    # meshio.write(msh_name[0:-4]+".xml", mesh)
            case '4-node tetrahedron':
                if self.order =='1':
                    mesh = meshio.Mesh(points, {"tetra":cells})
                    meshio.write(msh_name[0:-4]+".vtk", mesh, binary=True )
                else:
                    raise ValueError("Only first order element have been implemented in 1D for now")

        # Load the VTK mesh
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(msh_name[0:-4]+".vtk",)  
        reader.Update()
        self.vtk_mesh = reader.GetOutput()

    def ExportMeshVtk1D(self,flag_update = False):

        msh_name = 'Geometries/'+self.name_mesh
        meshBeam = meshio.read(msh_name)

        if flag_update:
            points = np.array(self.Nodes)[:,1:]
        else:
            points = meshBeam.points

        # crete meshio mesh based on points and cells from .msh file

        if self.order =='1':
            mesh = meshio.Mesh(points, {"line":meshBeam.cells_dict["line"]})
            meshio.write(msh_name[0:-4]+".vtk", mesh, binary=True )
               
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(msh_name[0:-4]+".vtk",)  
        reader.Update()
        self.vtk_mesh = reader.GetOutput()



    def ReadNormalVectors(self):
        print(" * Read normal vectors")
        normals = np.load('Geometries/'+self.name_mesh[0:-4]+"_normals.npy")
        norm = np.linalg.norm(normals, axis=1)
        self.normals = normals/norm[:,None]

    def AssemblyMatrix(self):
        import torch

        if self.dimension =='1':
            if self.order =='1':
                weights_assembly = torch.zeros(self.dim*self.NNodes,self.node_per_elem*self.Connectivity.shape[0])
                self.assembly_vector = torch.zeros(self.dim*self.NNodes)
                elem_range = np.arange(self.Connectivity.shape[0])
                ne_values = np.arange(self.node_per_elem) # {[N1 N2] [N2 N3] [N3 N4]}
                ne_values_j = np.array([1,0]) # Katka's left right implementation {[N2 N1] [N3 N2] [N4 N3]} otherwise same as ne_value
                i_values = self.Connectivity[:, ne_values]-1 
                j_values = 2 * (elem_range[:, np.newaxis])+ ne_values_j 
                weights_assembly[i_values.flatten().astype(int), j_values.flatten().astype(int)] = 1
                self.weights_assembly = weights_assembly
                #For 1D elements, add phantom elements assembly:
                weights_assembly_phantom = np.zeros((weights_assembly.shape[0],4))
                # weights_assembly_phantom[0,1] = 1  #  {[N1 N2] [N2 N3] [N3 N4]}
                # weights_assembly_phantom[1,-2] = 1 # {[N1 N2] [N2 N3] [N3 N4]}
                weights_assembly_phantom[0,0] = 1  # Katka's left right implementation {[N2 N1] [N3 N2] [N4 N3]}
                weights_assembly_phantom[1,-1] = 1 # Katka's left right implementation {[N2 N1] [N3 N2] [N4 N3]}
                self.weights_assembly_total = np.concatenate((weights_assembly,weights_assembly_phantom),axis=1)
                idx = np.where(np.sum((self.weights_assembly_total),axis=1)==2)
                self.assembly_vector[idx]=-1

            elif self.order =='2':
                weights_assembly = torch.zeros(self.dim*self.NNodes,self.node_per_elem*self.Connectivity.shape[0])
                self.assembly_vector = torch.zeros(self.dim*self.NNodes)

                elem_range = np.arange(self.Connectivity.shape[0])
                ne_values = np.arange(self.node_per_elem) # {[N1 N2] [N2 N3] [N3 N4]}
                ne_values_j = np.array([1,0,2]) # Katka's left right implementation {[N2 N1] [N3 N2] [N4 N3]} otherwise same as ne_value
                i_values = self.Connectivity[:, ne_values]-1 
                j_values = self.node_per_elem * (elem_range[:, np.newaxis])   + ne_values_j 
                weights_assembly[i_values.flatten().astype(int), j_values.flatten().astype(int)] = 1
                self.weights_assembly = weights_assembly
                #For 1D elements, add phantom elements assembly:
                weights_assembly_phantom = np.zeros((weights_assembly.shape[0],4))
                weights_assembly_phantom[0,0] = 1  # Katka's left right implementation {[N2 N1] [N3 N2] [N4 N3]}
                weights_assembly_phantom[1,-1] = 1 # Katka's left right implementation {[N2 N1] [N3 N2] [N4 N3]}

                self.weights_assembly_total = np.concatenate((weights_assembly,weights_assembly_phantom),axis=1)
                idx = np.where(np.sum((self.weights_assembly_total),axis=1)==2)
                self.assembly_vector[idx]=-1
                
        if self.dimension =='2':
            if self.order =='1':
                weights_assembly = torch.zeros(self.dim*self.NNodes,self.node_per_elem*self.Connectivity.shape[0])
                self.assembly_vector = torch.zeros(self.dim*self.NNodes)
                elem_range = np.arange(self.Connectivity.shape[0])
                ne_values = np.arange(self.node_per_elem) # {[N1 N2] [N2 N3] [N3 N4]}
                ne_values_j = np.array([1,0]) # Katka's left right implementation {[N2 N1] [N3 N2] [N4 N3]} otherwise same as ne_value
                i_values = self.Connectivity[:, ne_values]-1 
                print(i_values)

                j_values = 2 * (elem_range[:, np.newaxis])+ ne_values_j 
                weights_assembly[i_values.flatten().astype(int), j_values.flatten().astype(int)] = 1
                self.weights_assembly = weights_assembly
                #For 1D elements, add phantom elements assembly:
                weights_assembly_phantom = np.zeros((weights_assembly.shape[0],4))
                # weights_assembly_phantom[0,1] = 1  #  {[N1 N2] [N2 N3] [N3 N4]}
                # weights_assembly_phantom[1,-2] = 1 # {[N1 N2] [N2 N3] [N3 N4]}
                weights_assembly_phantom[0,0] = 1  # Katka's left right implementation {[N2 N1] [N3 N2] [N4 N3]}
                weights_assembly_phantom[1,-1] = 1 # Katka's left right implementation {[N2 N1] [N3 N2] [N4 N3]}
                self.weights_assembly_total = np.concatenate((weights_assembly,weights_assembly_phantom),axis=1)
                idx = np.where(np.sum((self.weights_assembly_total),axis=1)==2)
                self.assembly_vector[idx]=-1


    def GetCellIds(self, TrialCoordinates):
        locator = vtk.vtkCellLocator()
        locator.SetDataSet(self.vtk_mesh)
        locator.Update()

        ids = []

        for coord in TrialCoordinates:
            match self.dimension:
                case '1':
                    point = [coord, 0, 0]
                case '2':
                    point = [coord[0], coord[1], 0]
            ids.append(locator.FindCell(point))

        return ids


    # def GetCellIds1D(self, TrialCoordinates):
    #     locator = vtk.vtkCellLocator()
    #     locator.SetDataSet(self.vtk_mesh)
    #     locator.Update()

    #     ids = []

    #     for coord in TrialCoordinates:
    #         point = [coord, 0, 0]
    #         ids.append(locator.FindCell(point))

    #     return ids