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
    return subprocess.check_output(['git', 'describe', '--abbrev=0']).decode('ascii').strip()

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
                                                                                                                                    
class Mesh:
    def __init__(self,name,h_max, order, dimension):
        """inputs the name of the geometry and the maximum size of the element"""
        PrintWelcome()
        self.h_str = str(np.around(h_max, decimals=3))
        self.order = str(order)
        self.dimension = str(dimension)
        self.name = name
        self.name_mesh = self.name+'_order_'+self.order+'_'+self.h_str+'.msh'
        self.name_geo = self.name+'.geo'
        self.borders_exist = False
    
    def AddBorders(self,borders):
        self.borders = borders
        self.borders_nodes = []
        self.borders_exist = True

    def AddBCs(self,Volume,Exclude,Dirichlets):
        self.VolumeId = Volume
        self.ExcludeId = Exclude
        NumberOfBCs = len(Dirichlets)

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

            # ListOfPhysicisBCs.append(Dirichlets[i]["Entity"])
            # self.ListOfPhysicisBCs = list(set(ListOfPhysicisBCs))
        if NumberOfBCs == 0:
            self.NoBC = True
        else:
            self.NoBC = False

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
            mesh_command = '/Applications/Gmsh.app/Contents/MacOS/gmsh Geometries/'+self.name_geo+ \
                    ' -'+self.dimension+' -order '+self.order+' -o '+'Geometries/'+self.name_mesh+' -clmax '+self.h_str
                    #'- algo delquad'
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
                if self.borders_exist:
                    if ElemList[3] in self.borders: 
                        match ElemList[1]:
                            case 1:
                                node_per_elem = 2
                            case 8:
                                node_per_elem = 3
                        self.borders_nodes.append(ElemList[-node_per_elem:])  

                if self.NoBC == False:
                    for ID_idx in range(len(self.ListOfDirichletsBCsIds)):
                        if ElemList[3] == self.ListOfDirichletsBCsIds[ID_idx]: 
                            match ElemList[1]:
                                case 1:
                                    self.type = "2-node bar"
                                    self.dim = 1
                                    self.node_per_elem = 2
                                case 8:
                                    self.type = "3-node quadratic bar"
                                    self.dim = 1
                                    self.node_per_elem = 3
                            self.DirichletBoundaryNodes[ID_idx].append(ElemList[-self.node_per_elem:])  

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
* Number of Elements:       {self.Connectivity.shape[0]}\n \
* No excluded points:          {self.NoExcl}')
#* Number of free Dofs:      {self.NNodes-len(self.ListOfDirichletsBCsIds)}\n')


    def ExportMeshVtk(self,flag_update = False):
        msh_name = 'Geometries/'+self.name_mesh
        meshBeam = meshio.read(msh_name)

        if flag_update:
            points = np.array(self.Nodes)[:,1:]
            cells = (self.Connectivity-1).astype(np.int32)
        else:
            points = meshBeam.points
            cells = meshBeam.cells_dict["triangle"]
        # create meshio mesh based on points and cells from .msh file

        if self.order =='1':
            mesh = meshio.Mesh(points, {"triangle":cells})
            meshio.write(msh_name[0:-4]+".vtk", mesh, binary=False )
            mesh = meshio.Mesh(points[:,:2], {"triangle":cells})
            meshio.write(msh_name[0:-4]+".xml", mesh)

        elif self.order =='2':
            mesh = meshio.Mesh(points, {"triangle6":meshBeam.cells_dict["triangle6"]})
            meshio.write(msh_name[0:-4]+".vtk", mesh, binary=False )

            mesh = meshio.Mesh(points[:,:2], {"triangle":meshBeam.cells_dict["triangle6"][:,0:3]})
            meshio.write(msh_name[0:-4]+".xml", mesh)



        # Load the VTK mesh
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
            point = [coord[0], coord[1], 0]
            ids.append(locator.FindCell(point))

        return ids
