import numpy as np 
import os

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
def PrintWelcome():
    """Neural network reduced-order modelling for mechanics"""
    # Ivrit
    print(" \
\n \
\n \
  _   _            ____   ___  __  __           _     \n \
 | \ | | ___ _   _|  _ \ / _ \|  \/  | ___  ___| |__  \n \
 |  \| |/ _ \ | | | |_) | | | | |\/| |/ _ \/ __| '_ \ \n \
 | |\  |  __/ |_| |  _ <| |_| | |  | |  __/ (__| | | |\n \
 |_| \_|\___|\__,_|_| \_\ ___/|_|  |_|\___|\___|_| |_|\n ")


                                                                                                                                    
class Mesh:
    def __init__(self,name,h_max):
        """inputs the name of the geometry and the maximum size of the element"""
        PrintWelcome()
        self.h_str = str(h_max)
        self.name = name
        self.name_mesh = self.name+self.h_str+'.msh'
        self.name_geo = self.name+'.geo'
    
    def AddBCs(self,Volume,Dirichlets):
        self.VolumeId = Volume
        NumberOfBCs = len(Dirichlets)
        for i in range(NumberOfBCs):
            ListOfDirichletsBCsIds = [Dirichlets[i]["Entity"] for i in range(NumberOfBCs)]
            self.ListOfDirichletsBCsIds = ListOfDirichletsBCsIds
            # ListOfPhysicisBCs.append(Dirichlets[i]["Entity"])
            # self.ListOfPhysicisBCs = list(set(ListOfPhysicisBCs))
        
    def MeshGeo(self):
        path = 'Geometries/'+self.name_mesh
        if os.path.isfile(path):
            pass
        else:
            print('*************** Mesh Geometry  ****************\n' )
            # GMSH is in path but does not appear to be through python os.sytem
            mesh_command = '/Applications/Gmsh.app/Contents/MacOS/gmsh Geometries/'+self.name_geo+' -1 -o '+'Geometries/'+self.name_mesh+' -clmax '+self.h_str
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
            self.Connectivity = np.array(self.Connectivity)
            self.NElem = self.Connectivity.shape[0] # Only count the volume elements
            print(f'\n************ FINISHED READING MESH ************\n\n \
* Dimension of the problem: {self.dim}D\n \
* Elements type:            {self.type}\n \
* Number of Elements:       {self.Connectivity.shape[0]}\n \
* Number of free Dofs:      {self.NNodes-len(self.ListOfDirichletsBCsIds)}\n')
    

    def AssemblyMatrix(self):
        import torch
        weights_assembly = torch.zeros(self.dim*self.NNodes,self.node_per_elem*self.Connectivity.shape[0])
        elem_range = np.arange(self.Connectivity.shape[0])
        ne_values = np.arange(self.node_per_elem)
        i_values = self.Connectivity[:, ne_values]-1 
        j_values = 2 * (elem_range[:, np.newaxis])+ ne_values 
        weights_assembly[i_values.flatten().astype(int), j_values.flatten().astype(int)] = 1
        self.weights_assembly = weights_assembly
        #For 1D elements, add phantom elements assembly:
        weights_assembly_phantom = np.zeros((weights_assembly.shape[0],4))
        weights_assembly_phantom[0,1] = 1
        weights_assembly_phantom[1,-2] = 1
        self.weights_assembly_total = np.concatenate((weights_assembly,weights_assembly_phantom),axis=1)