import numpy as np 
import os

class Mesh:
    def __init__(self,name,h_max):
        """inputs the name of the geometry and the maximum size of the element"""
        self.h_str = str(h_max)
        self.name = name
        self.name_mesh = self.name+self.h_str+'.msh'
        self.name_geo = self.name+'.geo'
    
    def AddBCs(self,Volume,Dirichlets):
        self.VolumeId = Volume
        NumberOfBCs = len(Dirichlets)
        for i in range(NumberOfBCs):
            ListOfPhysicisBCs = [Dirichlets[i]["Entity"] for i in range(NumberOfBCs)]
            self.ListOfPhysicisBCs = ListOfPhysicisBCs
            # ListOfPhysicisBCs.append(Dirichlets[i]["Entity"])
            # self.ListOfPhysicisBCs = list(set(ListOfPhysicisBCs))
        
    def MeshGeo(self):
        path = '../Geometries/'+self.name_mesh
        if os.path.isfile(path):
            pass
        else:
            # GMSH is in path but does not appear to be through python os.sytem
            mesh_command = '/Applications/Gmsh.app/Contents/MacOS/gmsh Geometries/'+self.name_geo+' -1 -o '+'Geometries/'+self.name_mesh+' -clmax '+self.h_str
            os.system(mesh_command)
            
    
    def ReadMesh(self):
        with open(self.name_mesh) as mshfile:
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
            self.Elem = []
            for elem in range(self.NElem):
                line = mshfile.readline()
                Elems = line.split()  # Split the line at each space
                ElemList = [float(Elem_item) for Elem_item in Elems]
                if ElemList[3] == self.VolumeId:
                    self.Elem.append(ElemList)   