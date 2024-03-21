import myVTKPythonLibrary as myvtk
import numpy as numpy

name = "Rectangle_order_1_2.5_displacement.vtu"

mesh_u = myvtk.readUGrid( "num_solution/"+name)

dataArray = mesh_u.GetPoints().GetData()
nodal_displacement = numpy.zeros((mesh_u.GetNumberOfPoints(),2))

u =  mesh_u.GetPointData().GetArray("f_14")
print("nodes = ", mesh_u.GetNumberOfPoints())

for i in range(mesh_u.GetNumberOfPoints()):

    for j in range(2):
        nodal_displacement[i,j] = dataArray.GetComponent(i,j)

print(nodal_displacement)

numpy.save(name[:-4]+".npy", node)
