#%% Convert gmsh
import meshio
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import meshio
import numpy as np

# load .msh
meshBeam = meshio.read("Beam0.1.msh")

# Convert to VTK
# meshBeam.write("Beam.vtk")
meshio.write("Beam.vtk", meshBeam, file_format="vtk", binary=False)

# Load the VTK mesh
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName("Beam.vtk")  
reader.Update()
meshVTK = reader.GetOutput()

#%% Find element index
vtk_mesh = vtk.vtkPolyData()
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName("Beam.vtk")  
reader.Update()
vtk_mesh = reader.GetOutput()


# Create a vtkCellTreeLocator
locator = vtk.vtkCellLocator()
locator.SetDataSet(vtk_mesh)
locator.Update()


# Find the cell containing the specified point
x = 0.1
y = 0.0

point = [x, y, 0.0]  # Assuming a 2D point, set z-coordinate to 0.0
generic_cell = vtk.vtkGenericCell()
k_cell = vtk.mutable(0)
subId = vtk.mutable(0)
dist = vtk.mutable(0.)

# locator.FindCell(point, 1e-10,subId)
k_cell = locator.FindCell(point)

print("Cell ID:", k_cell)
# %%
