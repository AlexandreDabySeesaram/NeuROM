import fenics as fenics
import numpy as numpy
import matplotlib.pyplot as plt
import dolfin_mech as dmech

from dolfin import *

# Lame's constants
lmbda = 1.25
mu = 1.0
# Density
rho = 1.0

Ly = 50
g_y = 1.0

#-----------------------

# Bottom part of boundary
def bottom(x, on_boundary):
    return (on_boundary and fenics.near(x[1], 0.0))


# Strain function
def epsilon(u):
    return 0.5*(fenics.grad(u) + fenics.grad(u).T)


# Stress function
def sigma(u):
    return lmbda*fenics.div(u)*fenics.Identity(2) + 2*mu*epsilon(u)

#-----------------------


g = fenics.Constant((0.0, g_y))

#-----------------------

mesh_name = "Rectangle_order_1_2.5.xml"
mesh = fenics.Mesh("mesh/"+ mesh_name)
print(mesh)
# --------------------
# Function spaces
# --------------------

V = fenics.VectorFunctionSpace(mesh, "CG", 1)       # Continuous Galerkin ~ standard Lagrange family of elements
V0 = fenics.TensorFunctionSpace(mesh, "DG", 0)
Vs = fenics.TensorFunctionSpace(mesh, "CG", 1)

u_tr = fenics.TrialFunction(V)
u_test = fenics.TestFunction(V)

# --------------------
# Boundary conditions
# --------------------

boundaries = fenics.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
top = fenics.AutoSubDomain(lambda x: fenics.near(x[1], Ly))
top.mark(boundaries, 1)
ds = fenics.ds(subdomain_data=boundaries)

bcs = [fenics.DirichletBC(V, fenics.Constant((0.0, 0.0)), bottom), fenics.DirichletBC(V, fenics.Constant((0.0, 1.0)), top)]


# --------------------
# Initialization
# --------------------
u = fenics.Function(V)


Form = fenics.inner(sigma(u_tr), epsilon(u_test))*fenics.dx #- fenics.inner(g, u_test)*ds(1)

fenics.solve(fenics.lhs(Form) == fenics.rhs(Form), u, bcs)

stress = fenics.project(sigma(u), V0)

dmech.write_VTU_file(
    filebasename=mesh_name[:-4]+"_displacement",
    function=u)

dmech.write_VTU_file(
    filebasename=mesh_name[:-4]+"_stress",
    function=stress)
