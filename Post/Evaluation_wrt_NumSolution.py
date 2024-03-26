#%% Libraries import
import torch
torch.set_default_dtype(torch.float32)
#Import post processing libraries
import Post.Plots as Pplot
import matplotlib.pyplot as plt
import numpy
import torch.nn as nn

from Bin.PDE_Library import Mixed_2D_loss


def Read_NumSol(mesh, num_sol_name):
    mesh_coord = torch.tensor(mesh.Nodes, requires_grad = True)
    nodal_values = torch.tensor(numpy.load("/Users/skardova/Dropbox/Lungs/HiDeNN_1D/Fenics_solution_2D/"+num_sol_name))
    mesh_IDs_u = torch.tensor(mesh.GetCellIds(mesh_coord),dtype=torch.int)
    return mesh_coord, mesh_IDs_u, nodal_values


def NumSol_eval(mesh_u, mesh_du, Model_u, Model_du, num_sol_name, L):

    print()
    print("*************** Evaluation wrt. Numerical solution ***************\n")

    mesh_coord, mesh_IDs_u, nodal_values = Read_NumSol(mesh_u, num_sol_name)

    num_u_x = nodal_values[:,0]
    num_u_y = nodal_values[:,1]

    u_predicted_x = torch.tensor(Model_u.nodal_values[0])
    u_predicted_y = torch.tensor(Model_u.nodal_values[1])

    norm_num_ux = torch.norm(num_u_x)
    norm_num_uy = torch.norm(num_u_y)

    L2_diff_ux = torch.norm(u_predicted_x.detach() - num_u_x)
    L2_diff_uy = torch.norm(u_predicted_y.detach() - num_u_y)

    print("ux: |NN - Num|/|Num| = " , (L2_diff_ux/norm_num_ux).item())
    print("uy: |NN - Num|/|Num| = " , (L2_diff_uy/norm_num_uy).item())
    print()

    MSE_ux = torch.mean((u_predicted_x.detach() - num_u_x)**2)
    MSE_uy = torch.mean((u_predicted_y.detach() - num_u_y)**2)

    # print("ux: MSE(NN , Num) = " , MSE_ux.item())
    # print("uy: MSE(NN , Num) = " , MSE_uy.item())
    # print()

    ######################################################################################################
    n_train = 20
    TrailCoord_1d_x = torch.tensor([i for i in torch.linspace(int(L/4),int(3*L/4),n_train)],dtype=torch.float64,  requires_grad=True)
    TrailCoord_1d_y = torch.tensor([i for i in torch.linspace(int(5*L/4),int(3*5*L/4),int(5*n_train/2))],dtype=torch.float64,  requires_grad=True)

    PlotCoordinates = torch.cartesian_prod(TrailCoord_1d_x,TrailCoord_1d_y)
    IDs_u = torch.tensor(mesh_u.GetCellIds(PlotCoordinates),dtype=torch.int)
    IDs_du = torch.tensor(mesh_du.GetCellIds(PlotCoordinates),dtype=torch.int)


    u_predicted = Model_u(PlotCoordinates, IDs_u) 
    du_predicted = Model_du(PlotCoordinates, IDs_du) 

    l_pde, l_compat, s11, s22, s12 =  Mixed_2D_loss(u_predicted[0,:], u_predicted[1,:],
                                                    du_predicted[0,:], du_predicted[1,:], du_predicted[2,:],
                                                    PlotCoordinates, lmbda = 1.25, mu = 1.0)


    num_nodal_values_x = nn.ParameterList([nn.Parameter(torch.tensor([i])) for i in num_u_x])
    num_nodal_values_y = nn.ParameterList([nn.Parameter(torch.tensor([i])) for i in num_u_y])

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

    # print("s11: |NN - Num| = " , L2_diff_s11.item())
    # print("s22: |NN - Num| = " , L2_diff_s22.item())
    # print("s12: |NN - Num| = " , L2_diff_s12.item())

    # print()

    MSE_s11 = torch.mean((num_s11 - s11))
    MSE_s22 = torch.mean((num_s22 - s22))
    MSE_s12 = torch.mean((num_s12 - s12))

    print("s11: mean(|NN - Num|) = " , numpy.abs(MSE_s11.item()))
    print("s22: mean(|NN - Num|) = " , numpy.abs(MSE_s22.item()))
    print("s12: mean(|NN - Num|) = " , numpy.abs(MSE_s12.item()))

    print()

