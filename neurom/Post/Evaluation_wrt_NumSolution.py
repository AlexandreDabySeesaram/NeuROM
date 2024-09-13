#%% Libraries import
import torch
torch.set_default_dtype(torch.float32)
#Import post processing libraries
import Post.Plots as Pplot
import matplotlib.pyplot as plt
import numpy
import torch.nn as nn

from src.PDE_Library import Mixed_2D_loss, Stress


def Read_NumSol(num_sol_name):
    nodal_values = torch.tensor(numpy.load("/Users/skardova/Dropbox/Lungs/HiDeNN_1D/Fenics_solution_2D/"+num_sol_name))
    return nodal_values


def NumSol_eval(mesh_u, mesh_du, Model_u, Model_du, num_sol_name_displ, num_sol_name_stress, L, lmbda, mu):

    print()
    print("*************** Evaluation wrt. Numerical solution ***************\n")

    nodal_values_displ = Read_NumSol(num_sol_name_displ)
    nodal_values_stress = Read_NumSol(num_sol_name_stress)


    mesh_coord_stress = [(Model_du.coordinates[i]).clone().detach().requires_grad_(True) for i in range(len(Model_du.coordinates))]
    mesh_coord_stress = (torch.cat(mesh_coord_stress)).clone().detach().requires_grad_(True)

    cell_ID_stress = torch.tensor(mesh_du.GetCellIds(mesh_coord_stress),dtype=torch.int)

    num_u_x = nodal_values_displ[:,0]
    num_u_y = nodal_values_displ[:,1]

    numerical_s11 = nodal_values_stress[:,0]
    numerical_s12 = nodal_values_stress[:,1]
    numerical_s22 = nodal_values_stress[:,3]

    print("nodal_values_stress : ", nodal_values_stress.shape)

    u_predicted_x = torch.tensor(Model_u.nodal_values[0])
    u_predicted_y = torch.tensor(Model_u.nodal_values[1])

    # s11_predicted = torch.tensor(Model_du.nodal_values[0])
    # s22_predicted = torch.tensor(Model_du.nodal_values[1])
    # s12_predicted = torch.tensor(Model_du.nodal_values[2])

    stress_all_coord = [(Model_du.coordinates[i]).clone().detach().requires_grad_(True) for i in range(len(Model_du.coordinates))]
    stress_all_cell_IDs = torch.tensor([torch.tensor(mesh_du.GetCellIds(i),dtype=torch.int)[0] for i in stress_all_coord])
    stress_all_coord = (torch.cat(stress_all_coord)).clone().detach().requires_grad_(True)

    stress = Model_du(stress_all_coord, stress_all_cell_IDs)

    s11_predicted = stress[0,:]
    s22_predicted = stress[1,:]
    s12_predicted = stress[2,:]

    norm_num_ux = torch.norm(num_u_x)
    norm_num_uy = torch.norm(num_u_y)

    L2_diff_ux = torch.norm(u_predicted_x.detach() - num_u_x)
    L2_diff_uy = torch.norm(u_predicted_y.detach() - num_u_y)

    print("ux: |NN - Num|/|Num| = " , numpy.format_float_scientific((L2_diff_ux/norm_num_ux).item(),precision=4))
    print("uy: |NN - Num|/|Num| = " , numpy.format_float_scientific((L2_diff_uy/norm_num_uy).item(),precision=4))
    print()

    MSE_ux = torch.mean((u_predicted_x.detach() - num_u_x)**2)
    MSE_uy = torch.mean((u_predicted_y.detach() - num_u_y)**2)

    # print("ux: MSE(NN , Num) = " , MSE_ux.item())
    # print("uy: MSE(NN , Num) = " , MSE_uy.item())
    # print()

    ######################################################################################################
    
    u_predicted = Model_u(mesh_coord_stress, cell_ID_stress) 

    u_pred = u_predicted[0,:]
    v_pred = u_predicted[1,:]


    du = torch.autograd.grad(u_pred, mesh_coord_stress, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    dv = torch.autograd.grad(v_pred, mesh_coord_stress, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0]
    
    s11, s22, s12 = Stress(du[:,0], dv[:,1], 0.5*(du[:,1] + dv[:,0]), lmbda, mu)


    MAE_s11 = torch.mean(numpy.abs((numerical_s11 - s11).detach()))
    MAE_s22 = torch.mean(numpy.abs((numerical_s22 - s22).detach()))
    MAE_s12 = torch.mean(numpy.abs((numerical_s12 - s12).detach()))

    print("Num. solution stress vs stress(u_NN)")
    print("s11: mean(|NN - Num|) = " , numpy.format_float_scientific((MAE_s11.item()),precision=4))
    print("s22: mean(|NN - Num|) = " , numpy.format_float_scientific((MAE_s22.item()),precision=4))
    print("s12: mean(|NN - Num|) = " , numpy.format_float_scientific((MAE_s12.item()),precision=4))
    print()
    print("s11: mean(|NN - Num|)/mean(|num|) = " , numpy.format_float_scientific((MAE_s11/torch.mean(numpy.abs(numerical_s11).detach())).item(),precision=4))
    print("s22: mean(|NN - Num|)/mean(|num|) = " , numpy.format_float_scientific((MAE_s22/torch.mean(numpy.abs(numerical_s22).detach())).item(),precision=4))
    print("s12: mean(|NN - Num|)/mean(|num|) = " , numpy.format_float_scientific((MAE_s12/torch.mean(numpy.abs(numerical_s12).detach())).item(),precision=4))
    print()

    ######################################################################################################

    MAE_s11 = torch.mean(numpy.abs((numerical_s11 - s11_predicted).detach()))
    MAE_s22 = torch.mean(numpy.abs((numerical_s22 - s22_predicted).detach()))
    MAE_s12 = torch.mean(numpy.abs((numerical_s12 - s12_predicted).detach()))

    print("Num. solution stress vs NN stress")
    print("s11: mean(|NN - Num|) = " , numpy.format_float_scientific((MAE_s11.item()),precision=4))
    print("s22: mean(|NN - Num|) = " , numpy.format_float_scientific((MAE_s22.item()),precision=4))
    print("s12: mean(|NN - Num|) = " , numpy.format_float_scientific((MAE_s12.item()),precision=4))
    print()

    print("s11: mean(|NN - Num|)/mean(|num|) = " , numpy.format_float_scientific((MAE_s11/torch.mean(numpy.abs(numerical_s11).detach())).item(),precision=4))
    print("s22: mean(|NN - Num|)/mean(|num|) = " , numpy.format_float_scientific((MAE_s22/torch.mean(numpy.abs(numerical_s22).detach())).item(),precision=4))
    print("s12: mean(|NN - Num|)/mean(|num|) = " , numpy.format_float_scientific((MAE_s12/torch.mean(numpy.abs(numerical_s12).detach())).item(),precision=4))
    print()

    ######################################################################################################


def Num_to_NN(Model_u, Model_du, num_sol_name_displ, num_sol_name_stress):

    nodal_values_displ = Read_NumSol(num_sol_name_displ)
    nodal_values_stress = Read_NumSol(num_sol_name_stress)

    num_u_x = nodal_values_displ[:,0]
    num_u_y = nodal_values_displ[:,1]

    numerical_s11 = nodal_values_stress[:,0]
    numerical_s12 = nodal_values_stress[:,1]
    numerical_s22 = nodal_values_stress[:,3]

    
    Model_u.nodal_values[0] = nn.ParameterList([nn.Parameter(torch.tensor([i])) for i in num_u_x])
    Model_u.nodal_values[1] = nn.ParameterList([nn.Parameter(torch.tensor([i])) for i in num_u_y])

    Model_du.nodal_values[0] = nn.ParameterList([nn.Parameter(torch.tensor([i])) for i in numerical_s11])
    Model_du.nodal_values[1] = nn.ParameterList([nn.Parameter(torch.tensor([i])) for i in numerical_s22])
    Model_du.nodal_values[2] = nn.ParameterList([nn.Parameter(torch.tensor([i])) for i in numerical_s12])

    return Model_u, Model_du
