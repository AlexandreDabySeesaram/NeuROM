
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
import numpy as np
import matplotlib
import torch
import meshio
from scipy import ndimage

matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = "14"
from matplotlib.ticker import MaxNLocator

from Bin.PDE_Library import Stress


def PlotSolution_Coordinates_Analytical(A,E,InitialCoordinates,Coordinates,TrialCoordinates,AnalyticSolution,model,name):

    #plt.plot(InitialCoordinates,[coord*0 for coord in InitialCoordinates],'+k', markersize=2, label = 'Initial Nodes')
    
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(9, 7)

    param = model.coordinates[3]
    if param.requires_grad == True:
        plt.scatter(InitialCoordinates,[coord*0 for coord in InitialCoordinates], s=6, color="pink", alpha=0.5, label = 'Initial Nodes')
        
    plt.plot(Coordinates,[coord*0 for coord in Coordinates],'.k', markersize=2, label = 'Mesh Nodes')
    plt.plot(TrialCoordinates.data,AnalyticSolution(A,E,TrialCoordinates.data), label = 'Ground Truth')
    plt.plot(TrialCoordinates.data,model(TrialCoordinates).data,'--', label = 'HiDeNN')
    plt.xlabel(r'$\underline{x}$ [m]')
    plt.ylabel(r'$\underline{u}\left(\underline{x}\right)$')
    plt.legend(loc="upper left")
    # plt.title('Displacement')
    plt.savefig('Results/'+name+'.pdf', transparent=True)  
    #plt.show()
    plt.clf()

def PlotGradSolution_Coordinates_Analytical(A,E,InitialCoordinates,Coordinates,TrialCoordinates,AnalyticGradientSolution,model,Derivative,name):
    # Plots the gradient & compare to reference
    #plt.plot(InitialCoordinates,[coord*0 for coord in InitialCoordinates],'+k', markersize=2, label = 'Initial Nodes')'

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(9, 7)

    param = model.coordinates[3]
    if param.requires_grad == True:
        plt.scatter(InitialCoordinates,[coord*0 for coord in InitialCoordinates], s=6, color="pink", alpha=0.5, label = 'Initial Nodes')
    plt.plot(Coordinates,[coord*0 for coord in Coordinates],'.k', markersize=2, label = 'Mesh Nodes')
    plt.plot(TrialCoordinates.data,AnalyticGradientSolution(A,E,TrialCoordinates.data), label = 'Ground Truth')
    plt.plot(TrialCoordinates.data,Derivative(model(TrialCoordinates),TrialCoordinates).data,'--', label = 'HiDeNN')
    plt.xlabel(r'$\underline{x}$ [m]')
    plt.ylabel(r'$\frac{d\underline{u}}{dx}\left(\underline{x}\right)$')
    plt.legend(loc="upper left")
    # plt.title('Displacement first derivative')
    plt.savefig('Results/'+name+'.pdf', transparent=True)  
    #plt.show()
    plt.clf()

def PlotEnergyLoss(error,zoom,name):
    """Plots the error from the index given by zoom """
    plt.plot(error[zoom:])
    plt.xlabel(r'epochs')
    plt.ylabel(r'$J\left(\underline{u}\left(\underline{x}\right)\right)$')
    plt.savefig('Results/'+name+'.pdf', transparent=True)  
    #plt.show()
    plt.clf()

def PlotTrajectories(Coord_trajectories,name):
    """Plots the trajectories of the coordinates during training"""
    plt.plot(Coord_trajectories)
    plt.xlabel(r'epochs')
    plt.ylabel(r'$x_i\left(\underline{x}\right)$')
    plt.savefig('Results/'+name+'.pdf', transparent=True)  
    #plt.show()
    plt.clf()

def Plot_Loss_Modes(Modes_flag,error,name):
    # Lift to be able to use semilogy
    error3 = error-np.min(error)
    plt.plot(Modes_flag,color='#247ab5ff')
    ax = plt.gca()
    ax.tick_params(axis='y', colors='#247ab5ff')
    # plt.ylabel(r'$\Vert \underline{u}_{ex} - \underline{u}_{NN} \Vert^2$')
    plt.ylabel(r'$m$',color='#247ab5ff')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel(r'Epochs')
    # plt.ylim((0.01,20))
    ax2 = plt.gca().twinx()
    ax2.semilogy(error3,color='#d95319ff')
    ax2.tick_params(axis='y', colors='#d95319ff')
    ax2.set_ylabel(r'Lifted $J\left(\underline{u}\left(\underline{x}\right)\right)$',color='#d95319ff')
    plt.savefig('Results/'+name+'.pdf', transparent=True, bbox_inches = "tight")
    plt.clf()

def Plot_Lossdecay_Modes(Modes_flag,decay,name,threshold):
    # Lift to be able to use semilogy
    plt.plot(Modes_flag,color='#247ab5ff')
    ax = plt.gca()
    ax.tick_params(axis='y', colors='#247ab5ff')
    # plt.ylabel(r'$\Vert \underline{u}_{ex} - \underline{u}_{NN} \Vert^2$')
    plt.ylabel(r'$m$',color='#247ab5ff')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.xlabel(r'Epochs')

    # plt.ylim((0.01,20))
    ax2 = plt.gca().twinx()
    ax2.semilogy(decay,color='#d95319ff')
    ax2.tick_params(axis='y', colors='#d95319ff')
    ax2.set_ylabel(r'd log($J\left(\underline{u}\left(\underline{x}\right)\right)$)',color='#d95319ff')
    plt.axhline(threshold,color = 'k')
    plt.savefig('Results/'+name+'.pdf', transparent=True, bbox_inches = "tight")
    plt.clf()

def Plot_Compare_Loss2l2norm(error,error2,name):
    # Lift to be able to use semilogy
    # error3 = error-np.min(error)

    error3 = np.abs(error)

    plt.semilogy(error2,color='#247ab5ff')
    ax = plt.gca()
    ax.tick_params(axis='y', colors='#247ab5ff')
    # plt.ylabel(r'$\Vert \underline{u}_{ex} - \underline{u}_{NN} \Vert^2$')
    plt.ylabel(r'$\Xi$',color='#247ab5ff')
    # plt.ylim((0.01,20))
    ax2 = plt.gca().twinx()
    ax2.semilogy(error3,color='#d95319ff')
    ax2.tick_params(axis='y', colors='#d95319ff')
    ax2.set_ylabel(r'$J\left(\underline{u}\left(\underline{x}\right)\right)$',color='#d95319ff')
    plt.savefig('Results/'+name+'.pdf', transparent=True, bbox_inches = "tight")
    plt.clf()


def Plot_Compare_Loss2l2norm_Mixed(error_pde, error_constit, error2,name):
    # Lift to be able to use semilogy
    #error_pde = error_pde-np.min(error_pde)
    #error_constit = error_constit-np.min(error_constit)

    plt.semilogy(error2)
    plt.ylabel(r'$\Vert \underline{u}_{ex} - \underline{u}_{NN} \Vert^2$')
    #plt.ylabel(r'$\Xi$')
    #plt.ylim((0.01,20))
    ax2 = plt.gca().twinx()
    ax2.semilogy(error_pde,color='#F39C12', label = "PDE")
    ax2.semilogy(error_constit,color='#741B47', label = "Constitutive")

    ax2.set_ylabel(r'$J\left(\underline{u}\left(\underline{x}\right)\right)$')
    plt.savefig('Results/'+name+'.pdf', transparent=True) 
    plt.legend()
    plt.clf()


def Plot_end(error,error2):
    # Lift to be able to use semilogy
    #error3 = error-np.min(error)
    error3 = [-x for x in error]
    #plt.semilogy(error2[-1000:-1])
    #plt.ylabel(r'$\Vert \underline{u}_{ex} - \underline{u}_{NN} \Vert^2$')
    ax2 = plt.gca().twinx()
    ax2.semilogy(error3[1000:],color='#d95319ff', label="Min = "+str(np.format_float_scientific(np.min(error), precision=3))+", final = " +str(np.format_float_scientific(error[-1], precision=3)))
    ax2.set_ylabel(r'$J\left(\underline{u}\left(\underline{x}\right)\right)$')
    #plt.legend(bbox_to_anchor=[0.7, 0.5], loc='center')
    plt.legend()
    plt.savefig('Results/Loss_end.pdf', transparent=True)  
    plt.clf()


def Plot_LearningRate(Learning_Rate):
    plt.ylabel(r'Learning rate')
    plt.xlabel(r'Epochs')
    plt.semilogy(Learning_Rate, color='#F50C12', label="Final learning rate = "+ str(np.format_float_scientific(Learning_Rate[-1], precision=2)))
    plt.legend()
    plt.savefig('Results/Learning rate.pdf')  
    plt.clf()


def Plot_ShapeFuctions(TrialCoordinates, model, InitCoord, ProjectWeight):
    shape_function = model(TrialCoordinates)[1]

    if ProjectWeight == False:
        for i in range(shape_function.shape[1]):
            plt.plot(TrialCoordinates.data, shape_function[:,i].detach(), label="Shape function "+str(i))
    else:
        nodal_values_inter = model.InterpoLayer_uu.weight.data.detach()
        nodal_values = np.zeros(shape_function.shape[1])
        nodal_values[1:-1] = nodal_values_inter

        for i in range(shape_function.shape[1]):
            plt.plot(TrialCoordinates.data, nodal_values[i]*shape_function[:,i].detach(), label="Shape function "+str(i))
    plt.scatter(InitCoord,[coord*0 for coord in InitCoord], s=6, color="pink", alpha=0.5)

    #plt.legend()
    plt.savefig('Results/ShapeFunctions.pdf')
    plt.close()

def Plot_Parametric_Young(BeamROM,TrialCoordinates,A,AnalyticSolution,name_model = 'tmp'):
    import torch
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.size"] = "13"
    u0 = BeamROM.Space_modes[0].u_0
    uL = BeamROM.Space_modes[0].u_L

    PaperPara = torch.tensor([150])
    PaperPara = PaperPara[:,None] # Add axis so that dimensions match
    u_150 = BeamROM(TrialCoordinates,[PaperPara])
    u_analytical_150 = AnalyticSolution(A,PaperPara.item(),TrialCoordinates.data,u0,uL)
    plt.plot(TrialCoordinates.data,u_analytical_150, color="#01426A", label = r'$E = 150~$MPa Analytical solution')
    plt.plot(TrialCoordinates.data,u_150.data,'--', color="#01426A", label = r'$E = 150~$MPa HiDeNN solution')

    PaperPara = torch.tensor([200])
    PaperPara = PaperPara[:,None] # Add axis so that dimensions match
    u_200 = BeamROM(TrialCoordinates,[PaperPara])
    u_analytical_200 = AnalyticSolution(A,PaperPara.item(),TrialCoordinates.data,u0,uL)
    plt.plot(TrialCoordinates.data,u_analytical_200, color="#00677F", label = r'$E = 200~$MPa Analytical solution')
    plt.plot(TrialCoordinates.data,u_200.data,'--',color="#00677F", label = r'$E = 200~$MPa HiDeNN solution')

    PaperPara = torch.tensor([100])
    PaperPara = PaperPara[:,None] # Add axis so that dimensions match
    u_100 = BeamROM(TrialCoordinates,[PaperPara])
    u_analytical_100 = AnalyticSolution(A,PaperPara.item(),TrialCoordinates.data,u0,uL)
    plt.plot(TrialCoordinates.data,u_analytical_100,color="#A92021", label = r'$E = 100~$MPa Analytical solution')
    plt.plot(TrialCoordinates.data,u_100.data,'--',color="#A92021", label = r'$E = 100~$MPa HiDeNN solution')
    plt.legend(loc="upper left")
    plt.xlabel('x (mm)')
    plt.ylabel('u (mm)')
    plt.savefig('Results/Para_displacements'+name_model+'.pdf', transparent=True)  
    plt.show()
    plt.clf()

def Plot_BiParametric_Young(BeamROM,TrialCoordinates,A,AnalyticSolution,name_model = 'tmp'):
    import torch
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.size"] = "13"
    u0 = BeamROM.Space_modes[0].u_0
    uL = BeamROM.Space_modes[0].u_L

    PaperPara = torch.tensor([150])
    PaperPara = PaperPara[:,None] # Add axis so that dimensions match
    Paper150 = torch.tensor([150])
    Paper150 = Paper150[:,None] # Add axis so that dimensions match
    u_150 = BeamROM(TrialCoordinates,[PaperPara,PaperPara])
    u_analytical_150 = AnalyticSolution(A,PaperPara.item(),TrialCoordinates.data,u0,uL)
    plt.plot(TrialCoordinates.data,u_analytical_150, color="#01426A", label = r'$E = 150~$MPa Analytical solution')
    plt.plot(TrialCoordinates.data,u_150.data.view(-1),'--', color="#01426A", label = r'$E = 150~$MPa HiDeNN solution')

    PaperPara = torch.tensor([200])
    PaperPara = PaperPara[:,None] # Add axis so that dimensions match
    u_200 = BeamROM(TrialCoordinates,[Paper150,PaperPara])
    u_analytical_200 = AnalyticSolution(A,PaperPara.item(),TrialCoordinates.data,u0,uL)
    plt.plot(TrialCoordinates.data,u_analytical_200, color="#00677F", label = r'$E = 200~$MPa Analytical solution')
    plt.plot(TrialCoordinates.data,u_200.data.view(-1),'--',color="#00677F", label = r'$E = 200~$MPa HiDeNN solution')

    PaperPara = torch.tensor([100])
    PaperPara = PaperPara[:,None] # Add axis so that dimensions match
    u_100 = BeamROM(TrialCoordinates,[Paper150,PaperPara])
    u_analytical_100 = AnalyticSolution(A,PaperPara.item(),TrialCoordinates.data,u0,uL)
    plt.plot(TrialCoordinates.data,u_analytical_100,color="#A92021", label = r'$E = 100~$MPa Analytical solution')
    plt.plot(TrialCoordinates.data,u_100.data.view(-1),'--',color="#A92021", label = r'$E = 100~$MPa HiDeNN solution')
    plt.legend(loc="upper left")
    plt.xlabel('x (mm)')
    plt.ylabel('u (mm)')
    plt.savefig('Results/Para_displacements'+name_model+'.pdf', transparent=True)  
    plt.show()
    plt.clf()

def Plot_Parametric_Young_Interactive(BeamROM,TrialCoordinates,A,AnalyticSolution,name_model):
    from ipywidgets import interact, widgets
    import torch
    def interactive_plot(E):
        u0 = BeamROM.Space_modes[0].u_0
        uL = BeamROM.Space_modes[0].u_L
        # Calculate the corresponding function values for each x value
        u_analytical_E = AnalyticSolution(A,E,TrialCoordinates.data,u0,uL)
        Nodal_coordinates = [BeamROM.Space_modes[0].coordinates[l].data for l in range(len(BeamROM.Space_modes[0].coordinates))]
        Nodal_coordinates = torch.cat(Nodal_coordinates)
        u_analytical_E_discrete = AnalyticSolution(A,E,Nodal_coordinates.data,u0,uL)
        E = torch.tensor([E])
        E = E[:,None] # Add axis so that dimensions match
        u_E = BeamROM(TrialCoordinates,[E])
        u_NN_discrete = BeamROM(Nodal_coordinates,[E])
        error_tensor = u_analytical_E - u_E
        error_tensor_discrete = u_analytical_E_discrete - u_NN_discrete
        # Reative error in percentage
        error_norm = 100*torch.sqrt(torch.sum(error_tensor*error_tensor))/torch.sqrt(torch.sum(u_analytical_E*u_analytical_E))
        error_scientific_notation = f"{error_norm:.2e}"

        error_norm_discrete = 100*torch.sqrt(torch.sum(error_tensor_discrete*error_tensor_discrete))/torch.sqrt(torch.sum(u_analytical_E_discrete*u_analytical_E_discrete))
        error_scientific_notation_discrete = f"{error_norm_discrete:.2e}"
        # error_str = str(error_norm.item())
        title_error =  r'$\frac{\Vert u_{exact} - u_{ROM}\Vert}{\Vert u_{exact}\Vert}$ = '+error_scientific_notation+ '$\%$'+' Discrete: '+error_scientific_notation_discrete+ '$\%$'
        # Plot the function
        plt.plot(TrialCoordinates.data,u_analytical_E,color="#A92021", label = 'Ground truth')
        plt.plot(TrialCoordinates.data, u_E.data, label = 'Discrete solution')
        plt.title(title_error)
        plt.xlabel('x (mm)')
        plt.ylabel('u(x,E) (mm)')
        plt.legend(loc="upper left")
        plt.grid(True)
        plt.ylim((0,0.02))
        plt.show()

    # Create an interactive slider
    slider = widgets.FloatSlider(value=0, min=100, max=200, step=0.01, description='E (GPa)')

    # Connect the slider to the interactive plot function
    interactive_plot_widget = interact(interactive_plot, E=slider)

def Plot_BiParametric_Young_Interactive(BeamROM,TrialCoordinates,A,AnalyticBiParametricSolution,name_model):
    from ipywidgets import interact, widgets
    import torch
    def interactive_plot(E1,E2):
        u0 = BeamROM.Space_modes[0].u_0
        uL = BeamROM.Space_modes[0].u_L
        # Calculate the corresponding function values for each x value
        u_analytical_E = AnalyticBiParametricSolution(A,[E2,E1],TrialCoordinates.data,u0,uL).view(-1)
        E1 = torch.tensor([E1])
        E1 = E1[:,None] # Add axis so that dimensions match
        E2 = torch.tensor([E2])
        E2 = E2[:,None] # Add axis so that dimensions match        
        E = [E1,E2]
        u_E = BeamROM(TrialCoordinates,E).view(-1)
        error_tensor = u_analytical_E - u_E
        # Reative error in percentage
        error_norm = 100*torch.sqrt(torch.sum(error_tensor*error_tensor))/torch.sqrt(torch.sum(u_analytical_E*u_analytical_E))
        error_scientific_notation = f"{error_norm:.2e}"
        # error_str = str(error_norm.item())
        title_error =  r'$\frac{\Vert u_{exact} - u_{ROM}\Vert}{\Vert u_{exact}\Vert}$ = '+error_scientific_notation+ '$\%$'
        # Plot the function
        plt.plot(TrialCoordinates.data,u_analytical_E,color="#A92021", label = 'Ground truth')
        plt.plot(TrialCoordinates.data, u_E.data, label = 'Discrete solution')
        plt.title(title_error)
        plt.xlabel('x (mm)')
        plt.ylabel('u(x,E) (mm)')
        plt.legend(loc="upper left")
        plt.grid(True)
        plt.ylim((0,0.02))
        plt.show()

    # Create an interactive slider
    slider_E1 = widgets.FloatSlider(value=0, min=100, max=200, step=0.01, description='E1 (GPa)')
    slider_E2 = widgets.FloatSlider(value=0, min=100, max=200, step=0.01, description='E2 (GPa)')
    

    # Connect the slider to the interactive plot function
    interactive_plot_widget = interact(interactive_plot, E1=slider_E1, E2=slider_E2)

def PlotModes(BeamROM,TrialCoordinates,TrialPara,A,AnalyticSolution,name_model):
    import torch
    Space_modes = [BeamROM.Space_modes[l](TrialCoordinates) for l in range(BeamROM.n_modes_truncated)]
    u_i = torch.cat(Space_modes,dim=1)  
    for mode in range(BeamROM.n_modes_truncated):
        Para_mode_List = [BeamROM.Para_modes[mode][l](TrialPara)[:,None] for l in range(BeamROM.n_para)]
        if mode == 0:
            lambda_i = torch.unsqueeze(torch.cat(Para_mode_List,dim=1), dim=0)
            # Para_modes = torch.unsqueeze(Para_modes, dim=0)
        else:
            New_mode = torch.unsqueeze(torch.cat(Para_mode_List,dim=1), dim=0)
            lambda_i = torch.vstack((lambda_i,New_mode))

    for mode in range(BeamROM.n_modes_truncated):
        plt.plot(TrialCoordinates.data,u_i[:,mode].data,label='Mode'+str(mode+1))
        plt.xlabel('x (mm)')
        plt.legend(loc="upper left")
    plt.savefig('Results/Pre_trained_Space_modes'+str(BeamROM.n_modes_truncated)+'.pdf', transparent=True)  
    plt.clf()
        # plt.show()

    for mode in range(BeamROM.n_modes_truncated):
        plt.plot(TrialPara.data,lambda_i[mode,:,0].data,label='Mode'+str(mode+1))
        plt.xlabel('E (GPa)')
        plt.legend(loc="upper left")
    plt.savefig('Results/Pre_trained_Para_modes'+str(BeamROM.n_modes_truncated)+'.pdf', transparent=True)  
    plt.clf()
        # plt.show()

def PlotModesBi(BeamROM,TrialCoordinates,TrialPara,A,AnalyticSolution,name_model):
    import torch
    Space_modes = [BeamROM.Space_modes[l](TrialCoordinates) for l in range(BeamROM.n_modes_truncated)]
    u_i = torch.cat(Space_modes,dim=1)  
    for mode in range(BeamROM.n_modes_truncated):
        Para_mode_List = [BeamROM.Para_modes[mode][l](TrialPara[l][:,0].view(-1,1))[:,None] for l in range(BeamROM.n_para)]
        if mode == 0:
            lambda_i = [torch.unsqueeze(Para_mode_List[l],dim=0) for l in range(BeamROM.n_para)]
        else:
            New_mode = Para_mode_List
            lambda_i = [torch.vstack((lambda_i[l],torch.unsqueeze(New_mode[l],dim=0))) for l in range(BeamROM.n_para)]

    for mode in range(BeamROM.n_modes_truncated):
        plt.plot(TrialCoordinates.data,u_i[:,mode].data,label='Mode'+str(mode+1))
        plt.xlabel('x (mm)')
        plt.legend(loc="upper left")
    plt.savefig('Results/Pre_trained_Space_modes'+str(BeamROM.n_modes_truncated)+'.pdf', transparent=True)  
    plt.clf()
        # plt.show()

    for mode in range(BeamROM.n_modes_truncated):
        plt.plot(TrialPara[0].data,lambda_i[0][mode,:,0].data,label='Mode'+str(mode+1))
        plt.xlabel('E1 (GPa)')
        plt.legend(loc="upper left")
    plt.savefig('Results/Pre_trained_Para_modes_E1'+str(BeamROM.n_modes_truncated)+'.pdf', transparent=True)  
    plt.clf()

    for mode in range(BeamROM.n_modes_truncated):
        plt.plot(TrialPara[1].data,lambda_i[1][mode,:,0].data,label='Mode'+str(mode+1))
        plt.xlabel('E2 (GPa)')
        plt.legend(loc="upper left")
        plt.savefig('Results/Pre_trained_Para_modes_E2'+str(BeamROM.n_modes_truncated)+'.pdf', transparent=True)  
    plt.clf()
        # plt.show()

def PlotParaModes(BeamROM,TrialPara,name_model):
    import torch

    for mode in range(BeamROM.n_modes_truncated):
        Para_mode_List = [BeamROM.Para_modes[mode][l](TrialPara[l][:,0].view(-1,1))[:,None] for l in range(BeamROM.n_para)]
        if mode == 0:
            lambda_i = [torch.unsqueeze(Para_mode_List[l],dim=0) for l in range(BeamROM.n_para)]
        else:
            New_mode = Para_mode_List
            lambda_i = [torch.vstack((lambda_i[l],torch.unsqueeze(New_mode[l],dim=0))) for l in range(BeamROM.n_para)]


    for mode in range(BeamROM.n_modes_truncated):
        plt.plot(TrialPara[0].data,lambda_i[0][mode,:,0].data,label='Mode'+str(mode+1))
        plt.xlabel('E (GPa)')
        plt.legend(loc="upper left")
    plt.savefig('Results/Pre_trained_Para_modes_E1'+str(BeamROM.n_modes_truncated)+'.pdf', transparent=True)  
    plt.clf()


        # plt.show()

def AppInteractive(BeamROM, TrialCoordinates, A, AnalyticSolution):
    import matplotlib as mpl
    my_dpi = 50
    # mpl.rcParams['figure.dpi'] = 50
    import tkinter as tk
    from tkinter import ttk  # Import the themed Tkinter module
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import torch
    import numpy as np
    def interactive_plot(E):
        u0 = BeamROM.Space_modes[0].u_0
        uL = BeamROM.Space_modes[0].u_L
        u_analytical_E = AnalyticSolution(A, E, TrialCoordinates.data, u0, uL)
        E = torch.tensor([E])
        E = E[:, None]
        u_E = BeamROM(TrialCoordinates, [E])
        error_tensor = u_analytical_E - u_E
        error_norm = 100 * torch.sqrt(torch.sum(error_tensor * error_tensor)) / torch.sqrt(
            torch.sum(u_analytical_E * u_analytical_E)
        )
        error_scientific_notation = f"{error_norm:.2e}"
        title_error = r'$\frac{\Vert u_{exact} - u_{ROM}\Vert}{\Vert u_{exact}\Vert}$ = ' + error_scientific_notation + '$\%$'

        ax.clear()
        ax.plot(TrialCoordinates.data, u_analytical_E, color="#A92021", label='Ground truth')
        ax.plot(TrialCoordinates.data, u_E.data, label='Discrete solution')
        ax.set_title(title_error)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('u(x,E) (mm)')
        ax.legend(loc="upper left")
        ax.grid(True)
        ax.set_ylim((0, 0.02))
        canvas.draw()

    def on_slider_change(val):
        E_value = float(val)
        interactive_plot(E_value)
        label.config(text=f"E = {float(val):.2f}GPa")
        canvas.draw()
        

    root = tk.Tk()
    root.title("NeuROM - Interactive plot")
    root.minsize(400, 400)

    fig, ax = plt.subplots(figsize=(20/my_dpi, 20/my_dpi), dpi=my_dpi)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Use ttk.Scale 
    slider = ttk.Scale(root, from_=100, to=200, orient=tk.HORIZONTAL, length=300, command=on_slider_change)
    label = tk.Label(root, text="E = 100GPa")
    label.pack()
    slider.pack()
    # Set the initial value of the slider
    slider.set(100)
    # Call the initial plot
    interactive_plot(100)
    tk.mainloop()


def PlotGradSolution_Coordinates_Force(A,E,InitialCoordinates,Coordinates,TrialCoordinates,Force,model,Derivative,name):
    pred = model(TrialCoordinates)
    
    # Plots the gradient & compare to reference
    plt.scatter(InitialCoordinates,[coord*0 for coord in InitialCoordinates], s=6, color="pink", alpha=0.5)
    plt.plot(Coordinates,[coord*0 for coord in Coordinates],'.k', markersize=2, label = 'Mesh Nodes')
    plt.plot(TrialCoordinates.data, Force.data/(-A*E), label = 'Ground Truth')
    plt.plot(TrialCoordinates.data, Derivative(pred,TrialCoordinates).data,'--', label = 'HiDeNN')
    plt.xlabel(r'$\underline{x}$ [m]')
    plt.ylabel(r'$\frac{d\underline{u}}{dx}\left(\underline{x}\right)$')
    plt.legend(loc="upper left")
    plt.savefig('Results/'+name+'.pdf', transparent=True) 

    plt.clf()


def Plot_Rectangle_2Dresults(u_predicted, n_train_x, n_train_y, name):


    fig, ax = plt.subplots(1, 2, layout="constrained", figsize = (10,10), dpi=50)

    img = np.reshape(u_predicted[0,:].detach(), (n_train_x, n_train_y), order='C') 
    img0 = ax[0].imshow(ndimage.rotate(img, 90)) #, vmin = 0, vmax = 1)

    img = np.reshape(u_predicted[1,:].detach(), (n_train_x, n_train_y), order='C') 
    img1 = ax[1].imshow(ndimage.rotate(img, 90)) # , vmin = 0, vmax = 1)

    cb1 = fig.colorbar(img0, ax = ax[0], location="right", pad=0.2, shrink=0.8)
    cb2 = fig.colorbar(img1, ax = ax[1], location="right", pad=0.2, shrink=0.8)

    tick_font_size = 28
    cb1.ax.tick_params(labelsize=tick_font_size)
    cb2.ax.tick_params(labelsize=tick_font_size)

    plt.set_cmap("coolwarm") 

    fig.savefig('Results/2D_'+name+'.pdf', transparent=True) 

    plt.close()


def Plot_Rectangle_2Dresults_Derivative(u_predicted, e11, e22, e12, n_train_x, n_train_y, name):

    fig, ax = plt.subplots(2, 3, layout="constrained", figsize = (14,20), dpi=50)


    img = np.reshape(u_predicted[0,:].detach(), (n_train_x, n_train_y), order='C') 
    img0 = ax[0,0].imshow(ndimage.rotate(img, 90)) #, vmin = 0, vmax = 1)

    img = np.reshape(u_predicted[1,:].detach(), (n_train_x, n_train_y), order='C') 
    img1 = ax[0,1].imshow(ndimage.rotate(img, 90)) # , vmin = 0, vmax = 1)

    img = np.reshape(u_predicted[2,:].detach(), (n_train_x, n_train_y), order='C') 
    img2 = ax[0,2].imshow(ndimage.rotate(img, 90)) # , vmin = 0, vmax = 1)

    cb1 = fig.colorbar(img0, ax = ax[0,0], location="right", pad=0.2, shrink=0.8)
    cb2 = fig.colorbar(img1, ax = ax[0,1], location="right", pad=0.2, shrink=0.8)
    cb3 = fig.colorbar(img2, ax = ax[0,2], location="right", pad=0.2, shrink=0.8)


    img = np.reshape(e11.detach(), (n_train_x, n_train_y), order='C') 
    img3 = ax[1,0].imshow(ndimage.rotate(img, 90)) #, vmin = 0, vmax = 1)

    img = np.reshape(e22.detach(), (n_train_x, n_train_y), order='C') 
    img4 = ax[1,1].imshow(ndimage.rotate(img, 90)) # , vmin = 0, vmax = 1)

    img = np.reshape(e12.detach(), (n_train_x, n_train_y), order='C') 
    img5 = ax[1,2].imshow(ndimage.rotate(img, 90)) # , vmin = 0, vmax = 1)

    cb4 = fig.colorbar(img3, ax = ax[1,0], location="right", pad=0.2, shrink=0.8)
    cb5 = fig.colorbar(img4, ax = ax[1,1], location="right", pad=0.2, shrink=0.8)
    cb6 = fig.colorbar(img5, ax = ax[1,2], location="right", pad=0.2, shrink=0.8)

    tick_font_size = 28
    cb1.ax.tick_params(labelsize=tick_font_size)
    cb2.ax.tick_params(labelsize=tick_font_size)
    cb3.ax.tick_params(labelsize=tick_font_size)
    cb4.ax.tick_params(labelsize=tick_font_size)
    cb5.ax.tick_params(labelsize=tick_font_size)
    cb6.ax.tick_params(labelsize=tick_font_size)

    plt.set_cmap("coolwarm") 

    fig.savefig('Results/2D_'+name+'.pdf', transparent=True) 
    plt.close()


def Plot2DLoss(loss):

    plt.plot(loss[0],color='#F39C12', label = "$\mathrm{Loss_{PDE}}$")
    plt.plot(loss[1],color='#741B47', label = "$\mathrm{Loss_{Constitutive}}$")

    plt.yscale('log')
    plt.legend()
    plt.savefig('Results/2D_loss.pdf', transparent=True) 
    plt.clf()

def Plot1DSection(u_predicted, n_train_x, n_train_y, name):

    u = np.reshape(u_predicted[0,:].detach(), (n_train_x, n_train_y), order='C') 
    u = ndimage.rotate(u, 90)

    v = np.reshape(u_predicted[1,:].detach(), (n_train_x, n_train_y), order='C') 
    v = ndimage.rotate(v, 90)

    rows = u.shape[0]
    cols = u.shape[1]

    plt.plot(np.linspace(0,50,rows), v[::-1,int(cols/4)], label = "Section: x = 1/4Lx")
    plt.plot(np.linspace(0,50,rows), v[::-1,int(cols/2)], label = "Section: x = 1/2Lx")
    plt.plot(np.linspace(0,50,rows), v[::-1,int(3*cols/4)], label = "Section: x = 3/4Lx")

    plt.legend()
    plt.savefig('Results/1D_Section_x_'+name+'.pdf', transparent=True) 
    plt.clf()

    plt.plot(np.linspace(0,10,cols), u[int(rows/4),:], label = "Section: y = 3/4Ly")
    plt.plot(np.linspace(0,10,cols), u[int(rows/2),:], label = "Section: y = 1/2Ly")
    plt.plot(np.linspace(0,10,cols), u[int(3*rows/4),:], label = "Section: y = 1/4Ly")

    plt.legend()
    plt.savefig('Results/1D_Section_y_'+name+'.pdf', transparent=True) 
    plt.clf()



def Plot2Dresults(u_predicted, x, name):

    u_predicted = u_predicted.reshape(2,-1)
    x = x.reshape(-1,2)

    fig, ax = plt.subplots(1, 2, layout="constrained", figsize = (18,8), dpi=50)

    size =  0.5*8*50/ (np.sqrt(x.shape[0])/3)

    # img0 = ax[0].scatter( x[:,0].detach(),  x[:,1].detach(), c = u_predicted[0,:].detach(), vmin=-0.55, vmax=0.55, marker="s", s=size, alpha =1.0, cmap = "coolwarm")
    # img1 = ax[1].scatter( x[:,0].detach(),  x[:,1].detach(), c = u_predicted[1,:].detach(), vmin=0, vmax=1.7, marker="s", s=size, alpha =1.0, cmap = "coolwarm")
    
    img0 = ax[0].scatter( x[:,0].detach(),  x[:,1].detach(), c = u_predicted[0,:].detach(), marker="s", s=size, alpha =1.0, cmap = "coolwarm")
    img1 = ax[1].scatter( x[:,0].detach(),  x[:,1].detach(), c = u_predicted[1,:].detach(), marker="s", s=size, alpha =1.0, cmap = "coolwarm")
    
    ax[0].set_aspect('equal', adjustable='box')
    ax[1].set_aspect('equal', adjustable='box')

    cb1 = fig.colorbar(img0, ax = ax[0], location="right", pad=0.2, shrink=0.8)
    cb2 = fig.colorbar(img1, ax = ax[1], location="right", pad=0.2, shrink=0.8)

    tick_font_size = 28
    cb1.ax.tick_params(labelsize=tick_font_size)
    cb2.ax.tick_params(labelsize=tick_font_size)

    fig.savefig('Results/2D_'+name+'.pdf', transparent=True) 

    plt.close()

def Plot2Dresults_Derivative(u_predicted, e11, e22, e12, x, name):

    fig, ax = plt.subplots(2, 3, layout="constrained", figsize = (18, 10), dpi=50)

    size = 0.2*10*50/ (np.sqrt(x.shape[0])/3)

    # img0 = ax[0,0].scatter( x[:,0].detach(),  x[:,1].detach(), c = u_predicted[0,:].detach(), vmin=-0.1, vmax=0.1, marker="s", s=size, alpha =1.0, cmap = "coolwarm")
    # img1 = ax[0,1].scatter( x[:,0].detach(),  x[:,1].detach(), c = u_predicted[1,:].detach(), vmin=-0.25, vmax=0.25, marker="s", s=size, alpha =1.0, cmap = "coolwarm")
    # img2 = ax[0,2].scatter( x[:,0].detach(),  x[:,1].detach(), c = u_predicted[2,:].detach(), vmin=-0.055, vmax=0.055, marker="s", s=size, alpha =1.0, cmap = "coolwarm")

    img0 = ax[0,0].scatter( x[:,0].detach(),  x[:,1].detach(), c = u_predicted[0,:].detach(), marker="s", s=size, alpha =1.0, cmap = "coolwarm")
    img1 = ax[0,1].scatter( x[:,0].detach(),  x[:,1].detach(), c = u_predicted[1,:].detach(),  marker="s", s=size, alpha =1.0, cmap = "coolwarm")
    img2 = ax[0,2].scatter( x[:,0].detach(),  x[:,1].detach(), c = u_predicted[2,:].detach(),   marker="s", s=size, alpha =1.0, cmap = "coolwarm")
    
    cb1 = fig.colorbar(img0, ax = ax[0,0], location="right", pad=0.2, shrink=0.8)
    cb2 = fig.colorbar(img1, ax = ax[0,1], location="right", pad=0.2, shrink=0.8)
    cb3 = fig.colorbar(img2, ax = ax[0,2], location="right", pad=0.2, shrink=0.8)
    ax[0,0].set_aspect('equal', adjustable='box')
    ax[0,1].set_aspect('equal', adjustable='box')
    ax[0,2].set_aspect('equal', adjustable='box')

    # img3 = ax[1,0].scatter( x[:,0].detach(),  x[:,1].detach(), c = e11.detach(), vmin=-0.1, vmax=0.1, marker="s", s=size, alpha =1.0, cmap = "coolwarm")
    # img4 = ax[1,1].scatter( x[:,0].detach(),  x[:,1].detach(), c = e22.detach(), vmin=-0.25, vmax=0.25, marker="s", s=size, alpha =1.0, cmap = "coolwarm")
    # img5 = ax[1,2].scatter( x[:,0].detach(),  x[:,1].detach(), c = e12.detach(), vmin=-0.055, vmax=0.055, marker="s", s=size, alpha =1.0, cmap = "coolwarm")

    img3 = ax[1,0].scatter( x[:,0].detach(),  x[:,1].detach(), c = e11.detach(), marker="s", s=size, alpha =1.0, cmap = "coolwarm")
    img4 = ax[1,1].scatter( x[:,0].detach(),  x[:,1].detach(), c = e22.detach(), marker="s", s=size, alpha =1.0, cmap = "coolwarm")
    img5 = ax[1,2].scatter( x[:,0].detach(),  x[:,1].detach(), c = e12.detach(), marker="s", s=size, alpha =1.0, cmap = "coolwarm")

    cb4 = fig.colorbar(img3, ax = ax[1,0], location="right", pad=0.2, shrink=0.8)
    cb5 = fig.colorbar(img4, ax = ax[1,1], location="right", pad=0.2, shrink=0.8)
    cb6 = fig.colorbar(img5, ax = ax[1,2], location="right", pad=0.2, shrink=0.8)
    ax[1,0].set_aspect('equal', adjustable='box')
    ax[1,1].set_aspect('equal', adjustable='box')
    ax[1,2].set_aspect('equal', adjustable='box')

    tick_font_size = 28
    cb1.ax.tick_params(labelsize=tick_font_size)
    cb2.ax.tick_params(labelsize=tick_font_size)
    cb3.ax.tick_params(labelsize=tick_font_size)
    cb4.ax.tick_params(labelsize=tick_font_size)
    cb5.ax.tick_params(labelsize=tick_font_size)
    cb6.ax.tick_params(labelsize=tick_font_size)

    fig.savefig('Results/2D_'+name+'.pdf', transparent=True) 
    plt.close()


def Export_Displacement_to_vtk(mesh_name, Model_u, ep ):

    u_x = [u for u in Model_u.nodal_values[0]]
    u_y = [u for u in Model_u.nodal_values[1]]

    meshBeam = meshio.read('geometries/'+mesh_name)
    u = torch.stack([torch.cat(u_x),torch.cat(u_y)],dim=1)

    coordinates = [coord for coord in Model_u.coordinates]
    coordinates = torch.cat(coordinates,dim=0)

    sol = meshio.Mesh(coordinates.data, {"triangle6":meshBeam.cells_dict["triangle6"]},
                        point_data={"U":u.data})

    sol.write('Results/Paraview/Displacement_'+mesh_name[0:-4]+'_ep_'+str(ep)+'.vtk')

def Export_Displacement1_to_vtk(mesh_name, Model_u, ep ):

    u_x = [u for u in Model_u.nodal_values[0]]
    u_y = [u for u in Model_u.nodal_values[1]]

    meshBeam = meshio.read('geometries/'+mesh_name)
    u = torch.stack([torch.cat(u_x),torch.cat(u_y)],dim=1)

    coordinates = [coord for coord in Model_u.coordinates]
    coordinates = torch.cat(coordinates,dim=0)

    sol = meshio.Mesh(coordinates.data, {"triangle":meshBeam.cells_dict["triangle"]},
                        point_data={"U":u.data})

    sol.write('Results/Paraview/Displacement_'+mesh_name[0:-4]+'_ep_'+str(ep)+'.vtk')

def Export_Stress_to_vtk(Mesh, Model, ep ):

    mesh_name = Mesh.name_mesh

    stress_all_coord = [(Model.coordinates[i]).clone().detach().requires_grad_(True) for i in range(len(Model.coordinates))]
    stress_all_cell_IDs = torch.tensor([torch.tensor(Mesh.GetCellIds(i),dtype=torch.int)[0] for i in stress_all_coord])
    stress_all_coord = (torch.cat(stress_all_coord)).clone().detach().requires_grad_(True)

    stress = Model(stress_all_coord, stress_all_cell_IDs)

    # s11 = [u for u in Model.nodal_values[0]]
    # s22 = [u for u in Model.nodal_values[1]]
    # s12 = [u for u in Model.nodal_values[2]]

    meshBeam = meshio.read('geometries/'+mesh_name)
    u = torch.stack([stress[0,:],stress[1,:], stress[2,:]],dim=1)

    coordinates = [coord for coord in Model.coordinates]
    coordinates = torch.cat(coordinates,dim=0)

    sol = meshio.Mesh(coordinates.data, {"triangle":meshBeam.cells_dict["triangle"]},
                        point_data={"stress":u.data})

    sol.write('Results/Paraview/Stress_'+mesh_name[0:-4]+'_ep_'+str(ep)+'.vtk')