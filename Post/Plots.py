
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
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = "13"

def PlotSolution_Coordinates_Analytical(A,E,InitialCoordinates,Coordinates,TrialCoordinates,AnalyticSolution,model,name):

    #plt.plot(InitialCoordinates,[coord*0 for coord in InitialCoordinates],'+k', markersize=2, label = 'Initial Nodes')
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

def Plot_Compare_Loss2l2norm(error,error2,name):
    # Lift to be able to use semilogy
    error3 = error-np.min(error)
    plt.semilogy(error2)
    # plt.ylabel(r'$\Vert \underline{u}_{ex} - \underline{u}_{NN} \Vert^2$')
    plt.ylabel(r'$\Xi$')
    plt.ylim((0.01,20))
    ax2 = plt.gca().twinx()
    ax2.semilogy(error3,color='#F39C12')
    ax2.set_ylabel(r'Lifted $J\left(\underline{u}\left(\underline{x}\right)\right)$')
    plt.savefig('Results/'+name+'.pdf', transparent=True) 
    plt.clf()

def Plot_end(error,error2):
    # Lift to be able to use semilogy
    #error3 = error-np.min(error)
    error3 = [-x for x in error]
    #plt.semilogy(error2[-1000:-1])
    #plt.ylabel(r'$\Vert \underline{u}_{ex} - \underline{u}_{NN} \Vert^2$')
    ax2 = plt.gca().twinx()
    ax2.semilogy(error3[1000:],color='#F39C12', label="Min = "+str(np.format_float_scientific(np.min(error), precision=3))+", final = " +str(np.format_float_scientific(error[-1], precision=3)))
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
    u_150 = BeamROM(TrialCoordinates,PaperPara)
    u_analytical_150 = AnalyticSolution(A,PaperPara.item(),TrialCoordinates.data,u0,uL)
    plt.plot(TrialCoordinates.data,u_analytical_150, color="#01426A", label = r'$E = 150~$MPa Analytical solution')
    plt.plot(TrialCoordinates.data,u_150.data,'--', color="#01426A", label = r'$E = 150~$MPa HiDeNN solution')

    PaperPara = torch.tensor([200])
    PaperPara = PaperPara[:,None] # Add axis so that dimensions match
    u_200 = BeamROM(TrialCoordinates,PaperPara)
    u_analytical_200 = AnalyticSolution(A,PaperPara.item(),TrialCoordinates.data,u0,uL)
    plt.plot(TrialCoordinates.data,u_analytical_200, color="#00677F", label = r'$E = 200~$MPa Analytical solution')
    plt.plot(TrialCoordinates.data,u_200.data,'--',color="#00677F", label = r'$E = 200~$MPa HiDeNN solution')

    PaperPara = torch.tensor([100])
    PaperPara = PaperPara[:,None] # Add axis so that dimensions match
    u_100 = BeamROM(TrialCoordinates,PaperPara)
    u_analytical_100 = AnalyticSolution(A,PaperPara.item(),TrialCoordinates.data,u0,uL)
    plt.plot(TrialCoordinates.data,u_analytical_100,color="#A92021", label = r'$E = 100~$MPa Analytical solution')
    plt.plot(TrialCoordinates.data,u_100.data,'--',color="#A92021", label = r'$E = 100~$MPa HiDeNN solution')
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
        E = torch.tensor([E])
        E = E[:,None] # Add axis so that dimensions match
        u_E = BeamROM(TrialCoordinates,E)
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
    slider = widgets.FloatSlider(value=0, min=100, max=200, step=0.01, description='E (GPa)')

    # Connect the slider to the interactive plot function
    interactive_plot_widget = interact(interactive_plot, E=slider)

def PlotModes(BeamROM,TrialCoordinates,TrialPara,A,AnalyticSolution,name_model):
    import torch
    Space_modes = [BeamROM.Space_modes[l](TrialCoordinates) for l in range(BeamROM.n_modes)]
    u_i = torch.cat(Space_modes,dim=1)  
    for mode in range(BeamROM.n_modes):
        Para_mode_List = [BeamROM.Para_modes[mode][l](TrialPara)[:,None] for l in range(BeamROM.n_para)]
        if mode == 0:
            lambda_i = torch.unsqueeze(torch.cat(Para_mode_List,dim=1), dim=0)
            # Para_modes = torch.unsqueeze(Para_modes, dim=0)
        else:
            New_mode = torch.unsqueeze(torch.cat(Para_mode_List,dim=1), dim=0)
            lambda_i = torch.vstack((lambda_i,New_mode))

    for mode in range(BeamROM.n_modes):
        plt.plot(TrialCoordinates.data,u_i[:,mode].data,label='Mode'+str(mode+1))
        plt.xlabel('x (mm)')
        plt.legend(loc="upper left")
    plt.savefig('Results/Pre_trained_Space_modes'+str(BeamROM.n_modes)+'.pdf', transparent=True)  
    plt.clf()
        # plt.show()

    for mode in range(BeamROM.n_modes):
        plt.plot(TrialPara.data,lambda_i[mode,:,0].data,label='Mode'+str(mode+1))
        plt.xlabel('E (GPa)')
        plt.legend(loc="upper left")
    plt.savefig('Results/Pre_trained_Para_modes'+str(BeamROM.n_modes)+'.pdf', transparent=True)  
    plt.clf()
        # plt.show()