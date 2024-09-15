
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
try:
    import tikzplotlib
except:
    print('* WARNING: could not load tikzplotlib')
from matplotlib.legend import Legend
Legend._ncol = property(lambda self: self._ncols)

from matplotlib.lines import Line2D

Line2D._us_dashSeq    = property(lambda self: self._dash_pattern[1])
Line2D._us_dashOffset = property(lambda self: self._dash_pattern[0])


from ..src.PDE_Library import Strain, Stress, InternalEnergy_2D, VonMises, VonMises_plain_strain, AnalyticSolution, AnalyticGradientSolution

def export_csv(Name,y, x='None'):
    import pandas as pd
    x_values = list(range(1,len(y)))
    a = np.stack((y,x_values),axis=1)
    df = pd.DataFrame(a, columns=['y', 'x'])
    df.to_csv('Results/'+Name+'.csv')

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
    tikzplotlib.save('Results/'+name+'.tikz', axis_height='6.5cm', axis_width='9cm') 

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
    tikzplotlib.save('Results/'+name+'.tikz', axis_height='6.5cm', axis_width='9cm') 
    plt.clf()

def PlotEnergyLoss(error,zoom,name):
    """Plots the error from the index given by zoom """
    plt.plot(error[zoom:])
    plt.xlabel(r'epochs')
    plt.ylabel(r'$J\left(\underline{u}\left(\underline{x}\right)\right)$')
    plt.savefig('Results/'+name+'.pdf', transparent=True)  
    #plt.show()
    plt.clf()

def PlotTrajectories(Coord_trajectories,name, show):
    """Plots the trajectories of the coordinates during training"""
    
    if len(Coord_trajectories)>5000:
        x = np.arange(len(Coord_trajectories))
        plt.plot(x[0::500], Coord_trajectories[0::500])
    else:
        plt.plot(Coord_trajectories)

    plt.xlabel(r'epochs')
    plt.ylabel(r'$x_i\left(\underline{x}\right)$')
    plt.savefig('Results/'+name+'.pdf', transparent=True)  
    tikzplotlib.save('Results/'+name+'.tikz', axis_height='6.5cm', axis_width='9cm') 
    if show:
        plt.show()
    plt.clf()

def Plot_NegLoss_Modes(Modes_flag,error,name, sign = "Negative",tikz = False):
    """To keep back compatibility with previous verions"""
    Plot_PosNegLoss_Modes(Modes_flag,error,name, sign = "Negative",tikz = tikz)

def Plot_PosNegLoss_Modes(Modes_flag,error,name, sign = "Negative",tikz = False, Zoom_required = False):
    # from matplotlib.legend import Legend
    # Legend._ncol = property(lambda self: self._ncols)
    # Legend._ncol = property(lambda self: self._ncols)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    g1 = ax.plot(Modes_flag, color='#247ab5ff')
    ax.tick_params(axis='y', colors='#247ab5ff')
    plt.ylabel(r'$m$',color='#247ab5ff')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel(r'Epochs')
    ax2 = ax.twinx()
    match sign:
        case "Positive":
            g2 = ax2.semilogy(torch.tensor(error), color='#d95319ff')
            ax2.set_ylabel(r'$ + J\left(\underline{u}\left(\underline{x}\right)\right)$',color='#d95319ff')

        case "Negative":
            ax2.invert_yaxis()
            g2 = ax2.semilogy(-torch.tensor(error), color='#d95319ff')
            ax2.set_ylabel(r'$ - J\left(\underline{u}\left(\underline{x}\right)\right)$',color='#d95319ff')

    # g2 = ax2.semilogy(-torch.tensor(error),label = r'$ - J\left(\underline{u}\left(\underline{x}\right)\right)$', color='#d95319ff')
    ax2.tick_params(axis='y', colors='#d95319ff')

    lns = g1+g2
    labs = [l.get_label() for l in lns]
    # ax.legend(lns, labs, loc="upper center")
    if tikz:
        import tikzplotlib
        tikzplotlib.save('Results/'+name+'.tex')
    plt.savefig('Results/'+name+'.pdf', transparent=True, bbox_inches = "tight")


    plt.clf() 

    if Modes_flag[0] == Modes_flag[-1]:
        Zoom_required = False

    if Zoom_required:
        import numpy as np
        Zoom_depth = np.min(np.where(np.array(Modes_flag) == np.array(Modes_flag)[0]+1))
        Zoom_start_index = int(np.floor(0.9*Zoom_depth))
        second_stages_epochs = len(error) - len(Modes_flag)
        Modes_flag.extend([Modes_flag[-1]]*second_stages_epochs)
        x_indexes = np.arange(len(Modes_flag[Zoom_start_index:]))+Zoom_start_index
        fig = plt.figure()
        ax = fig.add_subplot(111)
        g1 = ax.plot(x_indexes,Modes_flag[Zoom_start_index:], color='#247ab5ff')
        ax.tick_params(axis='y', colors='#247ab5ff')
        plt.ylabel(r'$m$',color='#247ab5ff')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel(r'Epochs')
        ax2 = ax.twinx()
        match sign:
            case "Positive":
                g2 = ax2.semilogy(x_indexes,torch.tensor(error[Zoom_start_index:]), color='#d95319ff')
                ax2.set_ylabel(r'$ + J\left(\underline{u}\left(\underline{x}\right)\right)$',color='#d95319ff')

            case "Negative":
                ax2.invert_yaxis()
                g2 = ax2.semilogy(x_indexes,-torch.tensor(error[Zoom_start_index:]), color='#d95319ff')
                ax2.set_ylabel(r'$ - J\left(\underline{u}\left(\underline{x}\right)\right)$',color='#d95319ff')

        # g2 = ax2.semilogy(-torch.tensor(error),label = r'$ - J\left(\underline{u}\left(\underline{x}\right)\right)$', color='#d95319ff')
        ax2.tick_params(axis='y', colors='#d95319ff')

        lns = g1+g2
        labs = [l.get_label() for l in lns]
        # ax.legend(lns, labs, loc="upper center")
        if tikz:
            import tikzplotlib
            tikzplotlib.save('Results/'+name+'_zoom.tex')
        plt.savefig('Results/'+name+'_zoom.pdf', transparent=True, bbox_inches = "tight")


        plt.clf() 



def Plot_L2error_Modes(Modes_flag,error,name, tikz = False):
    # from matplotlib.legend import Legend
    # Legend._ncol = property(lambda self: self._ncols)
    # Legend._ncol = property(lambda self: self._ncols)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    g1 = ax.plot(Modes_flag, color='#247ab5ff')
    ax.tick_params(axis='y', colors='#247ab5ff')
    plt.ylabel(r'$m$',color='#247ab5ff')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel(r'Epochs')
    ax2 = ax.twinx()

    g2 = ax2.semilogy(torch.tensor(error), color='#d95319ff')
    ax2.set_ylabel(r'$\eta$',color='#d95319ff')


    # g2 = ax2.semilogy(-torch.tensor(error),label = r'$ - J\left(\underline{u}\left(\underline{x}\right)\right)$', color='#d95319ff')
    ax2.tick_params(axis='y', colors='#d95319ff')

    lns = g1+g2
    labs = [l.get_label() for l in lns]
    # ax.legend(lns, labs, loc="upper center")
    if tikz:
        import tikzplotlib
        tikzplotlib.save('Results/'+name+'.tex')
    plt.savefig('Results/'+name+'.pdf', transparent=True, bbox_inches = "tight")


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

def Plot_Lossdecay_Modes(Modes_flag,decay,name,threshold,tikz = False):
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
    if tikz:
        import tikzplotlib
        tikzplotlib.save('Results/'+name+'.tex')
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

def Plot_Parametric_Young(BeamROM,TrialCoordinates,A,AnalyticSolution,name_model = 'tmp',tikz = False):
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
    if tikz:
        import tikzplotlib
        tikzplotlib.save('Results/Para_displacements'+name_model+'.tex')
    plt.show()
    plt.clf()

def Plot_BiParametric_Young(BeamROM,TrialCoordinates,A,AnalyticBiParametricSolution,name_model = 'tmp',tikz = False):
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
    u_150 = BeamROM(TrialCoordinates,[Paper150,PaperPara])
    u_analytical_150 = AnalyticBiParametricSolution(A,[PaperPara.item(),Paper150[0][0].item()],TrialCoordinates.data,u0,uL)
    plt.plot(TrialCoordinates.data,u_analytical_150, color="#01426A", label = r'$E = 150~$MPa Analytical solution')
    plt.plot(TrialCoordinates.data,u_150.data.view(-1),'--', color="#01426A", label = r'$E = 150~$MPa HiDeNN solution')

    PaperPara = torch.tensor([200])
    PaperPara = PaperPara[:,None] # Add axis so that dimensions match
    u_200 = BeamROM(TrialCoordinates,[Paper150,PaperPara])
    u_analytical_200 = AnalyticBiParametricSolution(A,[PaperPara.item(),Paper150[0][0].item()],TrialCoordinates.data,u0,uL)
    plt.plot(TrialCoordinates.data,u_analytical_200, color="#00677F", label = r'$E = 200~$MPa Analytical solution')
    plt.plot(TrialCoordinates.data,u_200.data.view(-1),'--',color="#00677F", label = r'$E = 200~$MPa HiDeNN solution')

    PaperPara = torch.tensor([100])
    PaperPara = PaperPara[:,None] # Add axis so that dimensions match
    u_100 = BeamROM(TrialCoordinates,[Paper150,PaperPara])
    u_analytical_100 = AnalyticBiParametricSolution(A,[PaperPara.item(),Paper150[0][0].item()],TrialCoordinates.data,u0,uL)
    plt.plot(TrialCoordinates.data,u_analytical_100,color="#A92021", label = r'$E = 100~$MPa Analytical solution')
    plt.plot(TrialCoordinates.data,u_100.data.view(-1),'--',color="#A92021", label = r'$E = 100~$MPa HiDeNN solution')
    plt.legend(loc="upper left")
    plt.xlabel('x (mm)')
    plt.ylabel('u (mm)')
    plt.savefig('Results/Para_displacements'+name_model+'.pdf', transparent=True)  
    if tikz:
        import tikzplotlib
        tikzplotlib.save('Results/Para_displacements'+name_model+'.tex')
    plt.show()
    plt.clf()

def Plot_Parametric_Young_Interactive(BeamROM,TrialCoordinates,A,AnalyticSolution,name_model):
    from ipywidgets import interact, widgets
    import torch
    def interactive_plot(E):
        match BeamROM.config["solver"]["IntegralMethod"]:
            case "Trapezoidal":
                u0                      = BeamROM.Space_modes[0].u_0                      # Left BC
                uL                      = BeamROM.Space_modes[0].u_L                      # Right BC
            case "Gaussian_quad":
                u0 = BeamROM.Space_modes[0].ListOfDirichletsBCsValues[0]
                uL = BeamROM.Space_modes[0].ListOfDirichletsBCsValues[1]
        # u0 = BeamROM.Space_modes[0].u_0
        # uL = BeamROM.Space_modes[0].u_L
        # Calculate the corresponding function values for each x value
        u_analytical_E = AnalyticSolution(A,E,TrialCoordinates.data,u0,uL)
        Nodal_coordinates = [BeamROM.Space_modes[0].coordinates[l].data for l in range(len(BeamROM.Space_modes[0].coordinates))]
        Nodal_coordinates = torch.cat(Nodal_coordinates)
        u_analytical_E_discrete = AnalyticSolution(A,E,Nodal_coordinates.data,u0,uL)
        E = torch.tensor([E],dtype=u0.dtype)
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
        E1 = torch.tensor([E1],dtype=u0.dtype)
        E1 = E1[:,None] # Add axis so that dimensions match
        E2 = torch.tensor([E2],dtype=u0.dtype)
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

def ExportFinalResult_VTK(Model_FEM,Mat,Name_export):
        import meshio
        # Get nodal values from the trained model
        vers = 'new_V2'
        if vers == 'new_V2':
            u_x = Model_FEM.U_interm[-1][:,0]
            u_y = Model_FEM.U_interm[-1][:,1]
            List_elems = torch.arange(0,Model_FEM.NElem,dtype=torch.int)
            Model_FEM.train()
            _,xg,detJ = Model_FEM()
            Model_FEM.eval()
            eps =  Strain(Model_FEM(xg, List_elems),xg).to(torch.device('cpu'))
            sigma =  torch.stack(Stress(eps[:,0], eps[:,1], eps[:,2], Mat.lmbda, Mat.mu),dim=1).to(torch.device('cpu'))
            sigma_VM2  = VonMises_plain_strain(sigma, Mat.lmbda, Mat.mu).to(torch.device('cpu'))
            u = torch.stack([u_x,u_y,torch.zeros(u_x.shape[0], dtype=u_x.dtype).to(u_x.device)],dim=1).cpu()
            import numpy as np
            Coord = torch.hstack([Model_FEM.X_interm[-1], torch.zeros(Model_FEM.X_interm[-1][:,1].shape, dtype=u_x.dtype).to(u_x.device)[:,None]]).cpu()
            Coord_converged = np.array(Coord.cpu())
            Connect_converged = Model_FEM.connectivity
            sol = meshio.Mesh(Coord_converged, {"triangle":(Connect_converged-1)},
            point_data={"U":u.data}, 
            cell_data={"eps": [eps.data], "sigma": [sigma.data],  "sigma_vm": [sigma_VM2.data]}, )
            sol.write(
                "Results/Paraview/sol_u_end_training_"+Name_export+".vtk", 
            )
        else:
            u_x = [u.to(torch.device('cpu')) for u in Model_FEM.nodal_values_x]
            u_y = [u.to(torch.device('cpu')) for u in Model_FEM.nodal_values_y]
            # Compute the strain 
            List_elems = torch.arange(0,Model_FEM.NElem,dtype=torch.int)
            Model_FEM.train()
            _,xg,detJ = Model_FEM()
            Model_FEM.eval()
            eps =  Strain(Model_FEM(xg, List_elems),xg).to(torch.device('cpu'))
            sigma =  torch.stack(Stress(eps[:,0], eps[:,1], eps[:,2], Mat.lmbda, Mat.mu),dim=1).to(torch.device('cpu'))
            # sigma_VM = VonMises(sigma)
            sigma_VM2  = VonMises_plain_strain(sigma, Mat.lmbda, Mat.mu).to(torch.device('cpu'))
            # X_interm_tot = Model_FEM.training_recap["X_interm_tot"]
            X_interm_tot = [x_i.to(torch.device('cpu')) for x_i in Model_FEM.training_recap["X_interm_tot"]]

            X_interm_tot = [torch.cat([x_i,torch.zeros(x_i.shape[0],1)],dim=1) for x_i in X_interm_tot]
            u = torch.stack([torch.cat(u_x),torch.cat(u_y),torch.zeros(torch.cat(u_x).shape[0])],dim=1)
            import numpy as np
            Coord_converged = np.array([[Model_FEM.coordinates[i][0][0].to(torch.device('cpu')).item(),Model_FEM.coordinates[i][0][1].to(torch.device('cpu')).item(),0] for i in range(len(Model_FEM.coordinates))])
            Connect_converged = Model_FEM.connectivity
            sol = meshio.Mesh(Coord_converged, {"triangle":(Connect_converged-1)},
            point_data={"U":u.data}, 
            cell_data={"eps": [eps.data], "sigma": [sigma.data],  "sigma_vm": [sigma_VM2.data]}, )
            sol.write(
                "Results/Paraview/sol_u_end_training_"+Name_export+".vtk", 
            )

def ExportHistoryResult_VTK(Model_FEM,Mat,Name_export):
        import meshio
        # X_interm_tot        = Model_FEM.training_recap["X_interm_tot"]
        # U_interm_tot        = Model_FEM.training_recap["U_interm_tot"]
        # Gen_interm_tot      = Model_FEM.training_recap["Gen_interm_tot"]
        # detJ_tot            = Model_FEM.training_recap["detJ_tot"]
        # Connectivity_tot    = Model_FEM.training_recap["Connectivity_tot"]
        X_interm_tot = [x_i.to(torch.device('cpu')) for x_i in Model_FEM.training_recap["X_interm_tot"]]
        U_interm_tot = [x_i.to(torch.device('cpu')) for x_i in Model_FEM.training_recap["U_interm_tot"]]
        Gen_interm_tot      = Model_FEM.training_recap["Gen_interm_tot"]
        detJ_tot = [torch.abs(det_i).to(torch.device('cpu')) for det_i in Model_FEM.training_recap["detJ_tot"]]
        detJ_current_tot = [torch.abs(det_i).to(torch.device('cpu')) for det_i in Model_FEM.training_recap["detJ_current_tot"]]

        #Debug
        D_detJ = [(torch.abs(detJ_tot[i]) - torch.abs(detJ_current_tot[i]))/torch.abs(detJ_tot[i]) for i in range(len(detJ_tot))]   
        ##

        Connectivity_tot    = Model_FEM.training_recap["Connectivity_tot"]
        # Add 3-rd dimension
        X_interm_tot    = [torch.cat([x_i,torch.zeros(x_i.shape[0],1)],dim=1) for x_i in X_interm_tot]
        U_interm_tot = [torch.cat([u,torch.zeros(u.shape[0],1)],dim=1) for u in U_interm_tot]

        for timestep in range(len(U_interm_tot)):
            sol = meshio.Mesh(X_interm_tot[timestep].data, {"triangle":Connectivity_tot[timestep].data},
            point_data={"U":U_interm_tot[timestep]}, 
            cell_data={"Gen": [Gen_interm_tot[timestep]], "detJ_0": [detJ_tot[timestep].data], "detJ": [detJ_current_tot[timestep].data], "D_detJ": [D_detJ[timestep].data]})

            sol.write(
                f"Results/Paraview/TimeSeries/solution_"+Name_export+f"_{timestep}.vtk",  
            )



def Plot_Eval_1d(model, config, Mat, model_du = []):

    new_coord = [coord for coord in model.coordinates]
    new_coord = torch.cat(new_coord,dim=0)

    L = config["geometry"]["L"]
    A = config["geometry"]["A"]
    E = config["material"]["E"]
    n_visu = config["postprocess"]["n_visualization"]


    if config["solver"]["IntegralMethod"] == "Gaussian_quad":
        model.mesh.Nodes = [[i+1,new_coord[i].item(),0,0] for i in range(len(model.mesh.Nodes))]
        model.mesh.ExportMeshVtk1D(flag_update = True)

        PlotCoordinates = torch.tensor([i for i in torch.linspace(0,L,n_visu)],dtype=torch.float64, requires_grad=True)
        IDs_plot = torch.tensor(model.mesh.GetCellIds(PlotCoordinates),dtype=torch.int)

        model.eval()
        u_predicted = model(PlotCoordinates, IDs_plot)[:,0]
        du_dx = torch.autograd.grad(u_predicted, PlotCoordinates, grad_outputs=torch.ones_like(u_predicted), create_graph=True)[0]

        Coordinates = [model.coordinates[i].data.item() for i in range(len(model.coordinates))]
        Coordinates_du = Coordinates


    if config["solver"]["IntegralMethod"] == "Trapezoidal":
        PlotCoordinates = torch.tensor([[i] for i in torch.linspace(0,L,n_visu)], dtype=torch.float64, requires_grad=True)
        u_predicted = model(PlotCoordinates)
        du_dx = torch.autograd.grad(u_predicted, PlotCoordinates, grad_outputs=torch.ones_like(u_predicted), create_graph=True)[0]

        Coordinates = [model.coordinates[i].data.item() for i in range(len(model.coordinates))]
        Coordinates_du = Coordinates

    if config["solver"]["IntegralMethod"] == "None":
        PlotCoordinates = torch.tensor([[i] for i in torch.linspace(0,L,n_visu)], dtype=torch.float64, requires_grad=True)
        u_predicted = model(PlotCoordinates)
        du_dx = model_du(PlotCoordinates)

        Coordinates = [model.coordinates[i].data.item() for i in range(len(model.coordinates))]
        Coordinates_du = [model_du.coordinates[i].data.item() for i in range(len(model_du.coordinates))]

    analytical_norm = torch.linalg.vector_norm(AnalyticSolution(A,E,PlotCoordinates.data)).data
    l2_loss = torch.linalg.vector_norm(AnalyticSolution(A,E,PlotCoordinates.data) - u_predicted).data/analytical_norm
    print(f'* Final l2 loss : {np.format_float_scientific(l2_loss, precision=4)}')

    l2_loss_grad = torch.linalg.vector_norm(AnalyticGradientSolution(A,E,PlotCoordinates.data) - du_dx).data/torch.linalg.vector_norm(AnalyticGradientSolution(A,E,PlotCoordinates.data)).data
    print(f'* Final l2 loss grad : {np.format_float_scientific(l2_loss_grad, precision=4)}')

    if config["solver"]["FrozenMesh"] == False:
        plt.scatter(model.original_coordinates,[coord*0 for coord in model.original_coordinates], s=6, color="pink", alpha=0.5, label = 'Initial nodal position')


    plt.plot(Coordinates,[coord*0 for coord in Coordinates],'.k', markersize=2, label = 'Nodal position')
    plt.plot(PlotCoordinates.data,AnalyticSolution(A,E,PlotCoordinates.data), label = 'Analytical solution')
    plt.plot(PlotCoordinates.data,u_predicted.data,'--', label = 'Predicted solution')
    plt.xlabel(r'$\underline{x}$ [m]')
    plt.ylabel(r'$\underline{u}\left(\underline{x}\right)$')
    plt.legend(loc="upper left")
    # plt.title('Displacement')
    plt.savefig('Results/Displacement.pdf', transparent=True) 
    tikzplotlib.save('Results/Displacement.tikz', axis_height='6.5cm', axis_width='9cm') 
    plt.show()
    plt.clf()

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(9, 7)

    if config["solver"]["FrozenMesh"] == False:
        plt.scatter(model.original_coordinates,[coord*0 for coord in model.original_coordinates], s=6, color="pink", alpha=0.5, label = 'Initial nodal position')
        
    plt.plot(Coordinates_du,[coord*0 for coord in Coordinates_du],'.k', markersize=2, label = 'Nodal position')
    plt.plot(PlotCoordinates.data,AnalyticGradientSolution(A,E,PlotCoordinates.data), label = 'Analytical solution')
    plt.plot(PlotCoordinates.data,du_dx.data,'--', label = 'Predicted solution')
    plt.xlabel(r'$\underline{x}$ [m]')
    plt.ylabel(r'$\frac{d\underline{u}}{dx}\left(\underline{x}\right)$')
    plt.legend(loc="upper left")
    # plt.title('Displacement')
    plt.savefig('Results/Gradient.pdf', transparent=True)  
    tikzplotlib.save('Results/Gradient.tikz', axis_height='6.5cm', axis_width='9cm') 

    plt.show()
    plt.clf()



def ExportSamplesforEval(model,Mat,config):

    MaxElemSize = config["interpolation"]["MaxElemSize2D"]
    ref = config["training"]["multiscl_max_refinment"]-1

    model.mesh.Nodes = [[i+1,model.coordinates[i][0][0].item(),model.coordinates[i][0][1].item(),0] for i in range(len(model.coordinates))]
    model.mesh.Connectivity = model.connectivity
    model.mesh.ExportMeshVtk(flag_update = True)

    coord = torch.tensor(np.load("../2D_example/eval_coordinates.npy"), dtype=torch.float64, requires_grad=True)
    List_elems = torch.tensor(model.mesh.GetCellIds(coord),dtype=torch.int)

    u = model(coord, List_elems)
    eps =  Strain(model(coord, List_elems),coord)
    sigma =  torch.stack(Stress(eps[:,0], eps[:,1], eps[:,2], Mat.lmbda, Mat.mu),dim=1)
    sigma_VM = VonMises_plain_strain(sigma, Mat.lmbda, Mat.mu)

    if config["solver"]["FrozenMesh"] == False:
        np.save("../2D_example/NN_solution/"+str(MaxElemSize/(2**ref)) +"_free_u.npy", np.array(u.detach()))
        np.save("../2D_example/NN_solution/"+str(MaxElemSize/(2**ref)) +"_free_sigma.npy", np.array(sigma.detach()))
        np.save("../2D_example/NN_solution/"+str(MaxElemSize/(2**ref)) +"_free_sigma_VM.npy", np.array(sigma_VM.detach()))

    else:
        np.save("../2D_example/NN_solution/"+str(MaxElemSize/(2**ref)) + "_fixed_u.npy", np.array(u.detach()))
        np.save("../2D_example/NN_solution/"+str(MaxElemSize/(2**ref)) +"_fixed_sigma.npy", np.array(sigma.detach()))
        np.save("../2D_example/NN_solution/"+str(MaxElemSize/(2**ref)) +"_fixed_sigma_VM.npy", np.array(sigma_VM.detach()))


def Plot_2D_PyVista(ROM_model, Mesh_object, config, E = 5e-3, theta = 0, scalar_field_name = 'Ux', scaling_factor = 20, Interactive_parameter = 'theta', Plot_mesh = True, color_map = 'viridis'):
    import pyvista as pv                                                            # Import PyVista
    import torch.nn as nn

    pv.global_theme.font.family = 'times'                                           # Arial, courier or times
    pv.global_theme.font.size = 40
    pv.global_theme.font.title_size = 40
    pv.global_theme.font.label_size = 40
    pv.global_theme.font.fmt = '%.2e'

    filename = 'Geometries/'+Mesh_object.name_mesh                                  # Load mesh (used for projecting the solution only) 
    mesh = pv.read(filename)                                                        # Create pyvista mesh    
    Nodes = np.stack(Mesh_object.Nodes)                                             # Read coordinates where to evaluate the model

    match config["postprocess"]["PyVista_Type"]:
        case "Frame":                                                               # Only a single static solution
            E = torch.tensor([E],dtype=torch.float32)
            E = E[:,None]
            theta = torch.tensor([theta],dtype=torch.float32)
            theta = theta[:,None] 
            Para_coord_list = nn.ParameterList((E,theta))
            ROM_model.eval()                                                        # Put model in evaluation mode
            u_sol = ROM_model(torch.tensor(Nodes[:,1:]),Para_coord_list)            # Evaluate model

            match ROM_model.n_para:
                case 1:
                    u = torch.stack([(u_sol[0,:,0]),(u_sol[1,:,0]),torch.zeros(u_sol[0,:,0].shape[0])],dim=1)
                case 2:
                    u = torch.stack([(u_sol[0,:,0,0]),(u_sol[1,:,0,0]),torch.zeros(u_sol[0,:,0,0].shape[0])],dim=1)
            # Plot the mesh
            # scalar_field_name = 'Ux'
            mesh.point_data['U'] = u.data
            mesh.point_data['Ux'] = u[:,0].data
            mesh.point_data['Uy'] = u[:,1].data
            mesh.point_data['Uz'] = u[:,2].data

            plotter = pv.Plotter()
            plotter.add_mesh(mesh, scalars=scalar_field_name, cmap=color_map, scalar_bar_args={'title': 'Displacement', 'vertical': True})
            plotter.view_xy()
            def screenshot():
                print("Window size ", plotter.window_size)
                plotter.save_graphic("Results/"+config["geometry"]["Name"]
                                    +"_"+config["postprocess"]["scalar_field_name"]
                                    +"_"+config["postprocess"]["PyVista_Type"]
                                    +"_"+config["postprocess"]["Name_export"]
                                    +"_PyVista.pdf",raster=False)
                plotter.screenshot("Results/"+config["geometry"]["Name"]
                                    +"_"+config["postprocess"]["scalar_field_name"]
                                    +"_"+config["postprocess"]["PyVista_Type"]
                                    +"_"+config["postprocess"]["Name_export"]
                                    +"_PyVista.png", transparent_background=True, window_size=[2560,int(2560*plotter.window_size[1]/plotter.window_size[0])])
                print("Camera position ", plotter.camera_position)
            plotter.add_key_event("s", screenshot)
            plotter.show()
        case "Static":

            plotter = pv.Plotter(shape=(1, 2))                                      # Create a two-subfigures environment 
            plotter.subplot(0, 0)                                                   # Work on the first one
            filename = 'Geometries/'+Mesh_object.name_mesh
            mesh3 = pv.read(filename)
            parameter = config["parameters"]["para_1_min"]                          # Define the parameter to adjust and its initial value
            Param_trial_1 = torch.tensor([parameter],dtype=torch.float32)
            Param_trial_1 = Param_trial_1[:,None] 
            Param_trial_2 = torch.tensor([theta],dtype=torch.float32)               # Use given or default value of second parameter
            Param_trial_2 = Param_trial_1[:,None] 
            Para_coord_list = nn.ParameterList((Param_trial_1,Param_trial_2))
            ROM_model.eval()
            u_sol = ROM_model(torch.tensor(Nodes[:,1:]),Para_coord_list)
            match ROM_model.n_para:
                case 1:
                    u3 = torch.stack([(u_sol[0,:,0]),(u_sol[1,:,0]),torch.zeros(u_sol[0,:,0].shape[0])],dim=1)
                case 2:
                    u3 = torch.stack([(u_sol[0,:,0,0]),(u_sol[1,:,0,0]),torch.zeros(u_sol[0,:,0,0].shape[0])],dim=1)
            mesh3.point_data['U'] = u3.data
            mesh3.point_data['Ux'] = u3[:,0].data
            mesh3.point_data['Uy'] = u3[:,1].data
            mesh3.point_data['Uz'] = u3[:,2].data
            u3[:,2]+=0
            plotter.add_mesh(mesh3.warp_by_vector(vectors="U",factor=scaling_factor,inplace=True), scalars=scalar_field_name, cmap=color_map, scalar_bar_args={r'title': scalar_field_name+', theta ='+str(theta) , 'vertical': False}, show_edges=Plot_mesh)

            # Function to update the solution based on the parameter
            def update_solution_E(value):
                # plotter.clear()
                parameter = value
                stiffness = torch.tensor([parameter],dtype=torch.float32)
                stiffness = stiffness[:,None] 
                Param_trial = torch.tensor([theta],dtype=torch.float32)         # Use given or default value of second parameter
                Param_trial = Param_trial[:,None] 
                Para_coord_list = nn.ParameterList((stiffness,Param_trial))
                u_sol = ROM_model(torch.tensor(Nodes[:,1:]),Para_coord_list)
                match ROM_model.n_para:
                    case 1:
                        u3 = torch.stack([(u_sol[0,:,0]),(u_sol[1,:,0]),torch.zeros(u_sol[0,:,0].shape[0])],dim=1)
                    case 2:
                        u3 = torch.stack([(u_sol[0,:,0,0]),(u_sol[1,:,0,0]),torch.zeros(u_sol[0,:,0,0].shape[0])],dim=1)
                mesh3 = pv.read(filename)
                u3[:,2]+=200*value
                # mesh.warp_by_vector(vectors="U",factor=-20.0,inplace=True)
                mesh3.point_data['U'] = u3.data
                mesh3.point_data['Ux'] = u3[:,0].data
                mesh3.point_data['Uy'] = u3[:,1].data
                mesh3.point_data['Uz'] = u3[:,2].data
                plotter.add_mesh(mesh3.warp_by_vector(vectors="U",factor=scaling_factor,inplace=True), scalars=scalar_field_name, cmap=color_map, scalar_bar_args={r'title': scalar_field_name+', theta ='+str(theta) , 'vertical': False}, show_edges=Plot_mesh)
                return
            labels = dict(zlabel='E (MPa)', xlabel='x (mm)', ylabel='y (mm)')

            parameters_vect = [2e-3,3e-3,4e-3,5e-3,6e-3,7e-3,8e-3,9e-3,10e-3]

            for param in parameters_vect:
                update_solution_E(param)
            plotter.show_grid(
                color='gray',
                location='outer',
                grid='back',
                ticks='outside',
                xtitle='x (mm)',
                ytitle='y (mm)',
                ztitle='E (MPa)',
                font_size=10,
            )
            plotter.add_text("theta = 0", font_size=10)
            plotter.add_axes(**labels)

            plotter.subplot(0, 1)
            filename = 'Geometries/'+Mesh_object.name_mesh
            mesh2 = pv.read(filename)
            # Define the parameter to adjust and its initial value
            parameter = 1e-3

            Param_trial = torch.tensor([parameter],dtype=torch.float32)
            Param_trial = Param_trial[:,None] # Add axis so that dimensions match
            Para_coord_list = nn.ParameterList((Param_trial,Param_trial))

            ROM_model.eval()
            u_sol = ROM_model(torch.tensor(Nodes[:,1:]),Para_coord_list)
            match ROM_model.n_para:
                case 1:
                    u2 = torch.stack([(u_sol[0,:,0]),(u_sol[1,:,0]),torch.zeros(u_sol[0,:,0].shape[0])],dim=1)
                case 2:
                    u2 = torch.stack([(u_sol[0,:,0,0]),(u_sol[1,:,0,0]),torch.zeros(u_sol[0,:,0,0].shape[0])],dim=1)
            mesh2.point_data['U'] = u2.data
            mesh2.point_data['Ux'] = u2[:,0].data
            mesh2.point_data['Uy'] = u2[:,1].data
            mesh2.point_data['Uz'] = u2[:,2].data
            u2[:,2]+=0
            # plotter.add_mesh(mesh.warp_by_vector(vectors="U",factor=20.0,inplace=True), scalars=scalar_field_name, cmap=color_map, scalar_bar_args={'title': 'Displacement', 'vertical': False}, show_edges=Plot_mesh)

            # Function to update the solution based on the parameter
            def update_solution_t(value):
                # plotter.clear()
                parameter = value
                stiffness = torch.tensor([E],dtype=torch.float32)
                stiffness = stiffness[:,None] # Add axis so that dimensions match
                Param_trial = torch.tensor([parameter],dtype=torch.float32)
                Param_trial = Param_trial[:,None] # Add axis so that dimensions match
                Para_coord_list = nn.ParameterList((stiffness,Param_trial))
                u_sol = ROM_model(torch.tensor(Nodes[:,1:]),Para_coord_list)
                match ROM_model.n_para:
                    case 1:
                        u2 = torch.stack([(u_sol[0,:,0]),(u_sol[1,:,0]),torch.zeros(u_sol[0,:,0].shape[0])],dim=1)
                    case 2:
                        u2 = torch.stack([(u_sol[0,:,0,0]),(u_sol[1,:,0,0]),torch.zeros(u_sol[0,:,0,0].shape[0])],dim=1)
                mesh2 = pv.read(filename)
                u2[:,2]+=0.25*value
                # mesh.warp_by_vector(vectors="U",factor=-20.0,inplace=True)
                mesh2.point_data['U'] = u2.data
                mesh2.point_data['Ux'] = u2[:,0].data
                mesh2.point_data['Uy'] = u2[:,1].data
                mesh2.point_data['Uz'] = u2[:,2].data
                plotter.add_mesh(mesh2.warp_by_vector(vectors="U",factor=scaling_factor,inplace=True), scalars=scalar_field_name, cmap=color_map, scalar_bar_args={r'title': scalar_field_name+', E = '+str(E), 'vertical': False}, show_edges=Plot_mesh)
                return
            labels = dict(zlabel='E (MPa)', xlabel='x (mm)', ylabel='y (mm)')

            parameters_vect = [0,torch.pi/4,torch.pi/2,3*torch.pi/4,torch.pi,5*torch.pi/4,3*torch.pi/2,7*torch.pi/4,2*torch.pi]

            for param in parameters_vect:
                update_solution_t(param)
            plotter.show_grid(
                color='gray',
                location='outer',
                grid='back',
                ticks='outside',
                xtitle='x (mm)',
                ytitle='y (mm)',
                ztitle='theta (rad)',
                font_size=10,
            )
            plotter.add_axes(**labels)
            plotter.add_text("E ="+str(E), font_size=10)
            plotter.view_xy()
            def screenshot():
                print("Window size ", plotter.window_size)
                plotter.save_graphic("Results/"+config["geometry"]["Name"]
                                    +"_"+config["postprocess"]["scalar_field_name"]
                                    +"_"+config["postprocess"]["PyVista_Type"]
                                    +"_"+config["postprocess"]["Name_export"]
                                    +"_PyVista.pdf",raster=False)
                plotter.screenshot("Results/"+config["geometry"]["Name"]
                                    +"_"+config["postprocess"]["scalar_field_name"]
                                    +"_"+config["postprocess"]["PyVista_Type"]
                                    +"_"+config["postprocess"]["Name_export"]
                                    +"_PyVista.png", transparent_background=True, window_size=[2560,int(2560*plotter.window_size[1]/plotter.window_size[0])])
                print("Camera position ", plotter.camera_position)
            plotter.add_key_event("s", screenshot)
            plotter.show()
        case "Interactive":

            match Interactive_parameter:
                case 'E':
                    parameter = config["parameters"]["para_1_min"]
                    stiffness = torch.tensor([parameter],dtype=torch.float32)
                    stiffness = stiffness[:,None] 
                    Param_trial = torch.tensor([theta],dtype=torch.float32)             # Use given or default value of second parameter
                    Param_trial = Param_trial[:,None] 
                    Para_coord_list = nn.ParameterList((stiffness,Param_trial))
                case 'theta':
                    parameter = config["parameters"]["para_2_min"]
                    stiffness = torch.tensor([E],dtype=torch.float32)
                    stiffness = stiffness[:,None] 
                    Param_trial = torch.tensor([parameter],dtype=torch.float32)         # Use given or default value of second parameter
                    Param_trial = Param_trial[:,None] 
                    Para_coord_list = nn.ParameterList((stiffness,Param_trial))

            ROM_model.eval()
            u_sol = ROM_model(torch.tensor(Nodes[:,1:]),Para_coord_list)
            match ROM_model.n_para:
                case 1:
                    u = torch.stack([(u_sol[0,:,0]),(u_sol[1,:,0]),torch.zeros(u_sol[0,:,0].shape[0])],dim=1)
                case 2:
                    u = torch.stack([(u_sol[0,:,0,0]),(u_sol[1,:,0,0]),torch.zeros(u_sol[0,:,0,0].shape[0])],dim=1)
            mesh.point_data['U'] = u.data
            mesh.point_data['Ux'] = u[:,0].data
            mesh.point_data['Uy'] = u[:,1].data
            mesh.point_data['Uz'] = u[:,2].data
            plotter = pv.Plotter()
            plotter.add_mesh(mesh.warp_by_vector(vectors="U",factor=scaling_factor,inplace=True), scalars=scalar_field_name, cmap=color_map, scalar_bar_args={'title': 'Displacement', 'vertical': False}, show_edges=Plot_mesh)

            # Function to update the solution based on the parameter
            def update_solution2(value):
                # plotter.clear()
                parameter = value
                match Interactive_parameter:
                    case 'E':
                        parameter = config["parameters"]["para_1_min"]
                        stiffness = torch.tensor([value],dtype=torch.float32)
                        stiffness = stiffness[:,None] 
                        Param_trial = torch.tensor([theta],dtype=torch.float32)             # Use given or default value of second parameter
                        Param_trial = Param_trial[:,None] 
                        Para_coord_list = nn.ParameterList((stiffness,Param_trial))
                    case 'theta':
                        parameter = config["parameters"]["para_2_min"]
                        stiffness = torch.tensor([E],dtype=torch.float32)
                        stiffness = stiffness[:,None] 
                        Param_trial = torch.tensor([value],dtype=torch.float32)         # Use given or default value of second parameter
                        Param_trial = Param_trial[:,None] 
                        Para_coord_list = nn.ParameterList((stiffness,Param_trial))
                u_sol = ROM_model(torch.tensor(Nodes[:,1:]),Para_coord_list)
                match ROM_model.n_para:
                    case 1:
                        u = torch.stack([(u_sol[0,:,0]),(u_sol[1,:,0]),torch.zeros(u_sol[0,:,0].shape[0])],dim=1)
                    case 2:
                        u = torch.stack([(u_sol[0,:,0,0]),(u_sol[1,:,0,0]),torch.zeros(u_sol[0,:,0,0].shape[0])],dim=1)
                # u[:,2]+=200*value
                mesh.warp_by_vector(vectors="U",factor=-scaling_factor,inplace=True)
                mesh.point_data['U'] = u.data
                mesh.point_data['Ux'] = u[:,0].data
                mesh.point_data['Uy'] = u[:,1].data
                mesh.point_data['Uz'] = u[:,2].data
                # print(mesh.get_data_range(scalar_field_name))
                plotter.mapper.scalar_range = mesh.get_data_range(scalar_field_name)
                mesh.warp_by_vector(vectors="U",factor=scaling_factor,inplace=True)

                # plotter.render()
                return
            match Interactive_parameter:
                case 'E':
                    Slider_min = config["parameters"]["para_1_min"]
                    Slider_max = config["parameters"]["para_1_max"]
                    plotter.add_slider_widget(update_solution2, [Slider_min, Slider_max], title='E (MPa)')

                case 'theta':
                    Slider_min = config["parameters"]["para_2_min"]
                    Slider_max = config["parameters"]["para_2_max"]
                    plotter.add_slider_widget(update_solution2, [Slider_min, Slider_max], title='theta (rad)')
            plotter.view_xy()
            def screenshot():
                print("Window size ", plotter.window_size)
                plotter.save_graphic("Results/"+config["geometry"]["Name"]
                                    +"_"+config["postprocess"]["scalar_field_name"]
                                    +"_"+config["postprocess"]["PyVista_Type"]
                                    +"_"+config["postprocess"]["Name_export"]
                                    +"_PyVista.pdf",raster=False)
                plotter.screenshot("Results/"+config["geometry"]["Name"]
                                    +"_"+config["postprocess"]["scalar_field_name"]
                                    +"_"+config["postprocess"]["PyVista_Type"]
                                    +"_"+config["postprocess"]["Name_export"]
                                    +"_PyVista.png", transparent_background=True, window_size=[2560,int(2560*plotter.window_size[1]/plotter.window_size[0])])
                print("Camera position ", plotter.camera_position)
            plotter.add_key_event("s", screenshot)
            plotter.show()

        case "DualSliders":


            stiffness = torch.tensor([E],dtype=torch.float32)
            stiffness = stiffness[:,None] 
            Param_trial = torch.tensor([theta],dtype=torch.float32)             # Use given or default value of second parameter
            Param_trial = Param_trial[:,None] 
            Para_coord_list = nn.ParameterList((stiffness,Param_trial))

            ROM_model.eval()
            u_sol = ROM_model(torch.tensor(Nodes[:,1:]),Para_coord_list)
            match ROM_model.n_para:
                case 1:
                    u = torch.stack([(u_sol[0,:,0]),(u_sol[1,:,0]),torch.zeros(u_sol[0,:,0].shape[0])],dim=1)
                case 2:
                    u = torch.stack([(u_sol[0,:,0,0]),(u_sol[1,:,0,0]),torch.zeros(u_sol[0,:,0,0].shape[0])],dim=1)
            mesh.point_data['U'] = u.data
            mesh.point_data['Ux'] = u[:,0].data
            mesh.point_data['Uy'] = u[:,1].data
            mesh.point_data['Uz'] = u[:,2].data
            plotter = pv.Plotter()
            plotter.add_mesh(mesh.warp_by_vector(vectors="U",factor=scaling_factor,inplace=True), scalars=scalar_field_name, cmap=color_map, scalar_bar_args={'title': 'Displacement', 'vertical': False}, show_edges=Plot_mesh)

            class UpdateSliders:
                def __init__(self):
                    self.kwargs = {
                        'E': E,
                        'theta': theta,
                    }
                def __call__(self, param, value):
                    self.kwargs[param] = value
                    self.update()

                def update(self):
                    stiffness = torch.tensor([self.kwargs['E']],dtype=torch.float32)
                    stiffness = stiffness[:,None] 
                    Param_trial = torch.tensor([self.kwargs['theta']],dtype=torch.float32)         # Use given or default value of second parameter
                    Param_trial = Param_trial[:,None] 
                    Para_coord_list = nn.ParameterList((stiffness,Param_trial))
                    u_sol = ROM_model(torch.tensor(Nodes[:,1:]),Para_coord_list)
                    match ROM_model.n_para:
                        case 1:
                            u = torch.stack([(u_sol[0,:,0]),(u_sol[1,:,0]),torch.zeros(u_sol[0,:,0].shape[0])],dim=1)
                        case 2:
                            u = torch.stack([(u_sol[0,:,0,0]),(u_sol[1,:,0,0]),torch.zeros(u_sol[0,:,0,0].shape[0])],dim=1)
                    # u[:,2]+=200*value
                    mesh.warp_by_vector(vectors="U",factor=-scaling_factor,inplace=True)
                    mesh.point_data['U'] = u.data
                    mesh.point_data['Ux'] = u[:,0].data
                    mesh.point_data['Uy'] = u[:,1].data
                    mesh.point_data['Uz'] = u[:,2].data
                    # print(mesh.get_data_range(scalar_field_name))
                    plotter.mapper.scalar_range = mesh.get_data_range(scalar_field_name)
                    mesh.warp_by_vector(vectors="U",factor=scaling_factor,inplace=True)



            Slider_min_E = config["parameters"]["para_1_min"]
            Slider_max_E = config["parameters"]["para_1_max"]

            Slider_min_t = config["parameters"]["para_2_min"]
            Slider_max_t = config["parameters"]["para_2_max"]

            engine = UpdateSliders()

            plotter.add_slider_widget(  callback=lambda value: engine('E', value),
                                        rng = [Slider_min_E, Slider_max_E], 
                                        title='E (MPa)',
                                        pointa=(0.3, 0.925),
                                        pointb=(0.6, 0.925))

            plotter.add_slider_widget(  callback=lambda value: engine('theta', value),
                                        rng = [Slider_min_t, Slider_max_t], 
                                        title='theta (rad)',
                                        pointa=(0.64, 0.925),
                                        pointb=(0.94, 0.925))
            plotter.view_xy()
            def screenshot():
                print(engine.kwargs["E"])
                print("Window size ", plotter.window_size)
                plotter.save_graphic("Results/"+config["geometry"]["Name"]
                                    +"_"+config["postprocess"]["scalar_field_name"]
                                    +"_"+config["postprocess"]["PyVista_Type"]
                                    +"_E_"+"{:.2e}".format(engine.kwargs["E"])
                                    +"_theta_"+"{:.2e}".format(engine.kwargs["theta"])
                                    +"_"+config["postprocess"]["Name_export"]
                                    +"_PyVista.pdf",raster=False)
                plotter.screenshot("Results/"+config["geometry"]["Name"]
                                    +"_"+config["postprocess"]["scalar_field_name"]
                                    +"_"+config["postprocess"]["PyVista_Type"]
                                    +"_E_"+"{:.2e}".format(engine.kwargs["E"])
                                    +"_theta_"+"{:.2e}".format(engine.kwargs["theta"])
                                    +"_"+config["postprocess"]["Name_export"]
                                    +"_PyVista.png", transparent_background=True, window_size=[2560,int(2560*plotter.window_size[1]/plotter.window_size[0])])
                print("Camera position ", plotter.camera_position)
            def restart_camera():
                plotter.view_xy()
            print("************** Available commands *************\n")
            print("* Take a screenshot : s\n")
            print("* Reset the camera  : r\n")

            plotter.add_key_event("s", screenshot)
            plotter.add_key_event("r", restart_camera)


            plotter.show()

def Normalized_error_1D(model, config, Mat, model_du = []):

    new_coord = [coord for coord in model.coordinates]
    new_coord = torch.cat(new_coord,dim=0)

    L = config["geometry"]["L"]
    A = config["geometry"]["A"]
    E = config["material"]["E"]
    n_visu = config["postprocess"]["n_visualization"]

    if config["solver"]["IntegralMethod"] == "Gaussian_quad":
        model.mesh.Nodes = [[i+1,new_coord[i].item(),0,0] for i in range(len(model.mesh.Nodes))]
        model.mesh.ExportMeshVtk1D(flag_update = True)

        PlotCoordinates = torch.tensor([i for i in torch.linspace(0,L,n_visu)],dtype=torch.float64, requires_grad=True)
        IDs_plot = torch.tensor(model.mesh.GetCellIds(PlotCoordinates),dtype=torch.int)

        model.eval()
        u_predicted = model(PlotCoordinates, IDs_plot)[:,0]
        du_dx = torch.autograd.grad(u_predicted, PlotCoordinates, grad_outputs=torch.ones_like(u_predicted), create_graph=True)[0]


    if config["solver"]["IntegralMethod"] == "Trapezoidal":
        PlotCoordinates = torch.tensor([[i] for i in torch.linspace(0,L,n_visu)], dtype=torch.float64, requires_grad=True)
        u_predicted = model(PlotCoordinates)
        du_dx = torch.autograd.grad(u_predicted, PlotCoordinates, grad_outputs=torch.ones_like(u_predicted), create_graph=True)[0]
    
    if config["solver"]["IntegralMethod"] == "None":
        PlotCoordinates = torch.tensor([[i] for i in torch.linspace(0,L,n_visu)], dtype=torch.float64, requires_grad=True)
        u_predicted = model(PlotCoordinates)
        du_dx = model_du(PlotCoordinates)



    l2_loss = torch.linalg.vector_norm(AnalyticSolution(A,E,PlotCoordinates.data) - u_predicted).data/torch.linalg.vector_norm(AnalyticSolution(A,E,PlotCoordinates.data)).data
    l2_loss_grad = torch.linalg.vector_norm(AnalyticGradientSolution(A,E,PlotCoordinates.data) - du_dx).data/torch.linalg.vector_norm(AnalyticGradientSolution(A,E,PlotCoordinates.data)).data

    return l2_loss, l2_loss_grad