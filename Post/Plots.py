
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
import numpy as np


def PlotSolution_Coordinates_Analytical(A,E,InitialCoordinates,Coordinates,TrialCoordinates,AnalyticSolution,model,name):
    pred = model(TrialCoordinates)[0]

    #plt.plot(InitialCoordinates,[coord*0 for coord in InitialCoordinates],'+k', markersize=2, label = 'Initial Nodes')
    plt.scatter(InitialCoordinates,[coord*0 for coord in InitialCoordinates], s=6, color="pink", alpha=0.5, label = 'Initial Nodes')
    plt.plot(Coordinates,[coord*0 for coord in Coordinates],'.k', markersize=2, label = 'Mesh Nodes')
    plt.plot(TrialCoordinates.data,AnalyticSolution(A,E,TrialCoordinates.data), label = 'Ground Truth')
    plt.plot(TrialCoordinates.data,pred.data,'--', label = 'HiDeNN')
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
    pred = model(TrialCoordinates)[0]

    plt.scatter(InitialCoordinates,[coord*0 for coord in InitialCoordinates], s=6, color="pink", alpha=0.5, label = 'Initial Nodes')
    plt.plot(Coordinates,[coord*0 for coord in Coordinates],'.k', markersize=2, label = 'Mesh Nodes')
    plt.plot(TrialCoordinates.data,AnalyticGradientSolution(A,E,TrialCoordinates.data), label = 'Ground Truth')
    plt.plot(TrialCoordinates.data,Derivative(pred).data,'--', label = 'HiDeNN')
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
    plt.ylabel(r'$\Vert \underline{u}_{ex} - \underline{u}_{NN} \Vert^2$')
    ax2 = plt.gca().twinx()
    ax2.semilogy(error3,color='#F39C12')
    ax2.set_ylabel(r'Lifted $J\left(\underline{u}\left(\underline{x}\right)\right)$')
    plt.savefig('Results/'+name+'.pdf', transparent=True) 
     
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

