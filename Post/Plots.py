
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
    plt.plot(InitialCoordinates,[coord*0 for coord in InitialCoordinates],'+k', markersize=2, label = 'Initial Nodes')
    plt.plot(Coordinates,[coord*0 for coord in Coordinates],'.k', markersize=2, label = 'Mesh Nodes')
    plt.plot(TrialCoordinates.data,AnalyticSolution(A,E,TrialCoordinates.data), label = 'Ground Truth')
    plt.plot(TrialCoordinates.data,model(TrialCoordinates).data,'--', label = 'HiDeNN')
    plt.xlabel(r'$\underline{x}$ [m]')
    plt.ylabel(r'$\underline{u}\left(\underline{x}\right)$')
    plt.legend(loc="upper left")
    # plt.title('Displacement')
    plt.savefig('Results/'+name+'.pdf', transparent=True)  
    plt.show()
    plt.clf()

def PlotGradSolution_Coordinates_Analytical(A,E,InitialCoordinates,Coordinates,TrialCoordinates,AnalyticGradientSolution,model,Derivative,name):
    # Plots the gradient & compare to reference
    plt.plot(InitialCoordinates,[coord*0 for coord in InitialCoordinates],'+k', markersize=2, label = 'Initial Nodes')
    plt.plot(Coordinates,[coord*0 for coord in Coordinates],'.k', markersize=2, label = 'Mesh Nodes')
    plt.plot(TrialCoordinates.data,AnalyticGradientSolution(A,E,TrialCoordinates.data), label = 'Ground Truth')
    plt.plot(TrialCoordinates.data,Derivative(model(TrialCoordinates),TrialCoordinates).data,'--', label = 'HiDeNN')
    plt.xlabel(r'$\underline{x}$ [m]')
    plt.ylabel(r'$\frac{d\underline{u}}{dx}\left(\underline{x}\right)$')
    plt.legend(loc="upper left")
    # plt.title('Displacement first derivative')
    plt.savefig('Results/'+name+'.pdf', transparent=True)  
    plt.show()
    plt.clf()

def PlotEnergyLoss(error,zoom,name):
    """Plots the error from the index given by zoom """
    plt.plot(error[zoom:])
    plt.xlabel(r'epochs')
    plt.ylabel(r'$J\left(\underline{u}\left(\underline{x}\right)\right)$')
    plt.savefig('Results/'+name+'.pdf', transparent=True)  
    plt.show()

def PlotTrajectories(Coord_trajectories,name):
    """Plots the trajectories of the coordinates during training"""
    plt.plot(Coord_trajectories)
    plt.xlabel(r'epochs')
    plt.ylabel(r'$x_i\left(\underline{x}\right)$')
    plt.savefig('Results/'+name+'.pdf', transparent=True)  
    plt.show()

def Plot_Compare_Loss2l2norm(error,error2,name):
    # Lift to be able to use semilogy
    error3 = error-np.min(error)
    plt.semilogy(error2)
    plt.ylabel(r'$\Vert \underline{u}_{ex} - \underline{u}_{NN} \Vert^2$')
    ax2 = plt.gca().twinx()
    ax2.semilogy(error3,color='#F39C12')
    ax2.set_ylabel(r'Lifted $J\left(\underline{u}\left(\underline{x}\right)\right)$')
    plt.savefig('Results/'+name+'.pdf', transparent=True)  
