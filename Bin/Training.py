import numpy as numpy
import torch
import Post.Plots as Pplot
import copy

from Bin.PDE_Library import RHS, PotentialEnergy, \
    PotentialEnergyVectorised, AlternativePotentialEnergy, \
        Derivative, AnalyticGradientSolution, AnalyticSolution

def plot_everything(A,E,InitialCoordinates,Coordinates,
                                            TrialCoordinates,AnalyticSolution,MeshBeam,Coord_trajectories, error, error2):
    # Tests on trained data and compare to reference
    Pplot.PlotSolution_Coordinates_Analytical(A,E,InitialCoordinates,Coordinates,
                                            TrialCoordinates,AnalyticSolution,MeshBeam,
                                            'Solution_displacement')
    # Plots the gradient & compare to reference
    Pplot.PlotGradSolution_Coordinates_Analytical(A,E,InitialCoordinates,Coordinates,
                                                TrialCoordinates,AnalyticGradientSolution,
                                                MeshBeam,Derivative,'Solution_gradients')
    # Plots trajectories of the coordinates while training
    Pplot.PlotTrajectories(Coord_trajectories,'Trajectories')
    Pplot.Plot_Compare_Loss2l2norm(error,error2,'Loss_Comaprison')
    Pplot.Plot_ShapeFuctions(TrialCoordinates.detach(), MeshBeam, InitialCoordinates, True)

def FilterTrainingData(MeshBeam, TestData):

    TestData = numpy.array(TestData.detach())

    NodeCoordinates = [MeshBeam.coordinates[i].data.item() for i in range(len(MeshBeam.coordinates))]
    idx = numpy.where( numpy.isclose(NodeCoordinates,TestData))
    #print("idx = ", idx)

    #print([NodeCoordinates[i] for i in idx[1]])
    #print([TestData[i] for i in idx[0]])

    for i in idx[0]:
        TestData[i][0] = TestData[i][0] + (TestData[i+1][0] - TestData[i][0])/10

    #print([TestData[i] for i in idx[0]])

    return torch.tensor(TestData, dtype=torch.float64, requires_grad=True)

def Test_GenerateShapeFunctions(MeshBeam, TrialCoordinates):
    
    InitialCoordinates = [MeshBeam.coordinates[i].data.item() for i in range(len(MeshBeam.coordinates))]

    pred, ShapeFunctions = MeshBeam(TrialCoordinates)
    Pplot.Plot_ShapeFuctions(TrialCoordinates.detach(), MeshBeam, InitialCoordinates, False)

def Training_InitialStage(MeshBeam, A, E, L, n_elem, TrialCoordinates, optimizer, n_epochs, BoolCompareNorms, MSE):

    # Store the initial coordinates before training (could be merged with Coord_trajectories)
    InitialCoordinates = [MeshBeam.coordinates[i].data.item() for i in range(len(MeshBeam.coordinates))]
    error = []              # Stores the loss
    error2 = []             # Stores the L2 error compared to the analytical solution

    Coord_trajectories = [] # Stores the trajectories of the coordinates while training

    stagnancy_counter = 0
    epoch = 0
    loss_old = 1.0e3
    loss_min = 1.0e3
    loss_counter = 0

    coord_min_loss = [MeshBeam.coordinates[i].data.item() for i in range(len(MeshBeam.coordinates))]
    weights_min_loss = copy.deepcopy(MeshBeam.InterpoLayer_uu.weight.data.detach())

    while epoch<n_epochs and stagnancy_counter < 50 and loss_counter<1000:

        #TrialCoordinates = FilterTrainingData(MeshBeam, TrialCoordinates)

        coord_old = [MeshBeam.coordinates[i].data.item() for i in range(len(MeshBeam.coordinates))]
        weights_old = copy.deepcopy(MeshBeam.InterpoLayer_uu.weight.data.detach())

        # predict = forward pass with our model
        u_predicted, _ = MeshBeam(TrialCoordinates) 
        # loss (several ways to compute the energy loss)
        l = PotentialEnergyVectorised(A,E,u_predicted,TrialCoordinates,RHS(TrialCoordinates))

        # Mesh regularisation term
        # Compute the ratio of the smallest jacobian and the largest jacobian
        #Jacobians = [MeshBeam.coordinates[i]-MeshBeam.coordinates[i-1] for i in range(1,len(MeshBeam.coordinates))]
        #Jacobians = torch.stack(Jacobians)
        #Ratio = torch.max(Jacobians)/torch.min(Jacobians)
        # Add the ratio to the loss
        #l+=MeshBeam.alpha*(Ratio-1)

        loss_current = l.item()
        loss_decrease = (loss_old - loss_current)/numpy.abs(loss_old)
        loss_old = loss_current

        # check for new minimal loss - Update the state for revert
        if loss_min > loss_current:
            loss_min = loss_current
            coord_min_loss = [MeshBeam.coordinates[i].data.item() for i in range(len(MeshBeam.coordinates))]
            weights_min_loss = copy.deepcopy(MeshBeam.InterpoLayer_uu.weight.data.detach())

            loss_counter = 0
        else:
            loss_counter = loss_counter + 1

        # calculate gradients = backward pass
        l.backward()
        # update weights
        optimizer.step()
        #scheduler.step(l)

        # zero the gradients after updating
        optimizer.zero_grad()

        # Chock colission - Revert if needed
        coord_new = [MeshBeam.coordinates[i].data.item() for i in range(len(MeshBeam.coordinates))]
        coord_dif = numpy.array([x - coord_new[i - 1] for i, x in enumerate(coord_new) if i > 0])
        if numpy.all(coord_dif > ((L/n_elem)/10)) == False:
            for j in range(coord_dif.shape[0]):
                if coord_dif[j] < (L/n_elem)/10:

                    MeshBeam.coordinates[j].data = torch.Tensor([[coord_old[j]]])
                    MeshBeam.coordinates[j+1].data = torch.Tensor([[coord_old[j+1]]])


        with torch.no_grad():
            # Stores the loss
            error.append(l.item())
            # Stores the coordinates trajectories
            Coordinates_i = [MeshBeam.coordinates[i].data.item() for i in range(len(MeshBeam.coordinates))]
            Coord_trajectories.append(Coordinates_i)

            if BoolCompareNorms:
                # Copute and store the L2 error w.r.t. the analytical solution
                error2.append(MSE(AnalyticSolution(A,E,TrialCoordinates.data),u_predicted).data)

        if loss_decrease >= 0 and loss_decrease < 1.0e-7:
            stagnancy_counter = stagnancy_counter +1
        else:
            stagnancy_counter = 0

        
        if (epoch+1) % 200 == 0:
            print('epoch ', epoch+1, ' loss = ', numpy.format_float_scientific( l.item(), precision=4))
            print(" loss decrease = ",  numpy.format_float_scientific( loss_decrease, precision=4))

            plot_everything(A,E,InitialCoordinates,Coordinates_i,
                                            TrialCoordinates,AnalyticSolution,MeshBeam,Coord_trajectories,error, error2)                           
        
        epoch = epoch+1

    '''
    # Final loss evaluation - Revert to minimal-loss state if needed
    if loss_min < loss_current:
        print("Revert")
        for j in range(len(coord_min_loss)):
            MeshBeam.coordinates[j].data = torch.Tensor([[coord_min_loss[j]]])
        MeshBeam.InterpoLayer_uu.weight.data = torch.Tensor(weights_min_loss)
        print("minimal loss = ", loss_min)
        u_predicted, _ = MeshBeam(TrialCoordinates) 
        l = PotentialEnergyVectorised(A,E,u_predicted,TrialCoordinates,RHS(TrialCoordinates))
        print("loss after revert = ", l.item())

        with torch.no_grad():
            # Stores the loss
            error.append(l.item())
            # Stores the coordinates trajectories
            Coordinates_i = [MeshBeam.coordinates[i].data.item() for i in range(len(MeshBeam.coordinates))]
            Coord_trajectories.append(Coordinates_i)
            if BoolCompareNorms:
                # Copute and store the L2 error w.r.t. the analytical solution
                error2.append(MSE(AnalyticSolution(A,E,TrialCoordinates.data),u_predicted).data)

    plot_everything(A,E,InitialCoordinates,Coordinates_i,
                                                TrialCoordinates,AnalyticSolution,MeshBeam,Coord_trajectories,error, error2)
    '''

    return error, error2, InitialCoordinates, Coord_trajectories


def Training_FinalStageLBFGS(MeshBeam, A, E, L, n_elem, InitialCoordinates, TrialCoordinates, n_epochs, BoolCompareNorms, MSE, error=[], error2 =[],Coord_trajectories=[]):
    optim = torch.optim.LBFGS(MeshBeam.parameters(),
                    #history_size=5, 
                    #max_iter=15, 
                    #tolerance_grad = 1.0e-9,
                    line_search_fn="strong_wolfe")

    loss_old = error[-1]
    epoch = 0
    stagnancy_counter = 0

    while epoch<n_epochs and stagnancy_counter < 3:

        def closure():
            optim.zero_grad()
            u_predicted, _ = MeshBeam(TrialCoordinates) 
            l = PotentialEnergyVectorised(A,E,u_predicted,TrialCoordinates,RHS(TrialCoordinates))
            l.backward()
            return l

        optim.step(closure)
        l = closure()

        with torch.no_grad():
            # Stores the loss
            error.append(l.item())
            # Stores the coordinates trajectories
            Coordinates_i = [MeshBeam.coordinates[i].data.item() for i in range(len(MeshBeam.coordinates))]
            Coord_trajectories.append(Coordinates_i)

            if BoolCompareNorms:
                # Copute and store the L2 error w.r.t. the analytical solution
                u_predicted, _ = MeshBeam(TrialCoordinates) 
                error2.append(MSE(AnalyticSolution(A,E,TrialCoordinates.data),u_predicted).data)

        loss_current = l.item()
        loss_decrease = (loss_old - loss_current)/numpy.abs(loss_old)

        if loss_decrease >= 0 and loss_decrease < 1.0e-8:
            stagnancy_counter = stagnancy_counter +1
        else:
            stagnancy_counter = 0

        loss_old = loss_current

        if (epoch+1) % 1 == 0:
            print('epoch ', epoch+1, ' loss = ', numpy.format_float_scientific( l.item(), precision=4))
        epoch = epoch+1

    plot_everything(A,E,InitialCoordinates,Coordinates_i,
                                                TrialCoordinates,AnalyticSolution,MeshBeam,Coord_trajectories,error, error2)