import numpy as numpy
import torch
import Post.Plots as Pplot
import copy
import time 

from Bin.PDE_Library import RHS, PotentialEnergy, \
    PotentialEnergyVectorised, AlternativePotentialEnergy, \
        Derivative, AnalyticGradientSolution, AnalyticSolution,\
            PotentialEnergyVectorisedParametric

def plot_everything(A,E,InitialCoordinates,Coordinates,
                                            TrialCoordinates,AnalyticSolution,BeamModel,Coord_trajectories, error, error2):
    # Tests on trained data and compare to reference
    Pplot.PlotSolution_Coordinates_Analytical(A,E,InitialCoordinates,Coordinates,
                                            TrialCoordinates,AnalyticSolution,BeamModel,
                                            'Solution_displacement')
    # Plots the gradient & compare to reference
    Pplot.PlotGradSolution_Coordinates_Analytical(A,E,InitialCoordinates,Coordinates,
                                                TrialCoordinates,AnalyticGradientSolution,
                                                BeamModel,Derivative,'Solution_gradients')
    # Plots trajectories of the coordinates while training
    Pplot.PlotTrajectories(Coord_trajectories,'Trajectories')
    Pplot.Plot_Compare_Loss2l2norm(error,error2,'Loss_Comaprison')
    # Pplot.Plot_ShapeFuctions(TrialCoordinates.detach(), BeamModel, InitialCoordinates, True)

def FilterTrainingData(BeamModel, TestData):

    TestData = numpy.array(TestData.detach())

    NodeCoordinates = [BeamModel.coordinates[i].data.item() for i in range(len(BeamModel.coordinates))]
    idx = numpy.where( numpy.isclose(NodeCoordinates,TestData))
    #print("idx = ", idx)

    #print([NodeCoordinates[i] for i in idx[1]])
    #print([TestData[i] for i in idx[0]])

    for i in idx[0]:
        TestData[i][0] = TestData[i][0] + (TestData[i+1][0] - TestData[i][0])/10

    #print([TestData[i] for i in idx[0]])

    return torch.tensor(TestData, dtype=torch.float64, requires_grad=True)

def Test_GenerateShapeFunctions(BeamModel, TrialCoordinates):
    
    InitialCoordinates = [BeamModel.coordinates[i].data.item() for i in range(len(BeamModel.coordinates))]

    pred, ShapeFunctions = BeamModel(TrialCoordinates)
    Pplot.Plot_ShapeFuctions(TrialCoordinates.detach(), BeamModel, InitialCoordinates, False)

def Training_InitialStage(BeamModel, A, E, L, n_elem, TrialCoordinates, optimizer, n_epochs, BoolCompareNorms, MSE):

    # Store the initial coordinates before training (could be merged with Coord_trajectories)
    InitialCoordinates = [BeamModel.coordinates[i].data.item() for i in range(len(BeamModel.coordinates))]
    error = []              # Stores the loss
    error2 = []             # Stores the L2 error compared to the analytical solution

    Coord_trajectories = [] # Stores the trajectories of the coordinates while training

    stagnancy_counter = 0
    epoch = 0
    loss_old = 1.0e3
    loss_min = 1.0e3
    loss_counter = 0

    coord_min_loss = [BeamModel.coordinates[i].data.item() for i in range(len(BeamModel.coordinates))]
    weights_min_loss = copy.deepcopy(BeamModel.InterpoLayer_uu.weight.data.detach())

    print("**************** START TRAINING ***************\n")
    start_train_time = time.time()

    evaluation_time = 0
    loss_time = 0
    optimizer_time = 0
    backward_time = 0

    while epoch<n_epochs and stagnancy_counter < 50 and loss_counter<1000:

        #TrialCoordinates = FilterTrainingData(BeamModel, TrialCoordinates)

        coord_old = [BeamModel.coordinates[i].data.item() for i in range(len(BeamModel.coordinates))]
        weights_old = copy.deepcopy(BeamModel.InterpoLayer_uu.weight.data.detach())

        # predict = forward pass with our model
        start_time = time.time()
        u_predicted = BeamModel(TrialCoordinates) 
        evaluation_time += time.time() - start_time
        start_time = time.time()
        # loss 
        l = PotentialEnergyVectorised(A,E,u_predicted,TrialCoordinates,RHS(TrialCoordinates))
        loss_time += time.time() - start_time

        loss_current = l.item()
        loss_decrease = (loss_old - loss_current)/numpy.abs(loss_old)
        loss_old = loss_current

        # check for new minimal loss - Update the state for revert
        if loss_min > loss_current:
            loss_min = loss_current
            coord_min_loss = [BeamModel.coordinates[i].data.item() for i in range(len(BeamModel.coordinates))]
            weights_min_loss = copy.deepcopy(BeamModel.InterpoLayer_uu.weight.data.detach())

            loss_counter = 0
        else:
            loss_counter += 1

        # calculate gradients = backward pass
        start_time = time.time()
        l.backward()
        backward_time += time.time() - start_time
        # update weights
        start_time = time.time()
        optimizer.step()
        optimizer_time += time.time() - start_time
        #scheduler.step(l)

        # zero the gradients after updating
        optimizer.zero_grad()

        # Chock colission - Revert if needed
        # coord_new = [BeamModel.coordinates[i].data.item() for i in range(len(BeamModel.coordinates))]
        # coord_dif = numpy.array([x - coord_new[i - 1] for i, x in enumerate(coord_new) if i > 0])
        # if numpy.all(coord_dif > ((L/n_elem)/10)) == False:
        #     for j in range(coord_dif.shape[0]):
        #         if coord_dif[j] < (L/n_elem)/10:

        #             BeamModel.coordinates[j].data = torch.Tensor([[coord_old[j]]])
        #             BeamModel.coordinates[j+1].data = torch.Tensor([[coord_old[j+1]]])


        with torch.no_grad():
            # Stores the loss
            error.append(l.item())
            # Stores the coordinates trajectories
            Coordinates_i = [BeamModel.coordinates[i].data.item() \
                            for i in range(len(BeamModel.coordinates))]
            Coord_trajectories.append(Coordinates_i)

            if BoolCompareNorms:
                # Copute and store the L2 error w.r.t. the analytical solution
                error2.append(MSE(AnalyticSolution(A,E,TrialCoordinates.data),u_predicted).data)

        if loss_decrease >= 0 and loss_decrease < 1.0e-7:
            stagnancy_counter = stagnancy_counter +1
        else:
            stagnancy_counter = 0

        
        if (epoch+1) % 200 == 0:
            print('* epoch ', epoch+1, ' loss = ', numpy.format_float_scientific( l.item(), precision=4))
            print("* loss decrease = ",  numpy.format_float_scientific( loss_decrease, precision=4))

            # plot_everything(A,E,InitialCoordinates,Coordinates_i,
                                            # TrialCoordinates,AnalyticSolution,BeamModel,Coord_trajectories,error, error2)                           
        
        epoch = epoch+1
    stopt_train_time = time.time()
    print("*************** END OF TRAINING ***************\n")
    print(f'* Training time: {stopt_train_time-start_train_time}s\n\
* Evaluation time: {evaluation_time}s\n\
* Loss time: {loss_time}s\n\
* Backward time: {backward_time}s\n\
* Training time per epochs: {(stopt_train_time-start_train_time)/n_epochs}s\n\
* Optimiser time: {optimizer_time}s\n')

    
    # Final loss evaluation - Revert to minimal-loss state if needed
    if loss_min < loss_current:
        print("Revert")
        for j in range(len(coord_min_loss)):
            BeamModel.coordinates[j].data = torch.Tensor([[coord_min_loss[j]]])
        BeamModel.InterpoLayer_uu.weight.data = torch.Tensor(weights_min_loss)
        print("minimal loss = ", loss_min)
        u_predicted = BeamModel(TrialCoordinates) 
        l = PotentialEnergyVectorised(A,E,u_predicted,TrialCoordinates,RHS(TrialCoordinates))
        print("loss after revert = ", l.item())

        with torch.no_grad():
            # Stores the loss
            error.append(l.item())
            # Stores the coordinates trajectories
            Coordinates_i = [BeamModel.coordinates[i].data.item() for i in range(len(BeamModel.coordinates))]
            Coord_trajectories.append(Coordinates_i)
            if BoolCompareNorms:
                # Copute and store the L2 error w.r.t. the analytical solution
                error2.append(MSE(AnalyticSolution(A,E,TrialCoordinates.data),u_predicted).data)

    # plot_everything(A,E,InitialCoordinates,Coordinates_i,
                                                # TrialCoordinates,AnalyticSolution,BeamModel,Coord_trajectories,error, error2)
    

    return error, error2, InitialCoordinates, Coord_trajectories, BeamModel


def Training_FinalStageLBFGS(BeamModel, A, E, L, n_elem, InitialCoordinates, TrialCoordinates, n_epochs, BoolCompareNorms, MSE, error=[], error2 =[],Coord_trajectories=[]):
    optim = torch.optim.LBFGS(BeamModel.parameters(),
                    #history_size=5, 
                    #max_iter=15, 
                    #tolerance_grad = 1.0e-9,
                    line_search_fn="strong_wolfe")

    loss_old = error[-1]
    epoch = 0
    stagnancy_counter = 0

    while epoch<n_epochs and stagnancy_counter < 5:

        def closure():
            optim.zero_grad()
            u_predicted = BeamModel(TrialCoordinates) 
            l = PotentialEnergyVectorised(A,E,u_predicted,TrialCoordinates,RHS(TrialCoordinates))
            l.backward()
            return l

        optim.step(closure)
        l = closure()

        with torch.no_grad():
            # Stores the loss
            error.append(l.item())
            # Stores the coordinates trajectories
            Coordinates_i = [BeamModel.coordinates[i].data.item() for i in range(len(BeamModel.coordinates))]
            Coord_trajectories.append(Coordinates_i)

            if BoolCompareNorms:
                # Copute and store the L2 error w.r.t. the analytical solution
                u_predicted = BeamModel(TrialCoordinates) 
                error2.append(MSE(AnalyticSolution(A,E,TrialCoordinates.data),u_predicted).data)

        loss_current = l.item()
        loss_decrease = (loss_old - loss_current)/numpy.abs(loss_old)

        if loss_decrease >= 0 and loss_decrease < 1.0e-8:
            stagnancy_counter = stagnancy_counter +1
        else:
            stagnancy_counter = 0

        loss_old = loss_current

        if (epoch+1) % 1 == 0:
            print('* epoch ', epoch+1, ' loss = ', numpy.format_float_scientific( l.item(), precision=4))
        epoch = epoch+1

    plot_everything(A,E,InitialCoordinates,Coordinates_i,
                                                TrialCoordinates,AnalyticSolution,BeamModel,Coord_trajectories,error, error2)



def Training_NeuROM(model, A, L, TrialCoordinates,E_trial, optimizer, n_epochs, BoolCompareNorms, MSE):
    Loss_vect = []
    # time_start = time.time()
    for epoch in range(n_epochs):
        # loss_vect = torch.stack([PotentialEnergyVectorised(A,E,model(TrialCoordinates,E),TrialCoordinates,RHS(TrialCoordinates)) for E in E_trial])
        # loss = torch.sum(loss_vect)/E_trial.shape[0]

        loss = PotentialEnergyVectorisedParametric(model,A,E_trial,model(TrialCoordinates,E_trial),TrialCoordinates,RHS(TrialCoordinates))
        
        
        loss.backward()
        # update weights
        optimizer.step()
        # zero the gradients after updating
        optimizer.zero_grad()
        with torch.no_grad():
            Loss_vect.append(loss.item())
        if (epoch+1) % 100 == 0:
            print('epoch ', epoch+1, ' loss = ', loss.item())
            # if (epoch+1) % 1000 == 0:
                # time_end = time.time()
                # duration = time_end - time_start
                # print(f'* Last 1000 epochs took :{duration}s')
                # import matplotlib.pyplot as plt
                # plt.plot(E_trial.data,model.Para_modes[0](E_trial).data)
                # plt.show()
                # plt.clf()
                # plt.plot(TrialCoordinates.data,model.Space_modes[0](TrialCoordinates).data)
                # plt.show()
                # plt.clf()
                # time_start = time.time()
        # # if epoch == 3000:
        #     model.Freeze_Space()
        #     model.UnFreeze_Para()
    
    return Loss_vect
    

