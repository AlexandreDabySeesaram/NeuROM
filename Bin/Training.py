import numpy as numpy
import torch
import Post.Plots as Pplot
import copy
import time
import random 

from Bin.PDE_Library import RHS, PotentialEnergy, \
    PotentialEnergyVectorised, AlternativePotentialEnergy, \
        Derivative, AnalyticGradientSolution, AnalyticSolution,\
            PotentialEnergyVectorisedParametric,AnalyticParametricSolution, \
                PotentialEnergyVectorisedBiParametric

def plot_everything(A,E,InitialCoordinates,Coordinates,
                                            TrialCoordinates,AnalyticSolution,BeamModel,Coord_trajectories, error, error2):
    Pplot.PlotTrajectories(Coord_trajectories,'Trajectories')

    # Tests on trained data and compare to reference
    Pplot.PlotSolution_Coordinates_Analytical(A,E,InitialCoordinates,Coordinates,
                                            TrialCoordinates,AnalyticSolution,BeamModel,
                                            'Solution_displacement')
    # Plots the gradient & compare to reference
    Pplot.PlotGradSolution_Coordinates_Analytical(A,E,InitialCoordinates,Coordinates,
                                                TrialCoordinates,AnalyticGradientSolution,
                                                BeamModel,Derivative,'Solution_gradients')

    Pplot.Plot_Compare_Loss2l2norm(error,error2,'Loss_Comaprison')


def Collision_Check(MeshBeam, coord_old, proximity_limit):
    # Chock colission -> Revert if needed
    coord_new = [MeshBeam.coordinates[i].data.item() for i in range(len(MeshBeam.coordinates))]
    coord_dif = numpy.array([x - coord_new[i - 1] for i, x in enumerate(coord_new) if i > 0])
    if numpy.all(coord_dif > proximity_limit) == False:
        for j in range(coord_dif.shape[0]):
            if coord_dif[j] < proximity_limit:
                MeshBeam.coordinates[j].data = torch.Tensor([[coord_old[j]]])
                MeshBeam.coordinates[j+1].data = torch.Tensor([[coord_old[j+1]]])

def RandomSign():
    return 1 if random.random() < 0.5 else -1


def FilterTrainingData(BeamModel, TestData):
    ### Filter training data in order to avoid collision of training point and mesh coordinate (ie derivative being automatically zero)
    TestData = numpy.array(TestData.detach())

    NodeCoordinates = [BeamModel.coordinates[i].data.item() for i in range(len(BeamModel.coordinates))]
    idx = numpy.where( numpy.isclose(NodeCoordinates,TestData, rtol=0, atol=1.0e-5))

    # if len(idx[0])>0:
    #     print("nodes = ", [NodeCoordinates[j] for j in idx[1]])
    #     print("   ", [TestData[i].item() for i in idx[0]])

    for i in idx[0]:

        if i ==0:
            TestData[i][0] = TestData[i][0] + min(5.0e-5, 0.1* numpy.min([TestData[n]-TestData[n-1] for n in range(1,len(TestData))]))
        elif i == TestData.shape[0]-1:
            TestData[i][0] = TestData[i][0] - min(5.0e-5, 0.1* numpy.min([TestData[n]-TestData[n-1] for n in range(1,len(TestData))]))
        else:
            TestData[i][0] = TestData[i][0] + RandomSign()* min(5.0e-5, 0.1* numpy.min([TestData[n]-TestData[n-1] for n in range(1,len(TestData))]))
    
    # if len(idx[0])>0:
    #     print("   ", [TestData[i].item() for i in idx[0]])
    #     print("________________________")

    return torch.tensor(TestData, dtype=torch.float32, requires_grad=True)




def Test_GenerateShapeFunctions(BeamModel, TrialCoordinates):
    ### To be used only in testing phase. 
    ### In MeshNN(nn.Module), set:
    ###     return self.SumLayer(u), recomposed_vector_u
    
    InitialCoordinates = [BeamModel.coordinates[i].data.item() for i in range(len(BeamModel.coordinates))]

    pred, ShapeFunctions = BeamModel(TrialCoordinates)
    Pplot.Plot_ShapeFuctions(TrialCoordinates.detach(), BeamModel, InitialCoordinates, False)


def Training_InitialStage(BeamModel, A, E, L, TrialCoordinates, optimizer, n_epochs, BoolCompareNorms, MSE, BoolFilterTrainingData):

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

        if FilterTrainingData:
            TrialCoordinates = FilterTrainingData(BeamModel, TrialCoordinates)

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
                 
            torch.save(BeamModel.state_dict(),"Results/Net_u.pt")

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

        Collision_Check(BeamModel, coord_old, 1.0e-6)


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
        BeamModel.load_state_dict(torch.load("Results/Net_u.pt"))

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

    plot_everything(A,E,InitialCoordinates,Coordinates_i,
                                                TrialCoordinates,AnalyticSolution,BeamModel,Coord_trajectories,error, error2)

    return error, error2, InitialCoordinates, Coord_trajectories, BeamModel


def Training_FinalStageLBFGS(BeamModel, A, E, L, InitialCoordinates, TrialCoordinates, n_epochs, BoolCompareNorms, MSE, BoolFilterTrainingData, error=[], error2 =[],Coord_trajectories=[]):
    optim = torch.optim.LBFGS(BeamModel.parameters(),
                    #history_size=5, 
                    #max_iter=15, 
                    #tolerance_grad = 1.0e-9,
                    line_search_fn="strong_wolfe")

    loss_old = error[-1]
    epoch = 0
    stagnancy_counter = 0

    while epoch<n_epochs and stagnancy_counter < 5:

        if BoolFilterTrainingData:
                    TrialCoordinates = FilterTrainingData(BeamModel, TrialCoordinates)

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
        loss_old = loss_current

        if loss_decrease >= 0 and loss_decrease < 1.0e-7:
            stagnancy_counter = stagnancy_counter +1
        else:
            stagnancy_counter = 0

        if (epoch+1) % 5 == 0:
            print('* epoch ', epoch+1, ' loss = ', numpy.format_float_scientific( l.item(), precision=4))

        epoch = epoch+1

    plot_everything(A,E,InitialCoordinates,Coordinates_i,
                                                TrialCoordinates,AnalyticSolution,BeamModel,Coord_trajectories,error, error2)

    print("*************** END OF SECOND STAGE ***************\n")
    print(f'* Final training loss: {numpy.format_float_scientific( error[-1], precision=4)}')
    print(f'* Final l2 loss : {numpy.format_float_scientific( error2[-1], precision=4)}')


def Training_NeuROM(model, A, L, TrialCoordinates,E_trial, optimizer, n_epochs,BiPara):
    # Initialise vector of loss values
    Loss_vect = []
    # Initialise vector of L2 error
    L2_error = []
    # BCs used for the analytical comparison 
    u0 = model.Space_modes[0].u_0
    uL = model.Space_modes[0].u_L
    print("**************** START TRAINING ***************\n")
    time_start = time.time()
    epoch = 0
    loss_counter = 0
    save_time = 0
    loss_decrease_c = 1e-4 # Criterion of stagnation for the loss
    stagnancy_counter = 0
    while epoch<n_epochs and loss_counter<500 and stagnancy_counter<100:
        # Compute loss
        if not BiPara:
            loss = PotentialEnergyVectorisedParametric(model,A,E_trial,model(TrialCoordinates,E_trial),TrialCoordinates,RHS(TrialCoordinates))
        else:
            loss = PotentialEnergyVectorisedBiParametric(model,A,E_trial,model(TrialCoordinates,E_trial),TrialCoordinates,RHS(TrialCoordinates))
        loss_current = loss.item()
         # check for new minimal loss - Update the state for revert
        if epoch >1:

            loss_decrease = (loss_old - loss_current)/numpy.abs(loss_old)
            loss_old = loss_current
            if loss_decrease >= 0 and loss_decrease < loss_decrease_c:
                stagnancy_counter = stagnancy_counter +1
            else:
                stagnancy_counter = 0

            if loss_min > loss_current:
                save_start = time.time()
                with torch.no_grad():
                    loss_min = loss_current
                    # torch.save(model.state_dict(),"Results/Current_best")
                    Current_best = copy.deepcopy(model.state_dict()) # Store in variable instead of writing file
                    save_stop = time.time()
                    save_time+=(save_stop-save_start)
                    loss_counter = 0
            else:
                loss_counter += 1
                
        else:
            loss_min = loss_current + 1 
            loss_old = loss_current

        loss.backward()
        # update weights
        optimizer.step()
        # zero the gradients after updating
        optimizer.zero_grad()
        with torch.no_grad():
            epoch+=1
            Loss_vect.append(loss.item())
            numel_E = E_trial[0].shape[0]
            if not BiPara:
                L2_error.append((torch.norm(torch.sum(AnalyticParametricSolution(A,E_trial,TrialCoordinates.data,u0,uL)-model(TrialCoordinates,E_trial),dim=1)/numel_E).data)/(torch.norm(torch.sum(AnalyticParametricSolution(A,E_trial,TrialCoordinates.data,u0,uL),dim=1)/numel_E)))
        if (epoch+1) % 100 == 0:
            if not BiPara:
                print(f'epoch {epoch+1} loss = {numpy.format_float_scientific(loss.item(), precision=4)} error = {numpy.format_float_scientific(100*L2_error[-1], precision=4)}%')
            else:
                print(f'epoch {epoch+1} loss = {numpy.format_float_scientific(loss.item(), precision=4)}')

    time_stop = time.time()
    # print("*************** END OF TRAINING ***************\n")
    print("*************** END FIRST PHASE ***************\n")
    print(f'* Training time: {time_stop-time_start}s')
    print(f'* Saving time: {save_time}s')
    # Final loss evaluation - Revert to minimal-loss state if needed
    if loss_min < loss_current:
        print("*************** REVERT TO BEST  ***************\n")
        # model.load_state_dict(torch.load("Results/Current_best"))
        model.load_state_dict(Current_best) # Load from variable instead of written file
        print("* Minimal loss = ", loss_min)

    return Loss_vect, L2_error, (time_stop-time_start)
    

def Training_NeuROM_FinalStageLBFGS(model, A, L, TrialCoordinates,E_trial, optimizer, n_epochs, max_stagnation,Loss_vect,L2_error,training_time,BiPara):
    optim = torch.optim.LBFGS(model.parameters(),
                    #history_size=5, 
                    #max_iter=15, 
                    #tolerance_grad = 1.0e-9,
                    line_search_fn="strong_wolfe")
    epoch = 0
    stagnancy_counter = 0
    loss_old = Loss_vect[-1]
    # BCs used for the analytical comparison 
    u0 = model.Space_modes[0].u_0
    uL = model.Space_modes[0].u_L
    print("************** START SECOND PAHSE *************\n")
    time_start = time.time()
    while  epoch<n_epochs and stagnancy_counter < 5:
        # Compute loss

        def closure():
            optim.zero_grad()
            if not BiPara:
                loss = PotentialEnergyVectorisedParametric(model,A,E_trial,model(TrialCoordinates,E_trial),TrialCoordinates,RHS(TrialCoordinates))
            else:
                loss = PotentialEnergyVectorisedBiParametric(model,A,E_trial,model(TrialCoordinates,E_trial),TrialCoordinates,RHS(TrialCoordinates))
            loss.backward()
            return loss
        optim.step(closure)
        loss = closure()

        loss_current  = loss.item()
        loss_decrease = (loss_old - loss_current)/numpy.abs(loss_old)
        loss_old = loss_current
        if loss_decrease >= 0 and loss_decrease < 1.0e-7:
            stagnancy_counter = stagnancy_counter +1
        else:
            stagnancy_counter = 0

        with torch.no_grad():
            epoch+=1
            Loss_vect.append(loss.item())
            numel_E = E_trial[0].shape[0]
            if not BiPara:
                L2_error.append((torch.norm(torch.sum(AnalyticParametricSolution(A,E_trial,TrialCoordinates.data,u0,uL)-model(TrialCoordinates,E_trial),dim=1)/numel_E).data)/(torch.norm(torch.sum(AnalyticParametricSolution(A,E_trial,TrialCoordinates.data,u0,uL),dim=1)/numel_E)))
        if (epoch+1) % 5 == 0:
            if not BiPara:
                print(f'epoch {epoch+1} loss = {numpy.format_float_scientific(loss.item(), precision=4)} error = {numpy.format_float_scientific(100*L2_error[-1], precision=4)}%')
            else:
                print(f'epoch {epoch+1} loss = {numpy.format_float_scientific(loss.item(), precision=4)}')

    time_stop = time.time()
    print("*************** END OF TRAINING ***************\n")
    print(f'* Training time: {training_time+time_stop-time_start}s')

    return Loss_vect, L2_error

