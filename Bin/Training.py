import numpy as numpy
import torch
import Post.Plots as Pplot
import copy
import time
import torch
import random 

from Bin.PDE_Library import RHS, PotentialEnergy, \
    PotentialEnergyVectorised, AlternativePotentialEnergy, \
        Derivative, AnalyticGradientSolution, AnalyticSolution,\
            PotentialEnergyVectorisedParametric,AnalyticParametricSolution, \
                PotentialEnergyVectorisedBiParametric, MixedFormulation_Loss,\
                Mixed_2D_loss, Neumann_BC_rel, Constitutive_BC, GetRealCoord, Mixed_2D_loss_Displacement_based,\
                    InternalEnergy_2D, VolumeForcesEnergy_2D,InternalEnergy_2D_einsum


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

def plot_everything_mixed(A,E,InitialCoordinates_u, InitialCoordinates_du, Coordinates_u, Coordinates_du,
                                            TrialCoordinates,AnalyticSolution,BeamModel_u,BeamModel_du, \
                                            Coord_trajectories, error_pde, error_constit, error2):
    # Tests on trained data and compare to reference
    Pplot.PlotSolution_Coordinates_Analytical(A,E,InitialCoordinates_u,Coordinates_u,
                                            TrialCoordinates,AnalyticSolution,BeamModel_u,
                                            'Solution_displacement')
    # Plots the gradient & compare to reference

    Pplot.PlotGradSolution_Coordinates_Analytical(A,E,InitialCoordinates_u,Coordinates_u,
                                                TrialCoordinates,AnalyticGradientSolution,
                                                BeamModel_u,Derivative,'Solution_gradients_dudx')

    Pplot.PlotGradSolution_Coordinates_Force(A,E,InitialCoordinates_du,Coordinates_du,
                                                TrialCoordinates, RHS(TrialCoordinates),
                                                BeamModel_du,Derivative,'Solution_gradients_Force')

    Pplot.PlotSolution_Coordinates_Analytical(A,E,InitialCoordinates_du,Coordinates_du,
                                            TrialCoordinates,AnalyticGradientSolution,BeamModel_du,
                                            'Solution_gradients')

    # Plots trajectories of the coordinates while training
    Pplot.PlotTrajectories(Coord_trajectories,'Trajectories')

    Pplot.Plot_Compare_Loss2l2norm_Mixed(error_pde,error_constit, error2,'Loss_Comaprison')

def Plot_all_2D(Model_u, Model_du, IDs_u, IDs_du, PlotCoordinates, loss, n_train, stage):
    u_predicted = Model_u(PlotCoordinates, IDs_u) 
    du_predicted = Model_du(PlotCoordinates, IDs_du) 

    l_pde, l_compat, s11, s22, s12 =  Mixed_2D_loss(u_predicted[0,:], u_predicted[1,:],
                                                    du_predicted[0,:], du_predicted[1,:], du_predicted[2,:],
                                                    PlotCoordinates, lmbda = 1.25, mu = 1.0)
    l =  l_pde +l_compat
    print()
    print("     loss PDE = ", numpy.format_float_scientific(l_pde.item(),precision=4))
    print("     loss compatibility = ", numpy.format_float_scientific(l_compat.item(),precision=4))

    #Model_u.CheckBCValues()
    Pplot.Plot2Dresults(u_predicted, PlotCoordinates, "_u"+stage)
    #Pplot.Plot1DSection(u_predicted, n_train, 5*n_train, stage)
    Pplot.Plot2Dresults_Derivative(du_predicted, s11, s22, s12, PlotCoordinates, "_Stress" + stage)

    if len(loss)>0:
        Pplot.Plot2DLoss(loss)

    return l


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


def FilterBatchTrainingData(BeamModel, TestData):
    ### Filter training data in order to avoid collision of training point and mesh coordinate (ie derivative being automatically zero)
    TestData = numpy.sort(numpy.array(TestData.detach()), axis =0)

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


def FilterTrainingData(BeamModel, TestData):
    ### Filter training data in order to avoid collision of training point and mesh coordinate (ie derivative being automatically zero)
    TestData = numpy.array(TestData.detach())

    NodeCoordinates = [BeamModel.coordinates[i].data.item() for i in range(len(BeamModel.coordinates))]
    idx = numpy.where( numpy.isclose(NodeCoordinates,TestData, rtol=0, atol=1.0e-5))

    if len(idx[0])>0:
        print("nodes = ", [NodeCoordinates[j] for j in idx[1]])
        print("   ", [TestData[i].item() for i in idx[0]])

    for i in idx[0]:

        if i ==0:
            TestData[i][0] = TestData[i][0] + min(5.0e-5, 0.1* numpy.min([TestData[n]-TestData[n-1] for n in range(1,len(TestData))]))
        elif i == TestData.shape[0]-1:
            TestData[i][0] = TestData[i][0] - min(5.0e-5, 0.1* numpy.min([TestData[n]-TestData[n-1] for n in range(1,len(TestData))]))
        else:
            TestData[i][0] = TestData[i][0] + RandomSign()* min(5.0e-5, 0.1* numpy.min([TestData[n]-TestData[n-1] for n in range(1,len(TestData))]))
    
    if len(idx[0])>0:
        print("   ", [TestData[i].item() for i in idx[0]])
        print("________________________")

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

    analytical_norm = torch.norm(AnalyticSolution(A,E,TrialCoordinates.data)).data

    while epoch<n_epochs and stagnancy_counter < 50 and loss_counter<1000:

        if BoolFilterTrainingData:
            TrialCoordinates = FilterTrainingData(BeamModel, TrialCoordinates)

        coord_old = [BeamModel.coordinates[i].data.item() for i in range(len(BeamModel.coordinates))]
        # weights_old = copy.deepcopy(BeamModel.InterpoLayer_uu.weight.data.detach())

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
                error2.append(torch.norm(AnalyticSolution(A,E,TrialCoordinates.data) - u_predicted)/analytical_norm)

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
                error2.append(torch.norm(AnalyticSolution(A,E,TrialCoordinates.data)-u_predicted)/analytical_norm)

    plot_everything(A,E,InitialCoordinates,Coordinates_i,
                                                TrialCoordinates,AnalyticSolution,BeamModel,Coord_trajectories,error, error2)

    print(f'* Final training loss: {numpy.format_float_scientific( error[-1], precision=4)}')
    print(f'* Final l2 loss : {numpy.format_float_scientific( error2[-1], precision=4)}')

    return error, error2, InitialCoordinates, Coord_trajectories, BeamModel


def Training_FinalStageLBFGS(BeamModel, A, E, L, InitialCoordinates, TrialCoordinates, n_epochs, BoolCompareNorms, MSE, BoolFilterTrainingData, error=[], error2 =[],Coord_trajectories=[]):
    optim = torch.optim.LBFGS(BeamModel.parameters(),
                    #history_size=5, 
                    #max_iter=15, 
                    #tolerance_grad = 1.0e-9,
                    line_search_fn="strong_wolfe")
    print()
    print("*************** SECOND STAGE (LBFGS) ***************\n")
    loss_old = error[-1]
    epoch = 0
    stagnancy_counter = 0

    analytical_norm = torch.norm(AnalyticSolution(A,E,TrialCoordinates.data)).data

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
                error2.append(torch.norm(AnalyticSolution(A,E,TrialCoordinates.data)-u_predicted).data/analytical_norm)

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
    eval_time = 0
    back_time = 0
    update_time = 0
    loss_decrease_c = 1e-5 # Criterion of stagnation for the loss
    Add_mode_c = 1e-5 # Criterion of stagnation before adding a new mode
    FlagAddedMode_usefull = True    # Flag stating that the new mode did help speeding-up the convergence
    stagnancy_counter = 0
    local_stagnancy_counter = 0 # Stagnancy since last additoin of a mode
    FlagAddedMode = False # Flag activated whn a mode has been added
    Modes_vect = []
    Loss_decrease_vect = []
    Usefullness = 0
    while epoch<n_epochs and loss_counter<100:
        # Break if stagnation not solved by adding modes (hopefully that means convergence reached)
        if stagnancy_counter>5 and not FlagAddedMode_usefull:
            break 
        # Compute loss
        loss_time_start = time.time()
        if not BiPara:
            loss = PotentialEnergyVectorisedParametric(model,A,E_trial,model(TrialCoordinates,E_trial),TrialCoordinates,RHS(TrialCoordinates))
        else:
            loss = PotentialEnergyVectorisedBiParametric(model,A,E_trial,TrialCoordinates,RHS(TrialCoordinates))
        eval_time += time.time() - loss_time_start
        loss_current = loss.item()
         # check for new minimal loss - Update the state for revert
        if epoch >1:
            loss_decrease = (loss_old - loss_current)/numpy.abs(0.5*(loss_old + loss_current))
            Loss_decrease_vect.append(loss_decrease)
            loss_old = loss_current
            # if loss_decrease >= 0 and loss_decrease < loss_decrease_c:
            if numpy.abs(loss_decrease) < loss_decrease_c:
                stagnancy_counter = stagnancy_counter +1
            else:
                stagnancy_counter = 0
                if loss_decrease >= 0:
                    Usefullness+=1
                    if Usefullness>=15:
                        FlagAddedMode_usefull = True    # Flag stating that the new mode did help speeding-up the convergence


            if loss_min > loss_current:
                with torch.no_grad():
                    if (epoch+1) % 300 == 0:
                        save_start = time.time()
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

        backward_time_start = time.time()
        loss.backward()
        back_time += time.time() - backward_time_start
        # update weights
        update_time_start = time.time()
        optimizer.step()
        update_time += time.time() - update_time_start
        # zero the gradients after updating
        optimizer.zero_grad()
        Modes_vect.append(model.n_modes_truncated.detach().clone())
        if stagnancy_counter >5 and model.n_modes_truncated < model.n_modes and FlagAddedMode_usefull:
            model.AddMode()
            model.AddMode2Optimizer(optimizer)
            Addition_epoch_index = epoch
            FlagAddedMode = True
            FlagAddedMode_usefull = False    # Flag stating that the new mode did help speeding-up the convergence
            stagnancy_counter = 0
            Usefullness = 0
        if FlagAddedMode:
            if epoch == Addition_epoch_index+2:
                model.UnfreezeTruncated()
                stagnancy_counter = 0

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
                print(f'epoch {epoch+1} loss = {numpy.format_float_scientific(loss.item(), precision=4)} modes = {model.n_modes_truncated}')

    time_stop = time.time()
    # print("*************** END OF TRAINING ***************\n")
    print("*************** END FIRST PHASE ***************\n")
    print(f'* Training time: {time_stop-time_start}s')
    print(f'* Saving time: {save_time}s')
    print(f'* Evaluation time: {eval_time}s')
    print(f'* Backward time: {back_time}s')
    print(f'* Update time: {update_time}s')
    print(f'* Average epoch time: {(time_stop-time_start)/(epoch+1)}s')

    # Final loss evaluation - Revert to minimal-loss state if needed
    if loss_min < loss_current:
        print("*************** REVERT TO BEST  ***************\n")
        # model.load_state_dict(torch.load("Results/Current_best"))
        model.load_state_dict(Current_best) # Load from variable instead of written file
        print("* Minimal loss = ", loss_min)

    return Loss_vect, L2_error, (time_stop-time_start), Modes_vect, Loss_decrease_vect
    

def Training_NeuROM_FinalStageLBFGS(model, A, L, TrialCoordinates,E_trial, optimizer, n_epochs, max_stagnation,Loss_vect,L2_error,training_time,BiPara):
    optim = torch.optim.LBFGS([p for p in model.parameters() if p.requires_grad],
                    #history_size=5, 
                    #max_iter=15, 
                    #tolerance_grad = 1.0e-9,
                    line_search_fn="strong_wolfe")
    epoch = 0
    stagnancy_counter = 0
    # model.UnFreeze_Mesh()
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
                loss = PotentialEnergyVectorisedBiParametric(model,A,E_trial,TrialCoordinates,RHS(TrialCoordinates))
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

def Mixed_Training_InitialStage(BeamModel_u, BeamModel_du, A, E, L, CoordinatesBatchSet, PlotData, 
                                optimizer, n_epochs,
                                 BoolCompareNorms, MSE, BoolFilterTrainingData, w_pde, w_constit):

   # Store the initial coordinates before training (could be merged with Coord_trajectories)
    InitialCoordinates_u = [BeamModel_u.coordinates[i].data.item() for i in range(len(BeamModel_u.coordinates))]
    InitialCoordinates_du = [BeamModel_du.coordinates[i].data.item() for i in range(len(BeamModel_du.coordinates))]

    error_pde = []              # Stores the loss
    error_constit = []
    error2 = []             # Stores the L2 error compared to the analytical solution
    Coord_trajectories = [] # Stores the trajectories of the coordinates while training

    stagnancy_counter = 0
    epoch = 0
    loss_old = 1.0e3
    loss_min = 1.0e3
    loss_counter = 0

    analytical_norm = torch.norm(AnalyticSolution(A,E,PlotData.data))

    print("**************** START TRAINING ***************\n")
    start_train_time = time.time()

    evaluation_time = 0
    loss_time = 0
    optimizer_time = 0
    backward_time = 0

    while epoch<n_epochs and loss_counter<1000: #  stagnancy_counter < 50 and 

        for TrialCoordinates in CoordinatesBatchSet:
        #for i in range(1):

            #n_train_points = torch.randint(100,500,(1,))[0]
            #n_train_points = 500
            #TrialCoordinates = torch.tensor([[i] for i in torch.linspace(0,L,n_train_points)], dtype=torch.float64, requires_grad=True)

            if BoolFilterTrainingData: 
                TrialCoordinates = FilterBatchTrainingData(BeamModel_u, TrialCoordinates)
                TrialCoordinates = FilterBatchTrainingData(BeamModel_du, TrialCoordinates)

            coord_old_u = [BeamModel_u.coordinates[i].data.item() for i in range(len(BeamModel_u.coordinates))]
            coord_old_du = [BeamModel_du.coordinates[i].data.item() for i in range(len(BeamModel_du.coordinates))]

            # predict = forward pass 
            start_time = time.time()
            u_predicted = BeamModel_u(TrialCoordinates) 
            du_predicted = BeamModel_du(TrialCoordinates) 
            evaluation_time += time.time() - start_time
            
            start_time = time.time()
            # loss for weights update
            l_pde, l_constit  = MixedFormulation_Loss(A, E, u_predicted, du_predicted, TrialCoordinates, RHS(TrialCoordinates))
            l =  w_pde*l_pde + w_constit*l_constit
            loss_time += time.time() - start_time

            loss_current = l.item()
            loss_decrease = (loss_old - loss_current)/numpy.abs(loss_old)
            loss_old = loss_current
            
            #################################################

            # check for new minimal loss - Update the state for revert
            if loss_min > loss_current:
                loss_min = loss_current
                loss_counter = 0
            else:
                loss_counter = loss_counter + 1

            # calculate gradients = backward pass, using loss on one batch
            start_time = time.time()
            l.backward()
            backward_time += time.time() - start_time
            # update weights
            start_time = time.time()

            optimizer.step()

            optimizer_time += time.time() - start_time

            # zero the gradients after updating
            optimizer.zero_grad()

            Collision_Check(BeamModel_u, coord_old_u, 1.0e-6)
            Collision_Check(BeamModel_du, coord_old_du, 1.0e-6)

            with torch.no_grad():
                # Stores the loss
                error_pde.append(l_pde.item())
                error_constit.append(l_constit.item())

                # Stores the coordinates trajectories
                Coordinates_u_i = [BeamModel_u.coordinates[i].data.item() for i in range(len(BeamModel_u.coordinates))]
                Coordinates_du_i = [BeamModel_du.coordinates[i].data.item() for i in range(len(BeamModel_du.coordinates))]
                Coord_trajectories.append(Coordinates_u_i)

                if BoolCompareNorms:
                    # Copute and store the L2 error w.r.t. the analytical solution
                    error2.append(torch.norm(AnalyticSolution(A,E,TrialCoordinates.data) - u_predicted).data/analytical_norm)

            if loss_decrease < 1.0e-7:
                stagnancy_counter = stagnancy_counter +1
            else:
                stagnancy_counter = 0
                    
        if (epoch+1) % 200 == 0:
                print('* epoch ', epoch+1, ' loss = ', numpy.format_float_scientific( l.item(), precision=4))
                print('* loss PDE = ', numpy.format_float_scientific( l_pde.item(), precision=4))
                print('* loss constit = ', numpy.format_float_scientific( l_constit.item(), precision=4))
                #print("* loss decrease = ",  numpy.format_float_scientific( loss_decrease, precision=4))

                #plot_everything_mixed(A,E,InitialCoordinates_u, InitialCoordinates_du, Coordinates_u_i, Coordinates_du_i,
                #                            PlotData, AnalyticSolution, BeamModel_u, BeamModel_du, \
                #                            Coord_trajectories, error_pde, error_constit, error2)    

        epoch = epoch+1

    stopt_train_time = time.time()
    print("*************** END OF TRAINING ***************\n")
    print(f'* Training time: {stopt_train_time-start_train_time}s\n\
        * Evaluation time: {evaluation_time}s\n\
        * Loss time: {loss_time}s\n\
        * Backward time: {backward_time}s\n\
        * Training time per epochs: {(stopt_train_time-start_train_time)/n_epochs}s\n\
        * Optimiser time: {optimizer_time}s\n')

    u_predicted_1 = BeamModel_u(PlotData) 
    du_predicted_1 = BeamModel_du(PlotData) 
    l_pde_1, l_constit_1  = MixedFormulation_Loss(A, E, u_predicted_1, du_predicted_1, PlotData, RHS(PlotData))
    l1 =  l_pde_1 + l_constit_1

    with torch.no_grad():
        # Stores the loss
        error_pde.append(l_pde_1.item())
        error_constit.append(l_constit_1.item())        # Stores the coordinates trajectories
        Coordinates_u_i = [BeamModel_u.coordinates[i].data.item() for i in range(len(BeamModel_u.coordinates))]
        Coordinates_du_i = [BeamModel_du.coordinates[i].data.item() for i in range(len(BeamModel_du.coordinates))]

        Coord_trajectories.append(Coordinates_u_i)

        if BoolCompareNorms:
            # Copute and store the L2 error w.r.t. the analytical solution
            analytical_norm = torch.norm(AnalyticSolution(A,E,PlotData.data))
            error2.append(torch.norm(AnalyticSolution(A,E,PlotData.data) - u_predicted_1).data/analytical_norm)

    plot_everything_mixed(A,E,InitialCoordinates_u, InitialCoordinates_du, Coordinates_u_i, Coordinates_du_i,
                                            PlotData, AnalyticSolution, BeamModel_u, BeamModel_du, \
                                            Coord_trajectories,  error_pde, error_constit, error2)  
 
    print(f'* Final training loss: {numpy.format_float_scientific( error_pde[-1] + error_constit[-1], precision=4)}')
    print(f'* Final l2 loss : {numpy.format_float_scientific( error2[-1], precision=4)}')

    return error_pde, error_constit, error2, InitialCoordinates_u, InitialCoordinates_du, Coord_trajectories
    


def Training_FinalStageLBFGS_Mixed(BeamModel_u, BeamModel_du, A, E, L, InitialCoordinates_u, InitialCoordinates_du,
                                        TrialCoordinates, n_epochs, BoolCompareNorms, 
                                        MSE, BoolFilterTrainingData,
                                        error_pde, error_constit, error2, Coord_trajectories,
                                        w_pde, w_constit): 

    print()
    print("*************** SECOND STAGE (LBFGS) ***************\n")
    
    optim = torch.optim.LBFGS(list(BeamModel_u.parameters()) + list(BeamModel_du.parameters()),
                    history_size=5, 
                    max_iter=15, 
                    tolerance_grad = 1.0e-9,
                    line_search_fn="strong_wolfe")

    loss_old = error_pde[-1] + error_constit[-1]
    epoch = 0
    stagnancy_counter = 0

    analytical_norm = torch.norm(AnalyticSolution(A,E,TrialCoordinates.data)).data

    while epoch<n_epochs and stagnancy_counter < 5:

        if BoolFilterTrainingData:
            TrialCoordinates = FilterTrainingData(BeamModel_u, TrialCoordinates)
            TrialCoordinates = FilterTrainingData(BeamModel_u, TrialCoordinates)


        def closure():
            optim.zero_grad()
            u_predicted = BeamModel_u(TrialCoordinates) 
            du_predicted = BeamModel_du(TrialCoordinates) 
            l_pde, l_constit  = MixedFormulation_Loss(A, E, u_predicted, du_predicted, TrialCoordinates, RHS(TrialCoordinates))
            l =  w_pde*l_pde + w_constit*l_constit
            l.backward()
            return l_pde + l_constit

        optim.step(closure)
        l = closure()

        with torch.no_grad():
            # Stores the loss
            error_pde.append(l.item())
            u_predicted = BeamModel_u(TrialCoordinates) 
            # Stores the coordinates trajectories
            Coordinates_u_i = [BeamModel_u.coordinates[i].data.item() for i in range(len(BeamModel_u.coordinates))]
            Coordinates_du_i = [BeamModel_du.coordinates[i].data.item() for i in range(len(BeamModel_du.coordinates))]
            Coord_trajectories.append(Coordinates_u_i)

            if BoolCompareNorms:
                # Copute and store the L2 error w.r.t. the analytical solution
                error2.append(torch.norm(AnalyticSolution(A,E,TrialCoordinates.data) - u_predicted).data/analytical_norm)

        loss_current = l.item()
        loss_decrease = (loss_old - loss_current)/numpy.abs(loss_old)
        loss_old = loss_current

        if loss_decrease >= 0 and loss_decrease < 1.0e-7:
            stagnancy_counter = stagnancy_counter +1
        else:
            stagnancy_counter = 0

        if (epoch+1) % 1 == 0:
            print('* epoch ', epoch+1, ' loss = ', numpy.format_float_scientific( l.item(), precision=4))

        epoch = epoch+1

    plot_everything_mixed(A,E,InitialCoordinates_u, InitialCoordinates_du, Coordinates_u_i, Coordinates_du_i,
                                            TrialCoordinates, AnalyticSolution, BeamModel_u, BeamModel_du, \
                                            Coord_trajectories,  error_pde, [], error2)  

    print("*************** END OF SECOND STAGE ***************\n")
    print(f'* Final training loss: {numpy.format_float_scientific( error_pde[-1], precision=4)}')
    print(f'* Final l2 loss : {numpy.format_float_scientific( error2[-1], precision=4)}')


def LBFGS_Stage2_2D(Model_u, Model_du, Mesh, IDs_u, IDs_du, PlotCoordinates, 
                        #TrainCoordinates, TrainIDs_u, TrainIDs_du,
                        Cell_ids, Ref_Coord,
                        w0, w1, n_train, n_epochs, constit_point_coord, constit_cell_IDs_u, lmbda, mu):

    stagnancy_counter = 0
    loss_old = 1
    counter = 0

    optim = torch.optim.LBFGS(list(Model_u.parameters())+list(Model_du.parameters()),
                    history_size=5, 
                    max_iter=15, 
                    tolerance_grad = 1.0e-9,
                    line_search_fn="strong_wolfe")

    epoch = 0

    TrainCoordinates, TrainIDs_u, TrainIDs_du = GetRealCoord(Model_du, Mesh, Cell_ids, Ref_Coord)

    while stagnancy_counter < 5 and epoch<n_epochs:
        counter = counter+1

        def closure():
            optim.zero_grad()

            Neumann_BC_rel(Model_du)
            if len(constit_cell_IDs_u)>0:
                Constitutive_BC(Model_u, Model_du, constit_point_coord, constit_cell_IDs_u, lmbda, mu )

            u_predicted = Model_u(TrainCoordinates, TrainIDs_u) 
            du_predicted = Model_du(TrainCoordinates, TrainIDs_du) 

            l_pde, l_compat, s11, s22, s12 =  Mixed_2D_loss(u_predicted[0,:], u_predicted[1,:],
                                                            du_predicted[0,:], du_predicted[1,:], du_predicted[2,:], 
                                                            TrainCoordinates, lmbda = 1.25, mu = 1.0)
            l =  w0*l_pde +w1*l_compat

            l.backward()
            return l

        
        optim.step(closure)
        l = closure()
        loss_current = l.item()
        loss_decrease = (loss_old - loss_current)/numpy.abs(loss_old)
        loss_old = loss_current

        print("     Iter = ",counter," : Loss = ", numpy.format_float_scientific(l.item(), precision=4))
        #print("     ", (Model_du.nodal_values[0][0]).item(),(Model_du.nodal_values[0][1]).item())

        if loss_decrease >= 0 and loss_decrease < 1.0e-7:
            stagnancy_counter = stagnancy_counter +1
        else:
            stagnancy_counter = 0

        epoch = epoch+1

    Plot_all_2D(Model_u, Model_du, IDs_u, IDs_du, PlotCoordinates, [], n_train, "_Final")

    return Model_u, Model_du


def GradDescend_Stage1_2D(Model_u, Model_du, Mesh, IDs_u, IDs_du, PlotCoordinates,
                            CoordinatesBatchSet, w0, w1, n_epochs, optimizer, n_train, 
                            loss, constit_point_coord, constit_cell_IDs_u, lmbda, mu ):

    evaluation_time = 0
    loss_time = 0
    optimizer_time = 0
    backward_time = 0

    stagnancy_counter = 0
    loss_counter = 0

    start_train_time = time.time()
    loss_old = 1
    loss_current = 1
    loss_min = 1
    epoch = 0

    total = 20


    while epoch<n_epochs: #and (loss_counter<1 or loss_current > 1.0e-3): #and stagnancy_counter < 50 :

        # w0 = torch.randint(0, total,[1])
        # w1 = total-w0

        for DataSet in CoordinatesBatchSet:

            ##  Training points uniformly sampled in the domain 
            # TrialCoordinates =  DataSet[0]
            # TrialIDs_u = DataSet[1]
            # TrialIDs_du = DataSet[2]

            ##  n Training points generated in each element

            Cell_ids = DataSet[0]
            Ref_Coord = DataSet[1]

            TrialCoordinates, TrialIDs_u, TrialIDs_du = GetRealCoord(Model_du, Mesh, Cell_ids, Ref_Coord)

            ##### Should we move this to the loss function??? Maybe it's ok
            Neumann_BC_rel(Model_du)

            if len(constit_cell_IDs_u)>0:
                Constitutive_BC(Model_u, Model_du, constit_point_coord, constit_cell_IDs_u, lmbda, mu )
            ##### ------------------------------------------

            start_time = time.time()
            u_predicted = Model_u(TrialCoordinates, TrialIDs_u) 
            du_predicted = Model_du(TrialCoordinates, TrialIDs_du) 
            evaluation_time += time.time() - start_time

            start_time = time.time()
            l_pde, l_compat, _, _, _ =  Mixed_2D_loss(u_predicted[0,:], u_predicted[1,:],
                                                        du_predicted[0,:], du_predicted[1,:], du_predicted[2,:], 
                                                        TrialCoordinates, lmbda, mu )

            #l_pde, l_compat, _, _, _ =  Mixed_2D_loss_Displacement_based(Model_u, Model_du, Mesh, TrialCoordinates, lmbda, mu )

            l =  w0*l_pde +w1*l_compat
            loss_time += time.time() - start_time

            loss[0].append(l_pde.item())
            loss[1].append(l_compat.item())

            loss_current = l_pde.item()+l_compat.item()
            
            # if loss_min > loss_current:
            #     loss_min = loss_current
            #     loss_counter = 0

            #     torch.save(Model_u.state_dict(),"Results/Model_u.pt")
            #     torch.save(Model_du.state_dict(),"Results/Model_du.pt")
            # else:
            #     loss_counter += 1

            start_time = time.time()
            l.backward()
            backward_time += time.time() - start_time

            start_time = time.time()
            optimizer.step()
            optimizer_time += time.time() - start_time

            optimizer.zero_grad()

            # loss_decrease = (loss_old - loss_current)/numpy.abs(loss_old)
            # loss_old = loss_current

            # if loss_decrease >= 0 and loss_decrease < 1.0e-7:
            #     stagnancy_counter = stagnancy_counter +1
            # else:
            #     stagnancy_counter = 0

        if (epoch+1)%50 == 0:
            print("     epoch = ", epoch +1)
            print("     loss_counter = ", loss_counter)
            print("     mean loss PDE = ", numpy.format_float_scientific(numpy.mean(loss[0][-10:-1]), precision=4))
            print("     mean loss compatibility = ", numpy.format_float_scientific(numpy.mean(loss[1][-10:-1]), precision=4))
            print("     w0, w1 : ", w0, w1)
            #print("     ", (Model_du.nodal_values[0][0]).item(),(Model_du.nodal_values[0][1]).item())
            #print()
            #print("     var loss PDE = ", numpy.sqrt(numpy.var(loss[0][-10:-1])))
            #print("     var loss compatibility = ", numpy.sqrt(numpy.var(loss[1][-10:-1])))      
            print("     ...............................")

        if (epoch+1) % 200 == 0:
            l = Plot_all_2D(Model_u, Model_du, IDs_u, IDs_du, PlotCoordinates, loss, n_train, "_Stage1")
            print("     _______________________________")

        epoch = epoch+1

    stopt_train_time = time.time()

    #print("loss_current = ", loss_current)
    #print("loss_counter = ", loss_counter)

    l = Plot_all_2D(Model_u, Model_du, IDs_u, IDs_du, PlotCoordinates, loss, n_train, "_Stage1")

    # if loss_min < l:
    #     print("     ****** REVERT ******")
    #     Model_u.load_state_dict(torch.load("Results/Model_u.pt"))
    #     Model_du.load_state_dict(torch.load("Results/Model_du.pt"))

    #     l = Plot_all_2D(Model_u, Model_du, IDs_u, IDs_du, PlotCoordinates,loss, n_train, "_Stage1")
        
    print("*************** END OF TRAINING ***************\n")
    print(f'* Training time: {stopt_train_time-start_train_time}s\n\
        * Evaluation time: {evaluation_time}s\n\
        * Loss time: {loss_time}s\n\
        * Backward time: {backward_time}s\n\
        * Training time per epochs: {(stopt_train_time-start_train_time)/n_epochs}s\n\
        * Optimiser time: {optimizer_time}s\n')

    return Model_u, Model_du, loss





def Training_2D_Integral(model, optimizer, n_epochs,List_elems,Mat):
    
    # Initialise vector of loss values
    Loss_vect = []
    print("**************** START TRAINING ***************\n")
    time_start = time.time()
    epoch = 0
    save_time = 0
    eval_time = 0
    back_time = 0
    update_time = 0
    model.train()
    TrailCoord_1d_x = torch.tensor([i for i in torch.linspace(0,1,1)],dtype=torch.float64, requires_grad=True)
    TrailCoord_1d_y = torch.tensor([i for i in torch.linspace(0,5*1,5*1)],dtype=torch.float64,  requires_grad=True)
    PlotCoordinates = torch.cartesian_prod(TrailCoord_1d_x,TrailCoord_1d_y)
    model.Initresults()
    stagnation = False
    flag_Stop_refinement = False
    while epoch<model.Max_epochs and not stagnation:
        # Compute loss
        loss_time_start = time.time()
        u_predicted,xg,detJ = model()
        # loss_previous = torch.sum((0.5*InternalEnergy_2D(u_predicted,xg,Mat.lmbda, Mat.mu)-1*VolumeForcesEnergy_2D(u_predicted,xg,theta = torch.tensor(0*torch.pi/2), rho = 1e-9))*torch.abs(detJ))
        loss = torch.sum((0.5*InternalEnergy_2D_einsum(u_predicted,xg,Mat.lmbda, Mat.mu)-1*VolumeForcesEnergy_2D(u_predicted,xg,theta = torch.tensor(0*torch.pi/2), rho = 1e-9))*torch.abs(detJ))
        eval_time += time.time() - loss_time_start
        loss_current = loss.item()
        backward_time_start = time.time()
        loss.backward()
        back_time += time.time() - backward_time_start
        # update weights
        update_time_start = time.time()
        optimizer.step()
        update_time += time.time() - update_time_start
        # zero the gradients after updating
        optimizer.zero_grad()
        with torch.no_grad():
            epoch+=1
            if epoch >1:
                d_loss = 2*(torch.abs(loss.data-loss_old))/(torch.abs(loss.data+loss_old))
                loss_old = loss.data
                D_detJ = (torch.abs(model.detJ_0) - torch.abs(detJ))/torch.abs(model.detJ_0)
                if torch.max(D_detJ)>model.Jacobian_threshold:
                    indices = torch.nonzero(D_detJ > model.Jacobian_threshold)
                    # Re-initialise future splitted elements' jacobian as base for the newly splitted elements
                    model.detJ_0[indices] = detJ[indices]
                    Removed_elem_list = []
                    old_generation = model.elements_generation
                    for i in range(indices.shape[0]):
                        el_id = indices[i]  
                        if model.elements_generation[el_id.item()]<model.MaxGeneration:
                            model.MaxGeneration_elements=1
                        if el_id.item() not in Removed_elem_list and model.elements_generation[el_id.item()]<model.MaxGeneration:
                            el_id = torch.tensor([el_id],dtype=torch.int)
                            new_coordinate = xg[el_id]
                            model.eval()
                            newvalue = model(new_coordinate,el_id) 
                            model.train()
                            Removed_elems = model.SplitElemNonLoc(el_id)
                            Removed_elems[0] = Removed_elems[0].numpy()
                            # Update indexes 
                            for j in range(indices.shape[0]):
                                number_elems_above = len([e for e in Removed_elems if e < indices[j].numpy()])
                                indices[j] = indices[j] - number_elems_above
                            # Update indexes of Removed_elem_list
                            for j in range(len(Removed_elem_list)):
                                number_elems_above = len([e for e in Removed_elems if e < Removed_elem_list[j]])
                                Removed_elem_list[j] = Removed_elem_list[j] - number_elems_above

                            # Add newly removed elems to list
                            Removed_elem_list += Removed_elems
                            # List_elems = torch.arange(0,model.NElem,dtype=torch.int)
                            optimizer.add_param_group({'params': model.coordinates[-3:]})
                            optimizer.add_param_group({'params': model.nodal_values[0][-3:]})
                            optimizer.add_param_group({'params': model.nodal_values[1][-3:]})
                    _,_,detJ = model(PlotCoordinates, List_elems)
                    # model.detJ = detJ

                    # model.Freeze_Mesh()
                if d_loss < model.Stagnation_threshold:
                    stagnation = True
                    # stagnation = False
            else:
                loss_old = loss.item()
                detJ_0 = detJ
                model.detJ_0 = detJ
            Loss_vect.append(loss.item())
        if (epoch+1) % 50 == 0 or epoch ==1 or epoch==model.Max_epochs or stagnation:
            model.StoreResults()
            print(f'epoch {epoch+1} loss = {numpy.format_float_scientific(loss.item(), precision=4)}')

    time_stop = time.time()
    # print("*************** END OF TRAINING ***************\n")
    print("*************** END FIRST PHASE ***************\n")
    print(f'* Training time: {time_stop-time_start}s')
    print(f'* Saving time: {save_time}s')
    print(f'* Evaluation time: {eval_time}s')
    print(f'* Backward time: {back_time}s')
    print(f'* Update time: {update_time}s')
    print(f'* Average epoch time: {(time_stop-time_start)/(epoch+1)}s')

    return Loss_vect, (time_stop-time_start)
    