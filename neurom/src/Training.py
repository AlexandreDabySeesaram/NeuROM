import numpy as numpy
import torch
from ..HiDeNN_PDE import MeshNN, NeuROM, MeshNN_2D
from ..Post import Plots as Pplot
import copy
import time
from . import Pre_processing as pre
import torch
import random 
import torch.nn as nn
from .PDE_Library import RHS, PotentialEnergy, \
    PotentialEnergyVectorised, AlternativePotentialEnergy, \
        Derivative, AnalyticGradientSolution, AnalyticSolution,\
            PotentialEnergyVectorisedParametric,AnalyticParametricSolution, \
                PotentialEnergyVectorisedBiParametric, MixedFormulation_Loss,\
                Mixed_2D_loss, Neumann_BC_rel, Constitutive_BC, GetRealCoord,\
                    InternalEnergy_2D, VolumeForcesEnergy_2D,InternalEnergy_2D_einsum, InternalResidual,Strain_sqrt,InternalResidual_precomputed,\
                        InternalEnergy_2D_einsum_para,InternalEnergy_2D_einsum_Bipara, Strain, Stress, PotentialEnergyVectorisedParametric_Gauss,\
                            InternalEnergy_2D_einsum_BiStiffness,\
                            InternalEnergy_1D, WeakEquilibrium_1D

def plot_everything(A,E,InitialCoordinates,Coordinates,
                    TrialCoordinates,AnalyticSolution,BeamModel,Coord_trajectories, error, error2):
    Pplot.PlotTrajectories(Coord_trajectories,'Trajectories')

    # Tests on trained data and compare to reference
    Pplot.PlotSolution_Coordinates_Analytical(  A,E,InitialCoordinates,Coordinates,
                                                TrialCoordinates,AnalyticSolution,BeamModel,
                                                'Solution_displacement')
    # Plots the gradient & compare to reference
    Pplot.PlotGradSolution_Coordinates_Analytical(  A,E,InitialCoordinates,Coordinates,
                                                    TrialCoordinates,AnalyticGradientSolution,
                                                    BeamModel,Derivative,'Solution_gradients')

    Pplot.Plot_Compare_Loss2l2norm(error,error2,'Loss_Comaprison')

def plot_everything_mixed(  A,E,InitialCoordinates_u, InitialCoordinates_du, Coordinates_u, Coordinates_du,
                            TrialCoordinates,AnalyticSolution,BeamModel_u,BeamModel_du, \
                            Coord_trajectories, error_pde, error_constit, error2):
    # Tests on trained data and compare to reference
    Pplot.PlotSolution_Coordinates_Analytical(  A,E,InitialCoordinates_u,Coordinates_u,
                                                TrialCoordinates,AnalyticSolution,BeamModel_u,
                                                'Solution_displacement')
    # Plots the gradient & compare to reference

    Pplot.PlotGradSolution_Coordinates_Analytical(  A,E,InitialCoordinates_u,Coordinates_u,
                                                    TrialCoordinates,AnalyticGradientSolution,
                                                    BeamModel_u,Derivative,'Solution_gradients_dudx')

    Pplot.PlotGradSolution_Coordinates_Force(   A,E,InitialCoordinates_du,Coordinates_du,
                                                TrialCoordinates, RHS(TrialCoordinates),
                                                BeamModel_du,Derivative,'Solution_gradients_Force')

    Pplot.PlotSolution_Coordinates_Analytical(  A,E,InitialCoordinates_du,Coordinates_du,
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
    print("     total = ", numpy.format_float_scientific(l_pde.item() + l_compat.item(),precision=4))

    #Model_u.CheckBCValues()
    Pplot.Plot2Dresults(u_predicted, PlotCoordinates, "_u"+stage)
    #Pplot.Plot1DSection(u_predicted, n_train, 5*n_train, stage)
    Pplot.Plot2Dresults_Derivative(du_predicted, s11, s22, s12, PlotCoordinates, "_Stress" + stage)

    if len(loss)>0:
        Pplot.Plot2DLoss(loss)

    return l

def Collision_Check(model, coord_old, proximity_limit):

    correction = False
    cell_id = torch.arange(0,model.NElem,dtype=torch.int)

    cell_nodes_IDs = model.connectivity[cell_id,:]
    if cell_nodes_IDs.ndim == 1:
        cell_nodes_IDs = np.expand_dims(cell_nodes_IDs,0)


    node1_coord =  torch.cat([model.coordinates[int(row)-1] for row in cell_nodes_IDs[:,0]])
    node2_coord =  torch.cat([model.coordinates[int(row)-1] for row in cell_nodes_IDs[:,1]])    

    detJ = (node2_coord - node1_coord)

    idx = torch.where(detJ[:,0]<proximity_limit)[0]
    if len(idx)>0:
        correction = True
        for l in range(len(idx)):

            i = idx[l]

            cell_nodes_IDs = model.connectivity[i,:] - 1
            if cell_nodes_IDs.ndim == 1:
                cell_nodes_IDs = numpy.expand_dims(cell_nodes_IDs,0)

            # print(cell_nodes_IDs[:,0])
            # print(cell_nodes_IDs[:,1])
            # # print()
            # print(model.coordinates[int(cell_nodes_IDs[:,0].item())])
            # print(model.coordinates[int(cell_nodes_IDs[:,1].item())])
            # print()
            model.coordinates[int(cell_nodes_IDs[:,0].item())].data = torch.tensor( [[coord_old[int(cell_nodes_IDs[:,0].item())]]])
            model.coordinates[int(cell_nodes_IDs[:,1].item())].data = torch.tensor( [[coord_old[int(cell_nodes_IDs[:,1].item())]]])

            # print("After fix")
            # print(model.coordinates[int(cell_nodes_IDs[:,0].item())])
            # print(model.coordinates[int(cell_nodes_IDs[:,1].item())])
            # print()
    return correction

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

def Training_InitialStage(BeamModel, A, E, L, TrialCoordinates, optimizer, n_epochs, BoolCompareNorms, MSE, BoolFilterTrainingData, TestCoordinates):

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

    analytical_norm = torch.linalg.vector_norm(AnalyticSolution(A,E,TestCoordinates.data)).data

    while epoch<n_epochs and stagnancy_counter < 50 and loss_counter<1000: # 50, 1000

        if BoolFilterTrainingData:
            TrialCoordinates = FilterTrainingData(BeamModel, TrialCoordinates)

        coord_old = [BeamModel.coordinates[i].data.item() for i in range(len(BeamModel.coordinates))]
        # weights_old = copy.deepcopy(BeamModel.InterpoLayer_uu.weight.data.detach())

        # predict = forward pass with our model
        start_time = time.time()
        u_predicted = (TrialCoordinates) 
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
                error2.append(torch.linalg.vector_norm(AnalyticSolution(A,E,TrialCoordinates.data) - u_predicted)/analytical_norm)
                #error2.append(numpy.sqrt(scipy.integrate.trapezoid((AnalyticSolution(A,E,TrialCoordinates.data) - u_predicted)**2, TrialCoordinates.data,dx=0.1)))



        if loss_decrease >= 0 and loss_decrease < 1.0e-7:
            stagnancy_counter = stagnancy_counter +1
        else:
            stagnancy_counter = 0

        
        if (epoch+1) % 200 == 0:
            print('* epoch ', epoch+1, ' loss = ', numpy.format_float_scientific( l.item(), precision=4))
            print("* loss decrease = ",  numpy.format_float_scientific( loss_decrease, precision=4))
            Pplot.Plot_Compare_Loss2l2norm(error,error2,'Loss_Comaprison')

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
            u_predicted = BeamModel(TestCoordinates) 
            error2.append(torch.norm(AnalyticSolution(A,E,TestCoordinates.data)-u_predicted)/analytical_norm)

    plot_everything(A,E,InitialCoordinates,Coordinates_i,
                                                TrialCoordinates,AnalyticSolution,BeamModel,Coord_trajectories,error, error2)

    print(f'* Final training loss: {numpy.format_float_scientific( error[-1], precision=4)}')
    print(f'* Final l2 loss : {numpy.format_float_scientific( error2[-1], precision=4)}')

    return error, error2, InitialCoordinates, Coord_trajectories, BeamModel

def Training_FinalStageLBFGS(BeamModel, A, E, L, InitialCoordinates, TrialCoordinates, n_epochs, BoolCompareNorms, MSE, BoolFilterTrainingData, 
                                TestCoordinates,
                                error=[], error2 =[],Coord_trajectories=[]):
    optim = torch.optim.LBFGS(BeamModel.parameters(),
                    #history_size=5, 
                    #max_iter=15, 
                    #tolerance_grad = 1.0e-9,
                    line_search_fn="strong_wolfe")
    print()
    print("*************** SECOND STAGE (LBFGS) ***************\n")
    loss_old = 1
    epoch = 0
    stagnancy_counter = 0

    analytical_norm = torch.linalg.vector_norm(AnalyticSolution(A,E,TestCoordinates.data)).data
    analytical_grad_norm = torch.linalg.vector_norm(AnalyticGradientSolution(A,E,TestCoordinates.data)).data

    print()
    print("analytical_norm = ", analytical_norm)
    print()

    while epoch<n_epochs and stagnancy_counter < 5:

        coord_old = [BeamModel.coordinates[i].data.item() for i in range(len(BeamModel.coordinates))]

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

        Collision_Check(BeamModel, coord_old, 1.0e-6)


        with torch.no_grad():
            # Stores the loss
            error.append(l.item())
            # Stores the coordinates trajectories
            Coordinates_i = [BeamModel.coordinates[i].data.item() for i in range(len(BeamModel.coordinates))]
            Coord_trajectories.append(Coordinates_i)

        if BoolCompareNorms:
            # Copute and store the L2 error w.r.t. the analytical solution
            u_predicted = BeamModel(TestCoordinates) 
            error2.append(torch.linalg.vector_norm(AnalyticSolution(A,E,TestCoordinates.data)-u_predicted).data/analytical_norm)
            #error2.append(numpy.sqrt(scipy.integrate.trapezoid((AnalyticSolution(A,E,TrialCoordinates.data) - u_predicted)**2, TrialCoordinates.data,dx=0.1)))
            du_dx = torch.autograd.grad(u_predicted, TestCoordinates, grad_outputs=torch.ones_like(u_predicted), create_graph=True)[0]
            error2_grad = torch.linalg.vector_norm(AnalyticGradientSolution(A,E,TestCoordinates.data)-du_dx).data/analytical_grad_norm




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

    plot_everything(A,E,InitialCoordinates,Coordinates_i,
                        TrialCoordinates,AnalyticSolution,BeamModel,Coord_trajectories,error, error2)

    print("*************** END OF SECOND STAGE ***************\n")
    print(f'* Final training loss: {numpy.format_float_scientific( error[-1], precision=4)}')
    print(f'* Final l2 loss : {numpy.format_float_scientific( error2[-1], precision=4)}')

def Training_NeuROM(model, config, optimizer, Mat = 'NaN'):
    if config["interpolation"]["dimension"] == 1:
        A                   = config["geometry"]["A"]
        L                   = config["geometry"]["L"]
    n_epochs            = config["training"]["n_epochs"]
    # BiPara              = config["solver"]["BiPara"]
    loss_decrease_c     = config["training"]["loss_decrease_c"]
    ### Generate training points coordinates
    # In space
    Training_coordinates = torch.tensor([[i/50] for i in range(2,500)], 
                                        dtype=model.float_config.dtype, 
                                        device = model.float_config.device,
                                        requires_grad=True)
    # In the parameters space
    Training_para_coordinates_1 = torch.linspace(
                                                config["parameters"]["para_1_min"],
                                                config["parameters"]["para_1_max"],
                                                5*config["parameters"]["N_para_1"], 
                                                dtype=model.float_config.dtype, 
                                                device = model.float_config.device,
                                                requires_grad=True
                                                )

    Training_para_coordinates_1 = Training_para_coordinates_1[:,None]

    Training_para_coordinates_2 = torch.linspace(
                                                config["parameters"]["para_2_min"],
                                                config["parameters"]["para_2_max"],
                                                5*config["parameters"]["N_para_2"], 
                                                dtype=model.float_config.dtype, 
                                                device = model.float_config.device,
                                                requires_grad=True
                                                )

    Training_para_coordinates_2 = Training_para_coordinates_2[:,None]  
    match config["solver"]["N_ExtraCoordinates"]:
        case 2:
            Training_para_coordinates_list = nn.ParameterList(
                                                                (Training_para_coordinates_1,
                                                                Training_para_coordinates_2))
        case 1:
            Training_para_coordinates_list = [Training_para_coordinates_1]


    # BCs used for the analytical comparison 
    if config["interpolation"]["dimension"] == 1:
        match config["solver"]["IntegralMethod"]:
            case "Trapezoidal":
                u0                      = model.Space_modes[0].u_0                      # Left BC
                uL                      = model.Space_modes[0].u_L                      # Right BC
            case "Gaussian_quad":
                u0 = model.Space_modes[0].ListOfDirichletsBCsValues[0]
                uL = model.Space_modes[0].ListOfDirichletsBCsValues[1]
    print("**************** START TRAINING ***************\n")
    time_start              = time.time()
    epoch                   = 0                                             # Initial epoch number
    loss_counter            = 0                                             # Ounter for loss stoped decreasing 
    save_time               = 0                                             # Saving model stopwatch
    eval_time               = 0                                             # Evaluating model stopwatch                                   
    back_time               = 0                                             # Backpropagation stopwatch
    update_time             = 0                                             # Updating parameters stopwatch

    Add_mode_c              = 1e-5                                          # Criterion of stagnation before adding a new mode
    FlagAddedMode_usefull   = True                                          # Flag stating that the new mode did help speeding-up the convergence
    stagnancy_counter       = 0                                             # Stagnancy since last additoin of a mode
    local_stagnancy_counter = 0
    FlagAddedMode           = False                                         # Flag activated when a new mode has been added
    try:
        loss_init = model.training_recap["Loss_vect"]                       # Test if model.training_recap exists
    except:
        model.training_recap = {"Loss_vect":[],
                            "L2_error":[],                                  # Init L2 error
                            "training_time":0,                              # Init Training duration
                            "Mode_vect":[],                                 # Size ROB
                            "Loss_decrease_vect":[]                         # Init loss decrease rate
                            }
    Usefullness             = 0                                             # Number of iteration in a row during which the last added mode helped the convergence

    while epoch<n_epochs and loss_counter<100:
        if stagnancy_counter>5 and (not FlagAddedMode_usefull or model.n_modes_truncated >= model.n_modes):               # Break if stagnation not solved by adding modes (hopefully that means convergence reached)
            break 

        # Compute loss
        loss_time_start             = time.time()
        match config["solver"]["N_ExtraCoordinates"]:
            case 1:
                match config["interpolation"]["dimension"]:
                    case 1:
                        match config["solver"]["IntegralMethod"]:   
                            case "Gaussian_quad":
                                loss = PotentialEnergyVectorisedParametric_Gauss(model,A,Training_para_coordinates_list)
                            case "Trapezoidal":
                                loss = PotentialEnergyVectorisedParametric(model,A,Training_para_coordinates_list,model(Training_coordinates,Training_para_coordinates_list),Training_coordinates,RHS(Training_coordinates))
                    case 2:
                            loss = InternalEnergy_2D_einsum_para(model,Mat.lmbda, Mat.mu,Training_para_coordinates_list)
            case 2:
                match config["interpolation"]["dimension"]:
                    case 1:
                        loss = PotentialEnergyVectorisedBiParametric(model,A,Training_para_coordinates_list,Training_coordinates,RHS(Training_coordinates))
                    case 2:  
                        match config["solver"]["Problem"]:
                            case "AngleStiffness":
                                loss = InternalEnergy_2D_einsum_Bipara(model,Mat.lmbda, Mat.mu,Training_para_coordinates_list)
                            case "BiStiffness":
                                loss = InternalEnergy_2D_einsum_BiStiffness(model,Mat.lmbda, Mat.mu,Training_para_coordinates_list)
        eval_time                   += time.time() - loss_time_start
        loss_current                = loss.item()
         # check for new minimal loss - Update the state for revert
        if epoch >1:
            loss_decrease           = (loss_old - loss_current)/numpy.abs(0.5*(loss_old + loss_current))
            model.training_recap["Loss_decrease_vect"].append(loss_decrease)
            loss_old                = loss_current
            if numpy.abs(loss_decrease) < loss_decrease_c:                  # Check for stagnation of the loss
                stagnancy_counter   = stagnancy_counter +1                  # Increment stagnation
                Usefullness         = 0                                     # Reinit. usefullness of last added mode
            else:
                stagnancy_counter   = 0                                     # Reinit. stagnation
                if loss_decrease    >= 0:                                   # Check that loss decreases
                    Usefullness     +=1                                      # Increment usefullness of of the last added mode
                    if Usefullness>=15:                                     # Check if mode was usefull for more than 15 iterations in a raw
                        FlagAddedMode_usefull = True                        # Flag stating that the new mode did help speeding-up the convergence

            if loss_min > loss_current:  
                loss_min    = loss_current
                with torch.no_grad():
                    if (epoch+1) % 300 == 0:                                # Update saved model only every 300 iterations to save saving time
                        save_start  = time.time()
                        loss_min_saved = loss_current
                        # Current_best = copy.deepcopy(model.state_dict())    # Store in variable instead of writing file
                        # TODO: account for changinf number of mode,i.e., change in number of parameters     
                        save_stop   = time.time()
                        save_time   +=(save_stop-save_start)
                    loss_counter    = 0                                     # breaks series of non decreasing loss
            else:
                loss_counter += 1                                           # increments breaks series of non decreasing loss
                
        else:
            loss_min_saved          = loss_current + 1   
            loss_min                = loss_current + 1                      # Initialise to dummy (lagrger than current) loss min
            loss_old                = loss_current                          # Initialise previous loss

        backward_time_start         = time.time()
        loss.backward()
        back_time += time.time() - backward_time_start
        # update weights
        if config["interpolation"]["dimension"] ==1:
            model.SaveCoordinates()                                             # Save coordinates to check for collisions
        update_time_start           = time.time()
        optimizer.step()                                                    # Update parameters
        if config["interpolation"]["dimension"] ==1:
            for m in range(model.n_modes_truncated):                            # Check for collisions
                Collision_Check(model.Space_modes[m], model.Space_modes[m].coord_old, 1.0e-6)

        update_time                 += time.time() - update_time_start
        optimizer.zero_grad()                                               # zero the gradients after updating
        model.training_recap["Mode_vect"].append(model.n_modes_truncated.detach().clone())
        if (stagnancy_counter >5 or loss_counter>90) and model.n_modes_truncated < model.n_modes and FlagAddedMode_usefull:
        # if stagnancy_counter >5 and model.n_modes_truncated < model.n_modes and FlagAddedMode_usefull:
            model.AddMode()
            model.AddMode2Optimizer(optimizer)
            Addition_epoch_index = epoch
            # loss_counter            = 0 loss_counter>99
            FlagAddedMode           = True
            FlagAddedMode_usefull   = False                                 # Flag stating that the new mode did help speeding-up the convergence
            stagnancy_counter       = 0
            Usefullness             = 0
        if FlagAddedMode:
            if epoch == Addition_epoch_index+2:
                model.UnfreezeTruncated()
                if not config["training"]["multi_mode_training"]:
                    model.Freeze_N_1()
                stagnancy_counter   = 0

        with torch.no_grad():
            epoch+=1
            model.training_recap["Loss_vect"].append(loss.item())
            numel_E = Training_para_coordinates_list[0].shape[0]
            match config["solver"]["N_ExtraCoordinates"]: 
                case 1:
                    if config["interpolation"]["dimension"] == 1:
                        match config["solver"]["IntegralMethod"]:   
                            case "Trapezoidal":
                                model.training_recap["L2_error"].append((torch.norm(torch.sum(AnalyticParametricSolution(A,Training_para_coordinates_list,Training_coordinates.data,u0,uL)-model(Training_coordinates,Training_para_coordinates_list),dim=1)/numel_E).data)/(torch.norm(torch.sum(AnalyticParametricSolution(A,Training_para_coordinates_list,Training_coordinates.data,u0,uL),dim=1)/numel_E)))
                            case "Gaussian_quad":
                                model.training_recap["L2_error"].append(1)
        if (epoch+1) % 100 == 0:
            if config["solver"]["N_ExtraCoordinates"] == 1 and config["interpolation"]["dimension"] == 1:
                print(f'epoch {epoch+1} loss = {numpy.format_float_scientific(loss.item(), precision=5)} error = {numpy.format_float_scientific(100*model.training_recap["L2_error"][-1], precision=4)}% modes = {model.n_modes_truncated}')
            else:
                print(f'epoch {epoch+1} loss = {numpy.format_float_scientific(loss.item(), precision=5)} modes = {model.n_modes_truncated}')

    time_stop = time.time()
    model.training_recap["training_time"] += (time_stop-time_start)

    # print("*************** END OF TRAINING ***************\n")
    print("*************** END FIRST PHASE ***************\n")
    print(f'* Training time: {model.training_recap["training_time"]}s')
    print(f'* Saving time: {save_time}s')
    print(f'* Evaluation time: {eval_time}s')
    print(f'* Backward time: {back_time}s')
    print(f'* Update time: {update_time}s')
    print(f'* Average epoch time: {(time_stop-time_start)/(epoch+1)}s')

    # Final loss evaluation - Revert to minimal-loss state if needed
    if loss_min_saved < loss_current:
        print("*************** REVERT TO BEST  ***************\n")
        # model.load_state_dict(Current_best) # Load from variable instead of written file
        print("* Minimal loss = ", loss_min)
    
    return 

def Training_NeuROM_FinalStageLBFGS(model,config, Mat = 'NaN'):
    Current_best_model = copy.deepcopy(model.state_dict())    # Store in variable instead of writing file
    try:
        initial_loss = model.training_recap["Loss_vect"][-1]
    except:
        initial_loss = 1
        model.training_recap = {"Loss_vect":[],
                                "L2_error":[],
                                "training_time":0,
                                "Mode_vect":[],
                                "Loss_decrease_vect":[]
                                }    
    model.Freeze_Mesh()
    optim = torch.optim.LBFGS([p for p in model.parameters() if p.requires_grad],
                    #history_size=5, 
                    #max_iter=15, 
                    #tolerance_grad = 1.0e-9,
                    line_search_fn="strong_wolfe")
    
    A               = config["geometry"]["A"]
    L               = config["geometry"]["L"]
    n_epochs        = config["training"]["n_epochs"]
    # BiPara          = config["solver"]["BiPara"]
    loss_decrease_c = config["training"]["loss_decrease_c"]
    ### Generate training points coordinates
    # In space
    Training_coordinates = torch.tensor([[i/50] for i in range(2,500)], 
                                        dtype=model.float_config.dtype, 
                                        device = model.float_config.device,
                                        requires_grad=True)
    # In the parameters space
    Training_para_coordinates_1 = torch.linspace(
                                                config["parameters"]["para_1_min"],
                                                config["parameters"]["para_1_max"],
                                                5*config["parameters"]["N_para_1"], 
                                                dtype=model.float_config.dtype, 
                                                device = model.float_config.device,
                                                requires_grad=True
                                                )

    Training_para_coordinates_1 = Training_para_coordinates_1[:,None]

    Training_para_coordinates_2 = torch.linspace(
                                                config["parameters"]["para_2_min"],
                                                config["parameters"]["para_2_max"],
                                                5*config["parameters"]["N_para_2"], 
                                                dtype=model.float_config.dtype, 
                                                device = model.float_config.device,
                                                requires_grad=True
                                                )

    Training_para_coordinates_2 = Training_para_coordinates_2[:,None] 
    match config["solver"]["N_ExtraCoordinates"]:
        case 2:
            Training_para_coordinates_list = nn.ParameterList(
                                                                (Training_para_coordinates_1,
                                                                Training_para_coordinates_2))
        case 1:
            Training_para_coordinates_list = [Training_para_coordinates_1]

    epoch                       = 0
    stagnancy_counter           = 0
    # model.UnFreeze_Mesh()
    loss_old = initial_loss
    # BCs used for the analytical comparison 
    if config["interpolation"]["dimension"] == 1:
        match config["solver"]["IntegralMethod"]:
            case "Trapezoidal":
                u0                      = model.Space_modes[0].u_0                      # Left BC
                uL                      = model.Space_modes[0].u_L                      # Right BC
            case "Gaussian_quad":
                u0 = model.Space_modes[0].ListOfDirichletsBCsValues[0]
                uL = model.Space_modes[0].ListOfDirichletsBCsValues[1]
    print("************** START SECOND PAHSE *************\n")
    time_start = time.time()
    while  epoch<n_epochs and stagnancy_counter < 5:
        # Compute loss

        def closure():
            optim.zero_grad()
            # if not BiPara:
            #     match config["solver"]["IntegralMethod"]:   
            #         case "Gaussian_quad":
            #             loss = PotentialEnergyVectorisedParametric_Gauss(model,A,Training_para_coordinates_list)
            #         case "Trapezoidal":
            #             loss = PotentialEnergyVectorisedParametric(model,A,Training_para_coordinates_list,model(Training_coordinates,Training_para_coordinates_list),Training_coordinates,RHS(Training_coordinates))
            # else:
            #     loss = PotentialEnergyVectorisedBiParametric(model,A,Training_para_coordinates_list,Training_coordinates,RHS(Training_coordinates))
            match config["solver"]["N_ExtraCoordinates"]:
                case 1:
                    match config["interpolation"]["dimension"]:
                        case 1:
                            match config["solver"]["IntegralMethod"]:   
                                case "Gaussian_quad":
                                    loss = PotentialEnergyVectorisedParametric_Gauss(model,A,Training_para_coordinates_list)
                                case "Trapezoidal":
                                    loss = PotentialEnergyVectorisedParametric(model,A,Training_para_coordinates_list,model(Training_coordinates,Training_para_coordinates_list),Training_coordinates,RHS(Training_coordinates))
                        case 2:
                                loss = InternalEnergy_2D_einsum_para(model,Mat.lmbda, Mat.mu,Training_para_coordinates_list)
                case 2:
                    match config["interpolation"]["dimension"]:
                        case 1:
                            loss = PotentialEnergyVectorisedBiParametric(model,A,Training_para_coordinates_list,Training_coordinates,RHS(Training_coordinates))
                        case 2:
                            match config["solver"]["Problem"]:
                                case "AngleStiffness":
                                    loss = InternalEnergy_2D_einsum_Bipara(model,Mat.lmbda, Mat.mu,Training_para_coordinates_list)
                                case "BiStiffness":
                                    loss = InternalEnergy_2D_einsum_BiStiffness(model,Mat.lmbda, Mat.mu,Training_para_coordinates_list)            
            loss.backward()
            return loss

        # model.SaveCoordinates()                                             # Save coordinates to check for collisions
        optim.step(closure)
        # for m in range(model.n_modes_truncated):                            # Check for collisions
        #     Collision_Check(model.Space_modes[m], model.Space_modes[m].coord_old, 1.0e-6)

        loss                    = closure()

        loss_current            = loss.item()
        # loss_decrease           = (loss_old - loss_current)/numpy.abs(loss_old)
        loss_decrease           = (loss_old - loss_current)/numpy.abs(0.5*(loss_old + loss_current))
        model.training_recap["Loss_decrease_vect"].append(loss_decrease)
        loss_old = loss_current
        if loss_decrease >= 0 and loss_decrease < 1.0e-7:
            stagnancy_counter   = stagnancy_counter +1
        else:
            stagnancy_counter   = 0

        with torch.no_grad():
            epoch+=1
            model.training_recap["Loss_vect"].append(loss.item())
            model.training_recap["Mode_vect"].append(model.n_modes_truncated.detach().clone())
            numel_E = Training_para_coordinates_list[0].shape[0]
            match config["solver"]["N_ExtraCoordinates"]:
                case 1:
                    match config["solver"]["IntegralMethod"]:   
                        case "Trapezoidal":
                            model.training_recap["L2_error"].append((torch.norm(torch.sum(AnalyticParametricSolution(A,Training_para_coordinates_list,Training_coordinates.data,u0,uL)-model(Training_coordinates,Training_para_coordinates_list),dim=1)/numel_E).data)/(torch.norm(torch.sum(AnalyticParametricSolution(A,Training_para_coordinates_list,Training_coordinates.data,u0,uL),dim=1)/numel_E)))
                        case "Gaussian_quad":
                            model.training_recap["L2_error"].append(1)
        if (epoch+1) % 5 == 0:
            if config["solver"]["N_ExtraCoordinates"] == 1:
                print(f'epoch {epoch+1} loss = {numpy.format_float_scientific(loss.item(), precision=4)} error = {numpy.format_float_scientific(100*model.training_recap["L2_error"][-1], precision=4)}%')
            else:
                print(f'epoch {epoch+1} loss = {numpy.format_float_scientific(loss.item(), precision=4)}')

    time_stop = time.time()
    model.training_recap["training_time"]+=time_stop-time_start
    print("*************** END OF TRAINING ***************\n")
    print(f'* Training time: {model.training_recap["training_time"]}s')
    if model.training_recap["Loss_vect"][-1] > initial_loss:
        print("*************** REVERT TO 1st STAGE MODEL ***************\n")
        model.load_state_dict(Current_best_model) 
    return 

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
    loss_min = 1.0e5
    loss_counter = 0

    analytical_norm = torch.norm(AnalyticSolution(A,E,PlotData.data))

    print("**************** START TRAINING ***************\n")
    start_train_time = time.time()

    evaluation_time = 0
    loss_time = 0
    optimizer_time = 0
    backward_time = 0

    while epoch<n_epochs and loss_counter<1000*len(CoordinatesBatchSet) and  stagnancy_counter < 50*len(CoordinatesBatchSet): 

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
                print('*    loss PDE = ', numpy.format_float_scientific( l_pde.item(), precision=4))
                print('*    loss constit = ', numpy.format_float_scientific( l_constit.item(), precision=4))
                print()
                print("     stagnancy counter = ", stagnancy_counter)
                print("     loss counter = ", loss_counter)
                print()

                plot_everything_mixed(A,E,InitialCoordinates_u, InitialCoordinates_du, Coordinates_u_i, Coordinates_du_i,
                                           PlotData, AnalyticSolution, BeamModel_u, BeamModel_du, \
                                           Coord_trajectories, error_pde, error_constit, error2)    

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
                                        TrialCoordinates, PlotCoordinates, n_epochs, BoolCompareNorms, 
                                        MSE, BoolFilterTrainingData,
                                        error_pde, error_constit, error2, Coord_trajectories,
                                        w_pde, w_constit): 

    print()
    print("*************** SECOND STAGE (LBFGS) ***************\n")
    
    optim = torch.optim.LBFGS(list(BeamModel_u.parameters()) + list(BeamModel_du.parameters()),
                    # history_size=5, 
                    # max_iter=15, 
                    # tolerance_grad = 1.0e-9,
                    line_search_fn="strong_wolfe")

    loss_old = error_pde[-1] + error_constit[-1]
    epoch = 0
    stagnancy_counter = 0

    analytical_norm = torch.norm(AnalyticSolution(A,E,PlotCoordinates.data)).data

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
            u_predicted = BeamModel_u(PlotCoordinates) 
            # Stores the coordinates trajectories
            Coordinates_u_i = [BeamModel_u.coordinates[i].data.item() for i in range(len(BeamModel_u.coordinates))]
            Coordinates_du_i = [BeamModel_du.coordinates[i].data.item() for i in range(len(BeamModel_du.coordinates))]
            Coord_trajectories.append(Coordinates_u_i)

            if BoolCompareNorms:
                # Copute and store the L2 error w.r.t. the analytical solution
                error2.append(torch.norm(AnalyticSolution(A,E,PlotCoordinates.data) - u_predicted).data/analytical_norm)

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

    u_predicted = BeamModel_u(PlotCoordinates) 
    du_dx = BeamModel_du(PlotCoordinates) 
    # du_dx = torch.autograd.grad(u_predicted, PlotCoordinates, grad_outputs=torch.ones_like(u_predicted), create_graph=True)[0]
    l2_loss_grad = torch.linalg.vector_norm(AnalyticGradientSolution(A,E,PlotCoordinates.data) - du_dx).data/torch.linalg.vector_norm(AnalyticGradientSolution(A,E,PlotCoordinates.data)).data
    print(f'* Final l2 loss grad : {numpy.format_float_scientific(l2_loss_grad, precision=4)}')

def LBFGS_Stage2_2D(Model_u, Model_du, Mesh_u, Mesh_du, IDs_u, IDs_du, PlotCoordinates, 
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

    TrainCoordinates, TrainIDs_u, TrainIDs_du = GetRealCoord(Model_du, Mesh_du, Cell_ids, Ref_Coord)

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
        loss_decrease = (loss_old - loss_current)/(0.5*(numpy.abs(loss_old) + numpy.abs(loss_current)))
        loss_old = loss_current

        print("     Iter = ",counter," : Loss = ", numpy.format_float_scientific(l.item(), precision=4))
        #print("     ", (Model_du.nodal_values[0][0]).item(),(Model_du.nodal_values[0][1]).item())

        if loss_decrease >= 0 and loss_decrease < 1.0e-7:
            stagnancy_counter = stagnancy_counter +1
        else:
            stagnancy_counter = 0

        epoch = epoch+1

    Plot_all_2D(Model_u, Model_du, IDs_u, IDs_du, PlotCoordinates, [], n_train, "_Final")
    Pplot.Export_Displacement_to_vtk(Mesh_u.name_mesh, Model_u, "final")


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
    loss_decrease = []
    variance_u = []
    first_iteration = True
    stop = False

    min_loss = 100
    no_improvement = 0
    loss_decrease_c = 1.0e-3

    total = w0+w1


    while epoch<n_epochs and stop == False :

        # if epoch%50 == 0:
        #     w1 = torch.randint(1, int(total/4) ,[1])
        #     w0 = total - w1

        for DataSet in CoordinatesBatchSet:

            ##  Training points uniformly sampled in the domain 
            # TrialCoordinates =  DataSet[0]
            # TrialIDs_u = DataSet[1]
            # TrialIDs_du = DataSet[2]

            Cell_ids = DataSet[0]
            Ref_Coord = DataSet[1]

            TrialCoordinates, TrialIDs_u, TrialIDs_du = GetRealCoord(Model_du, Mesh_du, Cell_ids, Ref_Coord)

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

            loss_current = l_pde.item() + l_compat.item()

            # if first_iteration == True:
            #     variance_u.append(torch.mean((u_predicted - torch.zeros_like(u_predicted))**2).detach())
            #     first_iteration == False
            # else:
            #     variance_u.append(torch.mean((u_predicted - u_pred_old)**2).detach()/torch.mean(u_predicted**2).detach())
            # u_pred_old = u_predicted


            loss_decrease.append((loss_old - loss_current)/numpy.abs(0.5*(loss_old + loss_current)))
            loss_old = loss_current

            loss_current = l_pde.item()

            if loss_current < min_loss:
                min_loss = loss_current
                no_improvement = 0
            else:
                no_improvement = no_improvement +1


            start_time = time.time()
            l.backward()

            grads = [[p, p.grad] for p in Model_du.parameters()]

            backward_time += time.time() - start_time

            start_time = time.time()
            optimizer.step()
            optimizer_time += time.time() - start_time

            optimizer.zero_grad()

            #Model_u.Update_Middle_Nodes(Mesh_u)


        if (epoch+1)%100 == 0:
            print("     epoch = ", epoch +1)
            print("     loss_counter = ", loss_counter)
            print("     mean loss PDE = ", numpy.format_float_scientific(numpy.mean(loss[0][-500*n_batches_per_epoch:-1]), precision=4))
            print("     mean loss compatibility = ", numpy.format_float_scientific(numpy.mean(loss[1][-500*n_batches_per_epoch:-1]), precision=4))
            print("     mean loss decrease = ", numpy.format_float_scientific(numpy.mean(loss_decrease[-500*n_batches_per_epoch:-1]), precision=4))
            # print("     displacement variance = ", numpy.format_float_scientific(numpy.mean((variance_u[-500*n_batches_per_epoch:-1])), precision=4))
            print("     w0, w1 : ", w0, w1)
            print()
            # print("     stagnancy_counter = ", stagnancy_counter)   
            print("     no_improvement = ", no_improvement)   
            print("     ...............................")

            # print("coord = ", len(Model_du.coordinates))
            # print("param = ", len(grads))
            # print(grads[20])
            # print(grads[20 + len(Model_du.coordinates)])
            # print(grads[20 + 2*len(Model_du.coordinates)])
            # print(grads[20 + 3*len(Model_du.coordinates)])
            # print()
            # print()


            # if numpy.mean(loss_decrease[-500*n_batches_per_epoch:-1]) > 0 and  numpy.mean(loss_decrease[-500*n_batches_per_epoch:-1]) < 1.0e-4 \
            #         and no_improvement > 20*n_batches_per_epoch:
            #         stop = True


            if numpy.abs(numpy.mean(loss_decrease[-500*n_batches_per_epoch:-1])) < 1.0e-4 \
                    and no_improvement > 20*n_batches_per_epoch:
                    stop = True



        if (epoch+1) % 20 == 0:
            l = Plot_all_2D(Model_u, Model_du, IDs_u, IDs_du , PlotCoordinates, loss, n_train, "_Stage1")
            Pplot.Export_Displacement_to_vtk(Mesh_u.name_mesh, Model_u, epoch+1)
            Pplot.Export_Stress_to_vtk(Mesh_du, Model_du, epoch+1)

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

def Training_2D_Integral(model, optimizer, n_epochs, Mat, config):
    print("**************** START TRAINING ***************\n")
    Loss_vect                           = []                  # Initialise vector of loss values
    time_start                          = time.time()
    epoch                               = 0                   # Epoch counter
    save_time                           = 0                   # Stopwatch for saving the model     
    eval_time                           = 0                   # Stopwatch for evaluating the model     
    back_time                           = 0                   # Stopwatch for backpropagation
    update_time                         = 0                   # Stopwatch for updating the parameters
    model.train()                                             # Training mode
    
    model.Initresults()                                       # Initialise the structure for saving training history
    stagnation                          = False               # Stagnation of loss decay

    while epoch<model.Max_epochs and not stagnation:
        t0 = time.time()
        detJ_new = []
        xg_new = []

        def closure():

            optimizer.zero_grad()
            u_predicted,xg,detJ = model()

            xg_new.append(xg)
            detJ_new.append(detJ)

            detJ_small = detJ[torch.where(torch.abs(detJ)<1.0e-6)]

            if config["solver"]["volume_forces"] == True:
                loss = torch.sum((0.5*InternalEnergy_2D_einsum(u_predicted,xg,Mat.lmbda, Mat.mu)-10*VolumeForcesEnergy_2D(u_predicted,theta = torch.tensor(0*torch.pi/2), rho = 1e-9))*torch.abs(detJ))
            else:
                loss = torch.sum(0.5*InternalEnergy_2D_einsum(u_predicted,xg,Mat.lmbda, Mat.mu)*torch.abs(detJ))
            
            if config["solver"]["regul_term"] == True:
                regul = torch.sum(1/torch.abs(detJ_small))
                loss = loss + regul

            loss.backward(retain_graph=True)
            return loss

        optimizer.step(closure)
        loss = closure()
        tf = time.time()
        # print(f'epoch duration (ms): {1000*(tf-t0)}')

        with torch.no_grad():
            detJ = detJ_new[0]
            xg = xg_new[0]
            model.detJ = detJ

            epoch+=1
            if epoch >1:
                d_loss                  = 2*(torch.abs(loss.data-loss_old))/(torch.abs(loss.data+loss_old))     # Relative loss decay
                loss_old                = loss.data                                                             # Update old loss value
                D_detJ                  = (torch.abs(model.detJ_0) - torch.abs(detJ))/torch.abs(model.detJ_0)   # Relative delat jacobian
                if torch.max(D_detJ)>model.Jacobian_threshold:
                    indices             = torch.nonzero(D_detJ > model.Jacobian_threshold)
                    # Re-initialise future splitted elements' jacobian as base for the newly splitted elements
                    # model.detJ_0[indices] = detJ[indices]
                    Removed_elem_list = []
                    old_generation      = model.elements_generation
                    for i in range(indices.shape[0]):
                        el_id           = indices[i]  
                        if model.elements_generation[el_id.item()]<model.MaxGeneration:
                            model.MaxGeneration_elements=1
                        if el_id.item() not in Removed_elem_list and model.elements_generation[el_id.item()]<model.MaxGeneration:
                            # model.detJ_0[indices] = detJ[indices]
                            el_id = torch.tensor([el_id],dtype=torch.int)
                            new_coordinate = xg[el_id]
                            model.eval()
                            newvalue = model(new_coordinate,el_id) 
                            model.train()
                            Removed_elems = model.SplitElemNonLoc(el_id)                                    # Refine element el_id and remove consequently splitted element from the list of element to split
                            vers = 'New_V2'
                            if vers == 'New_V2':
                                Removed_elems = [e.numpy() for e in Removed_elems]
                            else:
                                Removed_elems[0] = Removed_elems[0].numpy()

                            # Add newly removed elems to list
                            Removed_elem_list += Removed_elems
                            # Update indexes 
                            for j in range(indices.shape[0]):
                                number_elems_above = len([e for e in Removed_elems if e < indices[j].numpy()])
                                indices[j] = indices[j] - number_elems_above
                            # Update indexes of Removed_elem_list
                            for j in range(len(Removed_elem_list)):
                                number_elems_above = len([e for e in Removed_elems if e < Removed_elem_list[j]])
                                Removed_elem_list[j] = Removed_elem_list[j] - number_elems_above

                            if optimizer.__class__.__name__ == "Adam":
                                # # Add newly removed elems to list
                                # Removed_elem_list += Removed_elems
                                optimizer.add_param_group({'params': model.coordinates[-3:]})
                                optimizer.add_param_group({'params': model.nodal_values[0][-3:]})
                                optimizer.add_param_group({'params': model.nodal_values[1][-3:]})
                            elif optimizer.__class__.__name__ == "LBFGS":
                                optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn="strong_wolfe")
                    # model.Freeze_Mesh()
                if d_loss < model.loss_decrease_c:
                    stagnation = True
            else:
                loss_old = loss.item()
                detJ_0 = detJ
                model.detJ_0 = detJ
                model.detJ = detJ
            Loss_vect.append(loss.item())

        if optimizer.__class__.__name__ == "Adam":
            if (epoch+1) % 50 == 0 or epoch ==1 or epoch==model.Max_epochs or stagnation:
                model.StoreResults()
                print(f'epoch {epoch+1} loss = {numpy.format_float_scientific(loss.item(), precision=4)}')
        elif optimizer.__class__.__name__ == "LBFGS":
            if (epoch+1) % config['postprocess']['ModularEpochsPrint'] == 0 or epoch ==1 or epoch==model.Max_epochs or stagnation:
                model.StoreResults()
                print(f'epoch {epoch+1} loss = {numpy.format_float_scientific(loss.item(), precision=4)}')

    time_stop = time.time()
    # print("*************** END OF TRAINING ***************\n")
    print("*************** END FIRST PHASE ***************\n")
    print(f'* Training time: {time_stop-time_start}s')
    # print(f'* Saving time: {save_time}s')
    # print(f'* Evaluation time: {eval_time}s')
    # print(f'* Backward time: {back_time}s')
    # print(f'* Update time: {update_time}s')
    print(f'* Average epoch time: {(time_stop-time_start)/(epoch+1)}s')

    return Loss_vect, (time_stop-time_start)
    
def Training_2D_NeuROM(model, config, optimizer,Mat):
    n_epochs = model.Max_epochs
    # In the parameters space
    Training_para_coordinates_1         = torch.linspace(
                                                config["parameters"]["para_1_min"],
                                                config["parameters"]["para_1_max"],
                                                5*config["parameters"]["N_para_1"], 
                                                dtype=torch.float32, 
                                                requires_grad=True
                                                )

    Training_para_coordinates_1         = Training_para_coordinates_1[:,None]

    Training_para_coordinates_2         = torch.linspace(
                                                config["parameters"]["para_2_min"],
                                                config["parameters"]["para_2_max"],
                                                5*config["parameters"]["N_para_2"], 
                                                dtype=torch.float32, 
                                                requires_grad=True
                                                )

    Training_para_coordinates_2         = Training_para_coordinates_2[:,None]  

    match config["solver"]["N_ExtraCoordinates"]:
        case 2:
            Training_para_coordinates_list  = nn.ParameterList(
                                                            (Training_para_coordinates_1,
                                                            Training_para_coordinates_2))
        case 1:
            Training_para_coordinates_list  = [Training_para_coordinates_1]

    time_start                          = time.time()
    epoch                               = 0 
    Loss_vect                           = []
    stagnation                          = False
    while epoch<model.Max_epochs and not stagnation:
        match config["solver"]["N_ExtraCoordinates"]:
            case 1:
                loss = InternalEnergy_2D_einsum_para(model,Mat.lmbda, Mat.mu,Training_para_coordinates_list)
            case 2:
                loss = InternalEnergy_2D_einsum_Bipara(model,Mat.lmbda, Mat.mu,Training_para_coordinates_list)

        loss.backward()
        # update weights
        optimizer.step()
        # zero the gradients after updating
        optimizer.zero_grad()
        with torch.no_grad():
            Loss_vect.append(loss.item())
            epoch                       +=1
            if epoch >1:
                d_loss                  = 2*(torch.abs(loss.data-loss_old))/(torch.abs(loss.data+loss_old))
                loss_old                = loss.data
                if d_loss < model.loss_decrease_c:
                    stagnation          = True
                    # stagnation = False
            else:
                loss_old                = loss.item()
            if (epoch+1) % 50 == 0 or epoch ==1 or epoch==model.Max_epochs or stagnation:
                print(f'epoch {epoch+1} loss = {numpy.format_float_scientific(loss.item(), precision=4)}')

    time_stop = time.time()
    print("*************** END OF TRAINING ***************\n")
    # print("*************** END FIRST PHASE ***************\n")
    print(f'* Training time: {time_stop-time_start}s')
    model.training_recap = {"Loss_vect":Loss_vect,
                        # "L2_error":L2_error,
                        "training_time":(time_stop-time_start),
                        # "Mode_vect":Modes_vect,
                        # "Loss_decrease_vect":Loss_decrease_vect
                        }
    return

def Training_2D_Residual(model, model_test, optimizer, n_epochs,List_elems,Mat):
    
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
    List_Dofs_free = (model_test.values[:,0] == 1).nonzero(as_tuple=True)[0]
    u_predicted_star_list_x = []
    u_predicted_star_list_y = []
    eps_predicted_star_list_x = []
    eps_predicted_star_list_y = []
    _,xg,detJ = model()
    model.eval()
    model_test.eval()
    # Pre compute the teste displacement and test strains
    for dof in List_Dofs_free:
        model_test.values = 0*model_test.values
        model_test.values[dof,:] = torch.tensor([1., 0.])
        model_test.SetBCs(len(model_test.ListOfDirichletsBCsValues)*[0])
        u_pred = model_test(xg,List_elems)
        u_predicted_star_list_x.append(u_pred.detach())
        eps_predicted_star_list_x.append(Strain_sqrt(u_pred,xg).detach())
        model_test.values = 0*model_test.values
        model_test.values[dof,:] = torch.tensor([0., 1.])
        model_test.SetBCs(len(model_test.ListOfDirichletsBCsValues)*[0])
        u_pred = model_test(xg,List_elems)
        u_predicted_star_list_y.append(u_pred.detach())
        eps_predicted_star_list_y.append(Strain_sqrt(u_pred,xg).detach())


    while epoch<model.Max_epochs and not stagnation:
        # Compute loss
        loss_time_start = time.time()
        u_predicted = model(xg,List_elems)
        eps = Strain_sqrt(u_predicted,xg)
        loss = 0
        for i in range(List_Dofs_free.shape[0]):
            u_predicted_star = u_predicted_star_list_x[i]
            eps_predicted_star = eps_predicted_star_list_x[i]
            # loss += torch.abs(torch.sum((InternalResidual_precomputed(eps,eps_predicted_star,Mat.lmbda, Mat.mu)-
            #                     1*VolumeForcesEnergy_2D(u_predicted_star,theta = torch.tensor(0*torch.pi/2), rho = 1e-9))
            #                     *torch.abs(detJ)))
            loss += torch.pow(torch.sum((InternalResidual_precomputed(eps,eps_predicted_star,Mat.lmbda, Mat.mu)-
                                1*VolumeForcesEnergy_2D(u_predicted_star,theta = torch.tensor(0*torch.pi/2), rho = 1e-9))
                                *torch.abs(detJ)),2)            
            u_predicted_star = u_predicted_star_list_y[i]
            eps_predicted_star = eps_predicted_star_list_y[i]
            # loss +=  torch.abs(torch.sum((InternalResidual_precomputed(eps,eps_predicted_star,Mat.lmbda, Mat.mu)-
            #                     1*VolumeForcesEnergy_2D(u_predicted_star,theta = torch.tensor(0*torch.pi/2), rho = 1e-9))
            #                     *torch.abs(detJ)))
            loss +=  torch.pow(torch.sum((InternalResidual_precomputed(eps,eps_predicted_star,Mat.lmbda, Mat.mu)-
                                1*VolumeForcesEnergy_2D(u_predicted_star,theta = torch.tensor(0*torch.pi/2), rho = 1e-9))
                                *torch.abs(detJ)),2)
        # loss = torch.abs(loss)
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
                if d_loss < model.loss_decrease_c:
                    stagnation = False
            else:
                loss_old = loss.item()
            Loss_vect.append(loss.item())
        if (epoch+1) % 1 == 0 or epoch ==1 or epoch==model.Max_epochs or stagnation:
            print(f'epoch {epoch+1} loss = {numpy.format_float_scientific(loss.item(), precision=4)}')

            # meshBeam = meshio.read('geometries/Square_order_2_0.312_0.625.vtk')
            # u = torch.stack([torch.cat(u_x),torch.cat(u_y)],dim=1)
            # coordinates = [coord for coord in model.coordinates]
            # coordinates = torch.cat(coordinates,dim=0)

            # sol = meshio.Mesh(coordinates.data, {"triangle6":meshBeam.cells_dict["triangle6"]},
            #                     point_data={"U":u.data})
            # sol.write('Results/Paraview/Displacement_'+str(epoch+1)+'.vtk')



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

def Training_2D_Residual_LBFGS(model, model_test, n_epochs,List_elems,Mat):
    optimizer = torch.optim.LBFGS(list(model.parameters()),
                    history_size=5, 
                    max_iter=15, 
                    tolerance_grad = 1.0e-9,
                    line_search_fn="strong_wolfe")    # Initialise vector of loss values
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
    List_Dofs_free = (model_test.values[:,0] == 1).nonzero(as_tuple=True)[0]
    u_predicted_star_list_x = []
    u_predicted_star_list_y = []
    eps_predicted_star_list_x = []
    eps_predicted_star_list_y = []
    _,xg,detJ = model()
    model.eval()
    model_test.eval()
    # Pre compute the test displacements and test strains
    for dof in List_Dofs_free:
        model_test.values = 0*model_test.values
        model_test.values[dof,:] = torch.tensor([1., 0.])
        model_test.SetBCs(len(model_test.ListOfDirichletsBCsValues)*[0])
        u_pred = model_test(xg,List_elems)
        u_predicted_star_list_x.append(u_pred.detach())
        eps_predicted_star_list_x.append(Strain_sqrt(u_pred,xg).detach())
        model_test.values = 0*model_test.values
        model_test.values[dof,:] = torch.tensor([0., 1.])
        model_test.SetBCs(len(model_test.ListOfDirichletsBCsValues)*[0])
        u_pred = model_test(xg,List_elems)
        u_predicted_star_list_y.append(u_pred.detach())
        eps_predicted_star_list_y.append(Strain_sqrt(u_pred,xg).detach())


    while epoch<model.Max_epochs and not stagnation:
        # Compute loss
        loss_time_start = time.time()



        def closure():
            optimizer.zero_grad()
            loss = 0
            u_predicted = model(xg,List_elems)
            eps = Strain_sqrt(u_predicted,xg)
            for i in range(List_Dofs_free.shape[0]):
                u_predicted_star = u_predicted_star_list_x[i]
                eps_predicted_star = eps_predicted_star_list_x[i]
                # loss += torch.abs(torch.sum((InternalResidual_precomputed(eps,eps_predicted_star,Mat.lmbda, Mat.mu)-
                #                     1*VolumeForcesEnergy_2D(u_predicted_star,theta = torch.tensor(0*torch.pi/2), rho = 1e-9))
                #                     *torch.abs(detJ)))
                loss += torch.pow(torch.sum((InternalResidual_precomputed(eps,eps_predicted_star,Mat.lmbda, Mat.mu)-
                                    1*VolumeForcesEnergy_2D(u_predicted_star,theta = torch.tensor(0*torch.pi/2), rho = 1e-9))
                                    *torch.abs(detJ)),2)                
                u_predicted_star = u_predicted_star_list_y[i]
                eps_predicted_star = eps_predicted_star_list_y[i]
                # loss +=  torch.abs(torch.sum((InternalResidual_precomputed(eps,eps_predicted_star,Mat.lmbda, Mat.mu)-
                #                     1*VolumeForcesEnergy_2D(u_predicted_star,theta = torch.tensor(0*torch.pi/2), rho = 1e-9))
                #                     *torch.abs(detJ)))
                loss +=  torch.pow(torch.sum((InternalResidual_precomputed(eps,eps_predicted_star,Mat.lmbda, Mat.mu)-
                                    1*VolumeForcesEnergy_2D(u_predicted_star,theta = torch.tensor(0*torch.pi/2), rho = 1e-9))
                                    *torch.abs(detJ)),2)                
            loss.backward(retain_graph=True)
            return loss

        optimizer.step(closure)
        loss = closure()


        # zero the gradients after updating
        # optimizer.zero_grad()
        with torch.no_grad():
            epoch+=1
            if epoch >1:
                d_loss = 2*(torch.abs(loss.data-loss_old))/(torch.abs(loss.data+loss_old))
                loss_old = loss.data
                if d_loss < model.loss_decrease_c:
                    stagnation = True
            else:
                loss_old = loss.item()
            Loss_vect.append(loss.item())
        if (epoch+1) % 1 == 0 or epoch ==1 or epoch==model.Max_epochs or stagnation:
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

def Training_2D_FEM(model, config, Mat):
    n_epochs = config["training"]["n_epochs"]
    n_refinement        = 0                                 # Initialise the refinement level
    stagnation          = False                             # Stagnation flag
    Loss_tot            = []                                # Vector of loss values throught training
    Duration_tot        = 0                                 # Stopwatch of the training
    U_interm_tot        = []                                # List of displacement solutions history throught training
    Gen_interm_tot      = []                                # List of the elements' generation history throught training
    X_interm_tot        = []                                # List nodal position history throught training
    Connectivity_tot    = []                                # List connectivity table history through training
    d_eps_max_vect      = []                                # List of reltive delta of maximum strain through training
    eps_max_vect        = []                                # List of maximum strain through training
    detJ_tot            = []                                # List of history of detJ through training
    detJ_current_tot    = []
    MaxElemSize         = config["interpolation"]["MaxElemSize2D"] # Initial value of max elem size from config file
    import meshio
    while n_refinement < config["training"]["multiscl_max_refinment"] and not stagnation:
        print(f"* Refinement level: {n_refinement}\n")
        n_refinement        +=1

        if config["training"]["optimizer"] == "adam":
            optimizer           = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
        elif config["training"]["optimizer"] == "lbfgs":
            optimizer           = torch.optim.LBFGS(model.parameters(), line_search_fn="strong_wolfe")


        # Call the mono-scale training routine
        match config["solver"]["TrainingStrategy"]:
            case "Integral":
                Loss_vect, Duration = Training_2D_Integral(model, optimizer, n_epochs,Mat, config)
            case "Residual":
                Loss_vect, Duration = Training_2D_Integral(model, optimizer, n_epochs,Mat, config)
            case "Mixed":
                Loss_vect, Duration = Training_2D_Integral(model, optimizer, n_epochs,Mat, config) 
        # Update the training history
        Loss_tot            += Loss_vect
        Duration_tot        += Duration
        U_interm_tot        += model.U_interm
        Gen_interm_tot      += model.G_interm
        detJ_tot            += model.Jacobian_interm
        X_interm_tot        += model.X_interm
        Connectivity_tot    += model.Connectivity_interm
        detJ_current_tot    += model.Jacobian_current_interm

        if config["training"]["multiscl_max_refinment"] >1:
            # Compute maximum strain 
            _,xg,detJ            = model()
            model.eval()
            List_elems           = torch.arange(0,model.NElem,dtype=torch.int)

            # if model.float_config.device != torch.device("mps"):
            #     device = model.float_config.device
            #     model.to(torch.device("mps"))
            #     xg = xg.to(torch.device("mps"))
            #     eps              =  Strain(model(xg, List_elems),xg)
            #     model.to(device)
            # else:
            eps              =  Strain(model(xg, List_elems),xg)

            max_eps              = torch.max(eps)

            if n_refinement > 1:
                d_eps_max        = 2*torch.abs(max_eps-max_eps_old)/(max_eps_old+max_eps_old)
                d_eps_max_vect.append(d_eps_max.data)
                eps_max_vect.append(max_eps.data)
                max_eps_old      = max_eps
                if d_eps_max < config["training"]["d_eps_max_threshold"]:
                    stagnation   = True
            else:
                max_eps_old      = max_eps
        if n_refinement < config["training"]["multiscl_max_refinment"] and not stagnation:

            MaxElemSize      = MaxElemSize/config["training"]["multiscl_refinment_cf"]  # Update max elem size
            Mesh_object_fine = pre.Mesh( config["geometry"]["Name"],                    # Create the mesh object
                                         MaxElemSize, 
                                         config["interpolation"]["order"], 
                                         config["interpolation"]["dimension"], 
                                         welcome = False)
            Mesh_object_fine.AddBorders( config["Borders"]["Borders"])
            Mesh_object_fine.AddBCs(     config["geometry"]["Volume_element"],
                                    [],
                                    config["DirichletDictionryList"])                   
            Mesh_object_fine.MeshGeo()                                                       
            Mesh_object_fine.ReadMesh()   
            Mesh_object_fine.ExportMeshVtk()
            List_elems          = torch.arange(0,Mesh_object_fine.NElem,dtype=torch.int)
            model_2 = MeshNN_2D(Mesh_object_fine, 2)                                    # Create the associated model (with 2 components)
            vers = 'New_V2'
            match vers:
                case 'old':
                    # Update model's mesh
                    model.mesh.Nodes    = [[i+1,model.coordinates[i][0][0].item(),model.coordinates[i][0][1].item(),0] for i in range(len(model.coordinates))]
                case 'New_V2':
                    coordinates_all = torch.ones_like(model.coordinates_all)
                    coordinates_all[model.coord_free] = model.coordinates['free']
                    coordinates_all[~model.coord_free] = model.coordinates['imposed']
                    Nodes = torch.hstack([torch.linspace(1,coordinates_all.shape[0],coordinates_all.shape[0], dtype = coordinates_all.dtype, device = coordinates_all.device)[:,None],
                                          coordinates_all])
                    Nodes = torch.hstack([Nodes,torch.zeros(Nodes.shape[0],1, dtype = Nodes.dtype, device = Nodes.device)])
                    model.mesh.Nodes = Nodes.detach().cpu().numpy()
            model.mesh.Connectivity = model.connectivity
            model.mesh.ExportMeshVtk(flag_update = True)
            if model_2.float_config.dtype != model.float_config.dtype:
                model_2.to(model.float_config.dtype)
                print(f'Finer model passed to dtype {model.float_config.dtype}')
            if model_2.float_config.device != model.float_config.device:
                model_2.to(model.float_config.device)
                print(f'Finer model passed to device {model.float_config.device}')
            model_2.Init_from_previous(model)                                           # Initialise fine model with coarse one
            model = model_2                                                             # model is now the fine 
            model.UnFreeze_FEM()
            model.Freeze_Mesh()
            if not config["solver"]["FrozenMesh"]:
                model.UnFreeze_Mesh()
            model.train()


            model.RefinementParameters( MaxGeneration = config["training"]["h_adapt_MaxGeneration"], 
                                        Jacobian_threshold = config["training"]["h_adapt_J_thrshld"])

            model.TrainingParameters(   loss_decrease_c = config["training"]["loss_decrease_c"], 
                                    Max_epochs = config["training"]["n_epochs"], 
                                    learning_rate = config["training"]["learning_rate"])
        else:
            model.train()
            model.training_recap = {"Loss_tot":Loss_tot,
                                    "Duration_tot":Duration_tot,
                                    "U_interm_tot":U_interm_tot,
                                    "Gen_interm_tot":Gen_interm_tot,
                                    "X_interm_tot":X_interm_tot,
                                    "Connectivity_tot":Connectivity_tot,
                                    "d_eps_max_vect":d_eps_max_vect,
                                    "eps_max_vect":eps_max_vect,
                                    "detJ_tot":detJ_tot,
                                    "detJ_current_tot":detJ_current_tot}
            print("***************** END TRAINING ****************\n")
            print(f'* Training time: {Duration_tot}s')                                    
    return model 

def Training_NeuROM_multi_level(model, config, Mat = 'NaN'):
    n_refinement                = 0
    MaxElemSize                 = pre.ElementSize(
                                dimension     = config["interpolation"]["dimension"],
                                L             = config["geometry"]["L"],
                                order         = config["interpolation"]["order"],
                                np            = config["interpolation"]["np"],
                                MaxElemSize2D = config["interpolation"]["MaxElemSize2D"]
                            )
    Excluded = []
    try:
        loss_init               = model.training_recap["Loss_vect"]         # Test if model.training_recap exists
    except:
        model.training_recap    = {"Loss_vect":[],
                                "L2_error":[],                              # Init L2 error
                                "training_time":0,                          # Init Training duration
                                "Mode_vect":[],                             # Size ROB
                                "Loss_decrease_vect":[]                     # Init loss decrease rate
                                }
    import meshio
    while n_refinement < config["training"]["multiscl_max_refinment"]:
        print(f"* Refinement level: {n_refinement}\n")
        n_refinement            +=1
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config["training"]["learning_rate"])
        match config["interpolation"]["dimension"]:
            case 1:
                Training_NeuROM(model,config,optimizer)                 # First stage of training (ADAM)
                Training_NeuROM_FinalStageLBFGS(model,config)           # Second stage of training (LBFGS)
            case 2:
                Training_NeuROM(model, config, optimizer, Mat)          # First stage of training (ADAM)
                Training_NeuROM_FinalStageLBFGS(model,config, Mat)      # Second stage of training (LBFGS)

        if n_refinement < config["training"]["multiscl_max_refinment"]:
            MaxElemSize      = MaxElemSize/config["training"]["multiscl_refinment_cf"]  # Update max elem size
            Mesh_object_fine = pre.Mesh( 
                                    config["geometry"]["Name"],                 # Create the mesh object
                                    MaxElemSize, 
                                    config["interpolation"]["order"], 
                                    config["interpolation"]["dimension"],
                                    welcome = False
                            )
            Mesh_object_fine.AddBorders(config["Borders"]["Borders"])
            Mesh_object_fine.AddBCs(                                                         # Include Boundary physical domains infos (BCs+volume)
                                            config["geometry"]["Volume_element"],
                                            Excluded,
                                            config["DirichletDictionryList"]
                                )                   
            Mesh_object_fine.MeshGeo()                                                       # Mesh the .geo file if .msh does not exist
            Mesh_object_fine.ReadMesh()                                                      # Parse the .msh file
            match config["interpolation"]["dimension"]:
                case 1:
                    if config["solver"]["IntegralMethod"] == "Gaussian_quad":
                        Mesh_object_fine.ExportMeshVtk1D()
                case 2:
                    Mesh_object_fine.ExportMeshVtk()
            if config["interpolation"]["dimension"] ==1 and config["solver"]["IntegralMethod"] == "Trapezoidal":
                Mesh_object_fine.AssemblyMatrix() 
            match config["solver"]["N_ExtraCoordinates"]:
                case 2:
                    ParameterHypercube = torch.tensor([ [   config["parameters"]["para_1_min"],
                                                            config["parameters"]["para_1_max"],
                                                            config["parameters"]["N_para_1"]],
                                                        [   config["parameters"]["para_2_min"],
                                                            config["parameters"]["para_2_max"],
                                                            config["parameters"]["N_para_2"]]])
                case 1:
                    ParameterHypercube = torch.tensor([[    config["parameters"]["para_1_min"],
                                                            config["parameters"]["para_1_max"],
                                                            config["parameters"]["N_para_1"]]])

            model_2 = NeuROM(                                                         # Build the surrogate (reduced-order) model
                                                        Mesh_object_fine, 
                                                        ParameterHypercube, 
                                                        config,
                                                        config["solver"]["n_modes_ini"],
                                                        config["solver"]["n_modes_max"]
                            )
            model.eval()
            if model_2.float_config.dtype != model.float_config.dtype:
                model_2.to(model.float_config.dtype)
                print(f'Finer model passed to dtype {model.float_config.dtype}')
            if model_2.float_config.device != model.float_config.device:
                model_2.to(model.float_config.device)
                print(f'Finer model passed to device {model.float_config.device}')
            model_2.Init_from_previous(model, Model_provided=True)                                           # Initialise fine model with coarse one

            # model_2.train()
            model_2.training_recap = model.training_recap
            model = model_2
            model.UnfreezeTruncated()
            model.Freeze_Mesh()                                                         # Set space mesh cordinates as untrainable
            model.Freeze_MeshPara()  
            if not config["solver"]["FrozenMesh"]:
                model.UnFreeze_Mesh()                                               # Set space mesh cordinates as trainable
            if not config["solver"]["FrozenParaMesh"]:
                model.UnFreeze_MeshPara()                                           # Set parameters mesh cordinates as trainable


            model.TrainingParameters(   loss_decrease_c = config["training"]["loss_decrease_c"], 
                                        Max_epochs = config["training"]["n_epochs"], 
                                        learning_rate = config["training"]["learning_rate"])
            model.train()
    try:
        return model, Mesh_object_fine
    except:
        return model, Mesh_object

def Training_1D_FEM_LBFGS(model, config, Mat, model_test = []):

    n_epochs = config["training"]["n_epochs_2"]
    A = config["geometry"]["A"]
    E = config["material"]["E"]
    L = config["geometry"]["L"]
    Show_trajectories = config["postprocess"]["Show_Trajectories"]

    max_stagnation_counter = config["training"]["Stagnation_counter_2"]
    stagnation_threshold = config["training"]["Stagnation_threshold_2"]

    if config["solver"]["IntegralMethod"] == "Trapezoidal":
        n_points = config["training"]["Points_per_element"]

        TrialCoordinates = torch.tensor([[i] for i in torch.linspace(0,L,n_points*model.NElem)], dtype=torch.float64, requires_grad=True)
    
    if config["solver"]["TrainingStrategy"]=="Mixed":

        if config["solver"]["IntegralMethod"] == "None":
            print("Training data")

        elif config["solver"]["IntegralMethod"] == "Gaussian_quad":
            model_test.Freeze_FEM()
            model_test.Freeze_Mesh()
            List_elems = torch.arange(0,model.NElem,dtype=torch.int)

    Coord_trajectories = []
    error = []

    epoch = 0
    eval_time = 0
    back_time = 0
    update_time = 0

    loss_old = 1.0
    stagnancy_counter = 0

    InitialCoordinates = [model.coordinates[i].data.item() for i in range(len(model.coordinates))]
    Coordinates = [model.coordinates[i].data.item() for i in range(len(model.coordinates))]
    
    
    optimizer = torch.optim.LBFGS(model.parameters(),
                    line_search_fn="strong_wolfe")

    Coordinates_i = [model.coordinates[i].data.item() for i in range(len(model.coordinates))]
    Coord_trajectories.append(Coordinates_i)

    while epoch<n_epochs and stagnancy_counter < max_stagnation_counter:

        coord_old = [model.coordinates[i].data.item() for i in range(len(model.coordinates))]
        # Compute loss

        if config["solver"]["TrainingStrategy"]=="Integral":
            if config["solver"]["IntegralMethod"] == "Trapezoidal":
                def closure():
                    optimizer.zero_grad()
                    u_predicted = model(TrialCoordinates) 
                    l = PotentialEnergyVectorised(A,E,u_predicted,TrialCoordinates,RHS(TrialCoordinates))
                    l.backward()
                    return l

            elif config["solver"]["IntegralMethod"] == "Gaussian_quad":
                def closure():
                    optimizer.zero_grad()

                    model.train()
                    loss_time_start = time.time()
                    u_predicted,xg,detJ = model()

                    loss = torch.sum(InternalEnergy_1D(u_predicted,xg,A, E)*torch.abs(detJ))

                    loss.backward()
                    return loss

        if config["solver"]["TrainingStrategy"]=="Mixed":

            if config["solver"]["IntegralMethod"] == "Gaussian_quad":
                def closure():
                    loss = 0
                    optimizer.zero_grad()
                    model_test.train()
                    model.eval()

                    for node in range(model.NElem-1):

                        model_test.SetFixedValues(node,1)
                        model_test.Freeze_FEM()

                        u_predicted_test, xg, detJ = model_test()
                        du_test_dx = torch.autograd.grad(u_predicted_test, xg, grad_outputs=torch.ones_like(u_predicted_test), create_graph=True)[0]
                        list_elem = List_elems.repeat(u_predicted_test.shape[1],1)

                        nonzeros = torch.where(u_predicted_test!=0)

                        xg = xg[nonzeros]
                        detJ = detJ[nonzeros]
                        u_predicted_test = u_predicted_test[nonzeros]
                        du_test_dx = du_test_dx[nonzeros]
                        list_elem = torch.transpose(list_elem,0,1)[nonzeros]

                        u_predicted = model(xg, list_elem)[:,0]

                        loss = loss + torch.pow(torch.sum(WeakEquilibrium_1D(u_predicted,u_predicted_test,du_test_dx,xg,A, E)*torch.abs(detJ)),2)

                    loss.backward()
                    return loss

            if config["solver"]["IntegralMethod"] == "None":

                def closure():
                    print("Mixed")

                    return loss

        model.train()

        optimizer.step(closure)
        loss = closure()
    
        with torch.no_grad():
            if config["solver"]["FrozenMesh"] == False:
                correction = True
                while correction:
                    correction = Collision_Check(model, coord_old, 1.0e-6)
                    if correction:
                        print("Correction")

        loss = closure()
        loss_current = loss.item()

        epoch = epoch+1

        error.append(loss.item())

        loss_decrease = (loss_old - loss_current)/numpy.abs(loss_old)
        loss_old = loss_current

        if loss_decrease >= 0 and loss_decrease < stagnation_threshold:
            stagnancy_counter = stagnancy_counter +1
        else:
            stagnancy_counter = 0

        with torch.no_grad():
            Coordinates_i = [model.coordinates[i].data.item() for i in range(len(model.coordinates))]
            Coord_trajectories.append(Coordinates_i)

        if epoch%1 == 0:
            print("epoch = ", epoch)
            print("     loss = ", loss_current)
            print("     loss_decrease = ", loss_decrease)


    print(f'* Final training loss: {numpy.format_float_scientific( error[-1], precision=4)}')

    # Pplot.Plot_Compare_Loss2l2norm(error,[],'Loss_Comaprison')
    if Show_trajectories:
        Pplot.PlotTrajectories(Coord_trajectories,'Trajectories',Show_trajectories)

    return model

def Training_1D_FEM_Gradient_Descent(model, config, Mat, model_test = []):

    n_epochs = config["training"]["n_epochs_1"]
    A = config["geometry"]["A"]
    E = config["material"]["E"]
    L = config["geometry"]["L"]

    learning_rate = config["training"]["learning_rate"]

    max_stagnation_counter = config["training"]["Stagnation_counter_1"]
    max_loss_counter = config["training"]["Loss_counter"]

    stagnation_threshold = config["training"]["Stagnation_threshold_1"]

    if config["solver"]["IntegralMethod"] == "Trapezoidal":
        n_points = config["training"]["Points_per_element"]
        TrialCoordinates = torch.tensor([[i] for i in torch.linspace(0,L,n_points*model.NElem)], dtype=torch.float64, requires_grad=True)

    if config["solver"]["TrainingStrategy"]=="Mixed":
        model_test.Freeze_FEM()
        model_test.Freeze_Mesh()
        List_elems = torch.arange(0,model.NElem,dtype=torch.int)

    Coord_trajectories = []
    error = []

    epoch = 0
    eval_time = 0
    back_time = 0
    update_time = 0

    loss_min = 1.0e3
    loss_counter = 0

    loss_old = 1.0
    stagnancy_counter = 0

    InitialCoordinates = [model.coordinates[i].data.item() for i in range(len(model.coordinates))]
    Coordinates = [model.coordinates[i].data.item() for i in range(len(model.coordinates))]
    
    
    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

    while epoch<n_epochs and stagnancy_counter < max_stagnation_counter:# and loss_counter < max_loss_counter:

        coord_old = [model.coordinates[i].data.item() for i in range(len(model.coordinates))]
        # Compute loss

        if config["solver"]["TrainingStrategy"]=="Integral":

            if config["solver"]["IntegralMethod"] == "Trapezoidal":
                def closure():
                    optimizer.zero_grad()
                    u_predicted = model(TrialCoordinates) 
                    l = PotentialEnergyVectorised(A,E,u_predicted,TrialCoordinates,RHS(TrialCoordinates))
                    l.backward()
                    return l

            elif config["solver"]["IntegralMethod"] == "Gaussian_quad":
                def closure():
                    optimizer.zero_grad()

                    model.train()
                    loss_time_start = time.time()
                    u_predicted,xg,detJ = model()

                    loss = torch.sum(InternalEnergy_1D(u_predicted,xg,A, E)*torch.abs(detJ))

                    loss.backward()
                    return loss

        if config["solver"]["TrainingStrategy"]=="Mixed":

            if config["solver"]["IntegralMethod"] == "Gaussian_quad":
                def closure():
                    loss = 0
                    optimizer.zero_grad()
                    model_test.train()
                    model.eval()

                    for node in range(model.NElem-1):

                        model_test.SetFixedValues(node,1)
                        model_test.Freeze_FEM()

                        u_predicted_test, xg, detJ = model_test()
                        du_test_dx = torch.autograd.grad(u_predicted_test, xg, grad_outputs=torch.ones_like(u_predicted_test), create_graph=True)[0]
                        list_elem = List_elems.repeat(u_predicted_test.shape[1],1)

                        nonzeros = torch.where(u_predicted_test!=0)

                        xg = xg[nonzeros]
                        detJ = detJ[nonzeros]
                        u_predicted_test = u_predicted_test[nonzeros]
                        du_test_dx = du_test_dx[nonzeros]
                        list_elem = torch.transpose(list_elem,0,1)[nonzeros]

                        u_predicted = model(xg, list_elem)[:,0]

                        loss = loss + torch.pow(torch.sum(WeakEquilibrium_1D(u_predicted,u_predicted_test,du_test_dx,xg,A, E)*torch.abs(detJ)),2)

                    loss.backward()
                    return loss

        model.train()

        optimizer.step(closure)

        loss = closure()
    
        with torch.no_grad():

            if config["solver"]["FrozenMesh"] == False:
                Collision_Check(model, coord_old, 1.0e-6)

            if loss_min > loss:
                loss_min = loss
                loss_counter = 0
            else:
                loss_counter += 1

        loss = closure()
        loss_current = loss.item()

        epoch = epoch+1

        error.append(loss.item())

        loss_decrease = (loss_old - loss_current)/numpy.abs(loss_old)
        loss_old = loss_current

        if loss_decrease >= 0 and loss_decrease < stagnation_threshold:
            stagnancy_counter = stagnancy_counter +1
        else:
            stagnancy_counter = 0

        with torch.no_grad():
            Coordinates_i = [model.coordinates[i].data.item() for i in range(len(model.coordinates))]
            Coord_trajectories.append(Coordinates_i)

        if epoch%10 == 0:
            print("epoch = ", epoch)
            print("     loss = ", loss_current)
            print("     loss_decrease = ", loss_decrease)

    print(f'* Final training loss: {numpy.format_float_scientific( error[-1], precision=4)}')

    Pplot.Plot_Compare_Loss2l2norm(error,[],'Loss_Comaprison')
    Pplot.PlotTrajectories(Coord_trajectories,'Trajectories')

    return model

def Training_1D_Mixed_LBFGS(model_u, model_du, config, Mat):

    n_epochs = config["training"]["n_epochs"]
    A = config["geometry"]["A"]
    E = config["material"]["E"]
    L = config["geometry"]["L"]

    max_stagnation_counter = config["training"]["Stagnation_counter"]
    stagnation_threshold = config["training"]["Stagnation_threshold"]

    w_pde = config["training"]["w_pde"]
    w_constit = config["training"]["w_constit"]

    n_points = config["training"]["Points_per_element"]

    TrialCoordinates = torch.tensor([[i] for i in torch.linspace(0,L,n_points*model_u.NElem)], dtype=torch.float64, requires_grad=True)


    Coord_trajectories = []
    error = []

    epoch = 0
    eval_time = 0
    back_time = 0
    update_time = 0

    loss_old = 1.0
    stagnancy_counter = 0

    InitialCoordinates = [model_u.coordinates[i].data.item() for i in range(len(model_u.coordinates))]
    
    
    optimizer = torch.optim.LBFGS(list(model_u.parameters())+list(model_du.parameters()),
                    line_search_fn="strong_wolfe")

    Coordinates_i = [model_u.coordinates[i].data.item() for i in range(len(model_u.coordinates))]
    Coord_trajectories.append(Coordinates_i)

    while epoch<n_epochs and stagnancy_counter < max_stagnation_counter:


        def closure():
            optimizer.zero_grad()

            u_predicted = model_u(TrialCoordinates) 
            du_predicted = model_du(TrialCoordinates) 

            l_pde, l_constit  = MixedFormulation_Loss(A, E, u_predicted, du_predicted, TrialCoordinates, RHS(TrialCoordinates))
            l =  w_pde*l_pde + w_constit*l_constit

            l.backward()
            return l

      
        model_u.train()
        model_du.train()

        optimizer.step(closure)
        loss = closure()
    
        with torch.no_grad():
            if config["solver"]["FrozenMesh"] == False:
                if config["solver"]["IntegralMethod"] == "Trapezoidal":
                    Collision_Check(model, coord_old, 1.0e-6)

        loss = closure()
        loss_current = loss.item()

        epoch = epoch+1

        error.append(loss.item())

        loss_decrease = (loss_old - loss_current)/numpy.abs(loss_old)
        loss_old = loss_current

        if loss_decrease >= 0 and loss_decrease < stagnation_threshold:
            stagnancy_counter = stagnancy_counter +1
        else:
            stagnancy_counter = 0

        with torch.no_grad():
            Coordinates_i = [model_u.coordinates[i].data.item() for i in range(len(model_u.coordinates))]
            Coord_trajectories.append(Coordinates_i)

        if epoch%1 == 0:
            print("epoch = ", epoch)
            print("     loss = ", loss_current)
            print("     loss_decrease = ", loss_decrease)


    print(f'* Final training loss: {numpy.format_float_scientific( error[-1], precision=4)}')
    print()
    Pplot.Plot_Compare_Loss2l2norm(error,[],'Loss_Comaprison')
    Pplot.PlotTrajectories(Coord_trajectories,'Trajectories')

    return model_u, model_du