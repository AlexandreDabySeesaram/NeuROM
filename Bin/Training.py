import numpy as numpy
import torch
import Post.Plots as Pplot
import copy

from Bin.PDE_Library import RHS, PotentialEnergy, \
    PotentialEnergyVectorised, AlternativePotentialEnergy, \
        Derivative, AnalyticGradientSolution, AnalyticSolution,\
        MixedFormulation_Loss

def plot_everything(A,E,InitialCoordinates, Coordinates,
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
    Pplot.Plot_ShapeFuctions(TrialCoordinates.detach(), MeshBeam, InitialCoordinates, True, 'Shape_Functions')


def plot_everything_mixed(A,E,InitialCoordinates_u, InitialCoordinates_du, Coordinates_u, Coordinates_du,
                                            TrialCoordinates,AnalyticSolution,MeshBeam_u,MeshBeam_du, \
                                            Coord_trajectories, error, error2):
    # Tests on trained data and compare to reference
    Pplot.PlotSolution_Coordinates_Analytical(A,E,InitialCoordinates_u,Coordinates_u,
                                            TrialCoordinates,AnalyticSolution,MeshBeam_u,
                                            'Solution_displacement')
    # Plots the gradient & compare to reference

    Pplot.PlotGradSolution_Coordinates_Analytical(A,E,InitialCoordinates_u,Coordinates_u,
                                                TrialCoordinates,AnalyticGradientSolution,
                                                MeshBeam_u,Derivative,'Solution_gradients_dudx')

    Pplot.PlotGradSolution_Coordinates_Force(A,E,InitialCoordinates_du,Coordinates_du,
                                                TrialCoordinates, RHS(TrialCoordinates),
                                                MeshBeam_du,Derivative,'Solution_gradients_Force')

    Pplot.PlotSolution_Coordinates_Analytical(A,E,InitialCoordinates_du,Coordinates_du,
                                            TrialCoordinates,AnalyticGradientSolution,MeshBeam_du,
                                            'Solution_gradients')

    # Plots trajectories of the coordinates while training
    Pplot.PlotTrajectories(Coord_trajectories,'Trajectories')
    Pplot.PlotTrajectories(Coord_trajectories,'Trajectories')

    Pplot.Plot_Compare_Loss2l2norm(error,error2,'Loss_Comaprison')
    Pplot.Plot_ShapeFuctions(TrialCoordinates.detach(), MeshBeam_u, InitialCoordinates_u, True, 'Shape_Functions_u')
    Pplot.Plot_ShapeFuctions(TrialCoordinates.detach(), MeshBeam_du, InitialCoordinates_du, True, 'Shape_Functions_du')


def FilterTrainingData(MeshBeam, TestData):

    TestData = numpy.array(TestData.detach())

    NodeCoordinates = [MeshBeam.coordinates[i].data.item() for i in range(len(MeshBeam.coordinates))]
    idx = numpy.where( numpy.isclose(NodeCoordinates,TestData))
    #print("idx = ", idx)

    #print([NodeCoordinates[i] for i in idx[1]])
    #print([TestData[i] for i in idx[0]])

    for i in idx[0]:
        if i < TestData.shape[0]-1:
            TestData[i][0] = TestData[i][0] + min(1.0e-5, (TestData[i+1][0] - TestData[i][0])/100)
        else: 
            TestData[i][0] = TestData[i][0] - min(1.0e-5, (TestData[i][0] - TestData[i-1][0])/100)

    #print([TestData[i] for i in idx[0]])

    return torch.tensor(TestData, dtype=torch.float64, requires_grad=True)

def Test_GenerateShapeFunctions(MeshBeam, TrialCoordinates, name):
    InitialCoordinates = [MeshBeam.coordinates[i].data.item() for i in range(len(MeshBeam.coordinates))]    
    pred, ShapeFunctions = MeshBeam(TrialCoordinates)
    Pplot.Plot_ShapeFuctions(TrialCoordinates.detach(), MeshBeam, InitialCoordinates, True, name)

def Collision_Check(MeshBeam, coord_old, proximity_limit):
    # Chock colission - Revert if needed
    coord_new = [MeshBeam.coordinates[i].data.item() for i in range(len(MeshBeam.coordinates))]
    coord_dif = numpy.array([x - coord_new[i - 1] for i, x in enumerate(coord_new) if i > 0])
    if numpy.all(coord_dif > proximity_limit) == False:
        for j in range(coord_dif.shape[0]):
            if coord_dif[j] < proximity_limit:
                MeshBeam.coordinates[j].data = torch.Tensor([[coord_old[j]]])
                MeshBeam.coordinates[j+1].data = torch.Tensor([[coord_old[j+1]]])


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

    TrialCoordinates = FilterTrainingData(MeshBeam, TrialCoordinates)

    while epoch<n_epochs and loss_counter<1000: # and stagnancy_counter < 50 :

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

        Collision_Check(MeshBeam, coord_old, (L/n_elem)/5)

        with torch.no_grad():
            # Stores the loss
            error.append(l.item())
            # Stores the coordinates trajectories
            Coordinates_i = [MeshBeam.coordinates[i].data.item() for i in range(len(MeshBeam.coordinates))]
            Coord_trajectories.append(Coordinates_i)

            if BoolCompareNorms:
                # Copute and store the L2 error w.r.t. the analytical solution
                error2.append(MSE(AnalyticSolution(A,E,TrialCoordinates.data),u_predicted).data)

        if loss_decrease < 1.0e-7:
            stagnancy_counter = stagnancy_counter +1
        else:
            stagnancy_counter = 0

        
        if (epoch+1) % 100 == 0:
            print('epoch ', epoch+1, ' loss = ', numpy.format_float_scientific( l.item(), precision=4))
            print(" loss decrease = ",  numpy.format_float_scientific( loss_decrease, precision=4))

            plot_everything(A,E,InitialCoordinates,Coordinates_i,
                                            TrialCoordinates,AnalyticSolution,MeshBeam,Coord_trajectories,error, error2)                           
        
        epoch = epoch+1

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
    print("  Final l2 loss = ", error2[-1].item())

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

def GetConnectivity(n_elem, order):

    if order ==1:
        # Linear shape functions
        connectivity_vector = (-1)*torch.ones(n_elem+1)
        connectivity_matrix = torch.zeros((n_elem+1,2*(n_elem+2)))
        # node 0
        connectivity_matrix[0,0] = 1.0
        connectivity_matrix[0,5] = 1.0

        # node n
        connectivity_matrix[n_elem,3] = 1.0     
        connectivity_matrix[n_elem,-2] = 1.0     

        for node in range(1,n_elem):
            row = node
            left = node-1

            cols = [4+left*2, 4+left*2+3]
            for col in cols:
                connectivity_matrix[row,col] = 1

    elif order ==2:
        # Quadragic shape fucntions
        connectivity_vector = (-1)*torch.ones(2*n_elem+1)
        connectivity_matrix = torch.zeros((2*n_elem+1,3*n_elem + 2*2))
        # node 0
        connectivity_matrix[0,0] = 1.0
        connectivity_matrix[0,6] = 1.0

        # node n
        connectivity_matrix[2*n_elem,3] = 1.0     
        connectivity_matrix[2*n_elem,-3] = 1.0   

        for el in range(n_elem):
            connectivity_matrix[el*2+1,4+el*3+1] = 1
            connectivity_vector[el*2+1]=0

        for node in range(1,n_elem):
            row = 2*node
            left = node-1
            cols = [4+left*3, 4+left*3+5]
            for col in cols:
                connectivity_matrix[row,col] = 1
    return connectivity_matrix, connectivity_vector



def MixedFormulation_Training_InitialStage(MeshBeam_u, MeshBeam_du, A, E, L, n_elem_u, n_elem_du, CoordinatesBatchSet, PlotData,
                                                optimizer, n_epochs, BoolCompareNorms, MSE,
                                                error, error2, Coord_trajectories):

    # Store the initial coordinates before training (could be merged with Coord_trajectories)
    InitialCoordinates_u = [MeshBeam_u.coordinates[i].data.item() for i in range(len(MeshBeam_u.coordinates))]
    InitialCoordinates_du = [MeshBeam_du.coordinates[i].data.item() for i in range(len(MeshBeam_du.coordinates))]

    stagnancy_counter = 0
    epoch = 0
    loss_old = 1.0e3
    loss_min = 1.0e3
    loss_counter = 0

    #coord_min_loss = [MeshBeam.coordinates[i].data.item() for i in range(len(MeshBeam.coordinates))]
    #weights_min_loss = copy.deepcopy(MeshBeam.InterpoLayer_uu.weight.data.detach())

    #TrialCoordinates = torch.tensor([[i/50] for i in range(-50,550)], dtype=torch.float64, requires_grad=True)

    while epoch<n_epochs and loss_counter<1000: # and stagnancy_counter < 50:

        #n_train_points = torch.randint(200,600,(1,))[0]
        #n_train_points = int(100 + epoch*0.1)
        #TrialCoordinates = torch.tensor([[i] for i in torch.linspace(0,L,n_train_points)], dtype=torch.float64, requires_grad=True)

        for TrialCoordinates in CoordinatesBatchSet:
            
            TrialCoordinates = FilterTrainingData(MeshBeam_u, TrialCoordinates)
            TrialCoordinates = FilterTrainingData(MeshBeam_du, TrialCoordinates)

            coord_old_u = [MeshBeam_u.coordinates[i].data.item() for i in range(len(MeshBeam_u.coordinates))]
            #weights_old_u = copy.deepcopy(MeshBeam_u.InterpoLayer_uu.weight.data.detach())
            coord_old_du = [MeshBeam_du.coordinates[i].data.item() for i in range(len(MeshBeam_du.coordinates))]
            #weights_old_du = copy.deepcopy(MeshBeam_du.InterpoLayer_uu.weight.data.detach())

            # predict = forward pass with our model
            u_predicted, _ = MeshBeam_u(TrialCoordinates) 
            du_predicted, _ = MeshBeam_du(TrialCoordinates) 

            # loss for weights update
            l_pde, l_constit  = MixedFormulation_Loss(A, E, u_predicted, du_predicted, TrialCoordinates, RHS(TrialCoordinates))
            l =  l_pde + l_constit

            #### loss for tracking #############
            u_predicted_1, _ = MeshBeam_u(PlotData) 
            du_predicted_1, _ = MeshBeam_du(PlotData) 
            l_pde_1, l_constit_1  = MixedFormulation_Loss(A, E, u_predicted_1, du_predicted_1, PlotData, RHS(PlotData))
            l1 =  l_pde_1 + l_constit_1

            loss_current = l1.item()
            loss_decrease = (loss_old - loss_current)/numpy.abs(loss_old)
            loss_old = loss_current
            ####################################

            # check for new minimal loss - Update the state for revert
            if loss_min > loss_current:
                loss_min = loss_current

                #coord_min_loss_u = [MeshBeam_u.coordinates[i].data.item() for i in range(len(MeshBeam_u.coordinates))]
                #weights_min_loss_u = copy.deepcopy(MeshBeam_u.InterpoLayer_uu.weight.data.detach())
                #coord_min_loss_du = [MeshBeam_du.coordinates[i].data.item() for i in range(len(MeshBeam_du.coordinates))]
                #weights_min_loss_du = copy.deepcopy(MeshBeam_du.InterpoLayer_uu.weight.data.detach())

                torch.save(MeshBeam_u.state_dict(),"Results/Net_u.pt")
                torch.save(MeshBeam_du.state_dict(),"Results/Net_du.pt")

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

            Collision_Check(MeshBeam_u, coord_old_u, (L/n_elem_u)/5)
            Collision_Check(MeshBeam_du, coord_old_du, (L/n_elem_du)/5)


            with torch.no_grad():
                # Stores the loss
                error.append(l1.item())
                # Stores the coordinates trajectories
                Coordinates_u_i = [MeshBeam_u.coordinates[i].data.item() for i in range(len(MeshBeam_u.coordinates))]
                Coordinates_du_i = [MeshBeam_du.coordinates[i].data.item() for i in range(len(MeshBeam_du.coordinates))]
                Coord_trajectories.append(Coordinates_du_i)

                if BoolCompareNorms:
                    # Copute and store the L2 error w.r.t. the analytical solution
                    error2.append(MSE(AnalyticSolution(A,E,PlotData.data),u_predicted_1).data)

            if loss_decrease < 1.0e-7:
                stagnancy_counter = stagnancy_counter +1
            else:
                    stagnancy_counter = 0
                    
        if (epoch+1) % 100 == 0:
            print('epoch ', epoch+1)
            print(' loss PDE  = ', numpy.format_float_scientific( l_pde_1.item(), precision=4), '; loss Const = ', numpy.format_float_scientific( l_constit_1.item(), precision=4))
            #print("    loss decrease = ",  numpy.format_float_scientific( loss_decrease, precision=4))
            print(" loss decrease = ",  numpy.format_float_scientific( loss_decrease, precision=4), \
                    "loss counter = ", loss_counter, "stagnancy_counter = ", stagnancy_counter)
            #print(" size = ", n_points)
            print(' loss = ', numpy.format_float_scientific( l1.item(), precision=4))
            print()
            
            #sorted, _ = torch.sort(TrialCoordinates,0)

            plot_everything_mixed(A,E,InitialCoordinates_u, InitialCoordinates_du, Coordinates_u_i, Coordinates_du_i,
                                            PlotData, AnalyticSolution, MeshBeam_u, MeshBeam_du, \
                                            Coord_trajectories, error, error2)                           
            
        epoch = epoch+1


    print(' loss = ', numpy.format_float_scientific( l1.item(), precision=4), ", loss counter = ", loss_counter, "stagnancy_counter = ", stagnancy_counter)
    print()

    if loss_min < loss_current:
        print("Revert")
        print("  Iter = ", epoch)

        '''
        for j in range(len(coord_min_loss_u)):
            MeshBeam_u.coordinates[j].data = torch.Tensor([[coord_min_loss_u[j]]])
        for j in range(len(coord_min_loss_du)):
            MeshBeam_du.coordinates[j].data = torch.Tensor([[coord_min_loss_du[j]]])

        MeshBeam_u.InterpoLayer_uu.weight.data = torch.Tensor(weights_min_loss_u)
        MeshBeam_du.InterpoLayer_uu.weight.data = torch.Tensor(weights_min_loss_du)
        '''

        MeshBeam_u.load_state_dict(torch.load("Results/Net_u.pt"))
        MeshBeam_du.load_state_dict(torch.load("Results/Net_du.pt"))


        print("  Minimal loss = ", loss_min)
        u_predicted_1, _ = MeshBeam_u(PlotData) 
        du_predicted_1, _ = MeshBeam_du(PlotData) 
        l_pde_1, l_constit_1  = MixedFormulation_Loss(A, E, u_predicted_1, du_predicted_1, PlotData, RHS(PlotData))
        l1 =  l_pde_1 + l_constit_1

        print("  Loss after revert = ", l1.item())

    with torch.no_grad():
        # Stores the loss
        error.append(l1.item())
        # Stores the coordinates trajectories
        Coordinates_u_i = [MeshBeam_u.coordinates[i].data.item() for i in range(len(MeshBeam_u.coordinates))]
        Coordinates_du_i = [MeshBeam_du.coordinates[i].data.item() for i in range(len(MeshBeam_du.coordinates))]

        Coord_trajectories.append(Coordinates_du_i)

        if BoolCompareNorms:
            # Copute and store the L2 error w.r.t. the analytical solution
            error2.append(MSE(AnalyticSolution(A,E,PlotData.data),u_predicted_1).data)

    plot_everything_mixed(A,E,InitialCoordinates_u, InitialCoordinates_du, Coordinates_u_i, Coordinates_du_i,
                                                PlotData, AnalyticSolution, MeshBeam_u, MeshBeam_du, Coord_trajectories,error, error2)
    print("  Final l2 loss = ", error2[-1].item())

    return error, error2, InitialCoordinates_u, InitialCoordinates_du, Coord_trajectories



def MixedFormulation_Training_FinalStageLBFGS(MeshBeam_u, MeshBeam_du, A, E, L, InitialCoordinates_u, InitialCoordinates_du, TrialCoordinates,\
                                                 n_epochs, BoolCompareNorms, MSE, error=[], error2 =[],Coord_trajectories=[]):

    optim = torch.optim.LBFGS( list(MeshBeam_u.parameters()) + list(MeshBeam_du.parameters()),
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

            u_predicted, _ = MeshBeam_u(TrialCoordinates) 
            du_predicted, _ = MeshBeam_du(TrialCoordinates) 

            # loss for weights update
            l_pde, l_constit  = MixedFormulation_Loss(A, E, u_predicted, du_predicted, TrialCoordinates, RHS(TrialCoordinates))
            l =  l_pde + l_constit

            l.backward()
            return l

        optim.step(closure)
        l = closure()


        with torch.no_grad():
                # Stores the loss
                error.append(l.item())
                u_predicted, _ = MeshBeam_u(TrialCoordinates) 

                # Stores the coordinates trajectories
                Coordinates_u_i = [MeshBeam_u.coordinates[i].data.item() for i in range(len(MeshBeam_u.coordinates))]
                Coordinates_du_i = [MeshBeam_du.coordinates[i].data.item() for i in range(len(MeshBeam_du.coordinates))]
                Coord_trajectories.append(Coordinates_u_i)

                if BoolCompareNorms:
                    # Copute and store the L2 error w.r.t. the analytical solution
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

    plot_everything_mixed(A,E,InitialCoordinates_u, InitialCoordinates_du, Coordinates_u_i, Coordinates_du_i,
                                                TrialCoordinates, AnalyticSolution, MeshBeam_u, MeshBeam_du, Coord_trajectories,error, error2)