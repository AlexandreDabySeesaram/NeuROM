import numpy as np
import torch
import matplotlib.pyplot as plt

def RHS(x):
    """Defines the right hand side (RHS) of the equation (the body force)"""
    b = -(4*np.pi**2*(x-2.5)**2-2*np.pi)/(torch.exp(np.pi*(x-2.5)**2)) \
        - (8*np.pi**2*(x-7.5)**2-4*np.pi)/(torch.exp(np.pi*(x-7.5)**2))
    return  b

def PotentialEnergy(A,E,u,x,b):
    """Computes the potential energy of the Beam, which will be used as the loss of the HiDeNN"""
    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    integral = 0
    for i in range(1,u.data.shape[0]):
        integral += (0.25*A*E*(x[i]-x[i-1])*(du_dx[i]**2+du_dx[i-1]**2)) \
            - 0.5*((x[i]-x[i-1])*(u[i]*b[i]+u[i-1]*b[i-1]))
    return integral

def PotentialEnergyVectorisedParametric(model,A, E, u, x, b):
    """Computes the potential energy of the Beam, which will be used as the loss of the HiDeNN"""
    # u_i = model.Space_modes[0](x)
    Space_modes = [model.Space_modes[l](x) for l in range(model.n_modes_truncated)]
    u_i = torch.cat(Space_modes,dim=1)  
    E = E[0] # From the multi-parametric version to the simple parameter one
    for mode in range(model.n_modes_truncated):
        Para_mode_List = [model.Para_modes[mode][l](E)[:,None] for l in range(model.n_para)]
        if mode == 0:
            lambda_i = torch.unsqueeze(torch.cat(Para_mode_List,dim=1), dim=0)
        else:
            New_mode = torch.unsqueeze(torch.cat(Para_mode_List,dim=1), dim=0)
            lambda_i = torch.vstack((lambda_i,New_mode))
    
    # u_i,coef = GramSchmidt_test(u_i)

    du_dx = [torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0] for u_x in Space_modes]
    # du_dx = [torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0] for u_x in u_i.t()]
    du_dx = torch.cat(du_dx,dim=1) 

    # u_i,du_dx = ModifiedGramSchmidt(u_i,du_dx)

    # du_dx = du_dx/coef
    du_dx = torch.matmul(du_dx,lambda_i.view(model.n_modes_truncated,lambda_i.shape[1]))
    # Calculate dx
    dx = x[1:] - x[:-1]

    u_debug = torch.matmul(u_i,lambda_i.view(model.n_modes_truncated,lambda_i.shape[1]))
    # Vectorised calculation of the integral terms
    int_term1_x = 0.25 * A * dx * (du_dx[1:]**2 + du_dx[:-1]**2)
    int_term1 = torch.einsum('ik,kj->ik',int_term1_x,E)
    # int_term2 = 0.5 * dx * (u[1:] * b[1:] + u[:-1] * b[:-1])
    int_term2 = 0.5 * dx * (u_debug[1:] * b[1:] + u_debug[:-1] * b[:-1])

    # Vectorised calculation of the integral using the trapezoidal rule
    integral = torch.sum(torch.sum(int_term1 - int_term2,axis=0))/(E.shape[0])
    # integral = torch.abs(torch.norm(u_i)-1)+torch.sum(torch.sum(int_term1 - int_term2,axis=0))/(E.shape[0])

    return integral

def EnhanceGramSchmidt(B,L):
    """This is a modified version of the Gram-Schmidt algorithm that allows to output the orth(no)gonal basis while modifiying the parameter modes so that the full field remains unchanged
     WARNING: Works only for a single parameter as extra-coordinates
     
     Args:
        B basis to orthonormalize
        L First parameter modes
    Returns: 
        orth_basis, L_updated
    """
    def projection(u, v):
        if u.norm() > 1e-6:
            return ((v * u).sum() / (u * u).sum()), ((v * u).sum() / (u * u).sum()) * u
        else:
            return 0, 0 * u
    n_mode = B.shape[1]
    orth_basis = torch.zeros_like(B, device=B.device)
    orth_basis[:, 0] = B[:, 0].clone()
    L_correction_List = []
    for n in range(1, n_mode):
        vn = B[:,n].clone()
        un = 0
        L_correction = torch.zeros_like(L, device=B.device)
        L_n = L[n,:,0,0].clone()
        for j in range(0, n):
            uj = orth_basis[:, j].clone()
            coef, proj = projection(uj, vn)
            un = un + proj
            L_correction[j,:,0,0] = coef*L_n
        L_correction_List.append(L_correction) 
        orth_basis[:, n] = vn - un
    L_updated = L.clone()+sum(L_correction_List)
    return orth_basis, L_updated



def GramSchmidt(B):
    """This is a Gram-Schmidt algorithm that allows to output the orth(no)gonal basis 
     Args:
        B basis to orthonormalize
    Returns: 
        orth_basis
    """    
    def projection(u, v):
        if u.norm() > 1e-6:
            return ((v * u).sum() / (u * u).sum()) * u
        else:
            return 0 * u
    n_mode = B.shape[1]
    orth_basis = torch.zeros_like(B, device=B.device)
    orth_basis[:, 0] = B[:, 0].clone()
    for n in range(1, n_mode):
        vn = B[:,n].clone()
        un = 0
        for j in range(0, n):
            uj = orth_basis[:, j].clone()
            un = un + projection(uj, vn)
        orth_basis[:, n] = vn - un
    # If we want the basis to be orthonormal
    for n in range(n_mode):
        un = orth_basis[:, n].clone()
        if un.norm()>1e-6:
            orth_basis[:, n] = un / un.norm()
        else:
            orth_basis[:, n] = un * 0

    return orth_basis


def GramSchmidt_test(B):
    """This is a Gram-Schmidt algorithm that allows to output the orth(no)gonal basis 
     Args:
        B basis to orthonormalize
    Returns: 
        orth_basis
    """    
    def projection(u, v):
        if u.norm() > 1e-6:
            return ((v * u).sum() / (u * u).sum()) * u
        else:
            return 0 * u
    n_mode = B.shape[1]
    orth_basis = torch.zeros_like(B, device=B.device)
    orth_basis[:, 0] = B[:, 0].clone()
    for n in range(1, n_mode):
        vn = B[:,n].clone()
        un = 0
        for j in range(0, n):
            uj = orth_basis[:, j].clone()
            un = un + projection(uj, vn)
        orth_basis[:, n] = vn - un
    # If we want the basis to be orthonormal
    # for n in range(n_mode):
    #     un = orth_basis[:, n].clone()
    #     if un.norm()>1e-6:
    #         orth_basis[:, n] = un / un.norm()
    #     else:
    #         orth_basis[:, n] = un * 0

    ##########Debug
    # orth_basis = orth_basis/1000
    # print(torch.norm(orth_basis))
    coef = (torch.norm(orth_basis))
    # coef = 100

    orth_basis = orth_basis/coef

    ###############
    return orth_basis, coef


def PotentialEnergyVectorisedBiParametric(model,A, E, x, b):
    """Computes the potential energy of the Beam, which will be used as the loss of the HiDeNN"""
    with torch.no_grad(): #No derivatino of the heavyside function
        f_1 = torch.heaviside(x-5,torch.tensor(1, dtype = torch.float32))
        f_2 = 1-torch.heaviside(x-5,torch.tensor(1, dtype = torch.float32))

    ###### Enables orthogonalisation process #######
    Orthgonality = False                                    # Enable othogonality constraint of the space modes
    if Orthgonality and model.training:
        Space_modes_nodal = [torch.unsqueeze(model.Space_modes[l].InterpoLayer_uu.weight.data,dim=1) for l in range(model.n_modes_truncated)]
        Space_modes_nodal = torch.cat(Space_modes_nodal,dim=1)
        Space_modes_nodal = GramSchmidt(Space_modes_nodal)
        for mode in range(model.n_modes_truncated):
            model.Space_modes[mode].InterpoLayer_uu.weight.data = Space_modes_nodal[:,mode]

      

    Space_modes = [model.Space_modes[l](x) for l in range(model.n_modes_truncated)]
    u_i = torch.cat(Space_modes,dim=1)  

    Para_mode_Lists = [
        [model.Para_modes[mode][l](E[l][:,0].view(-1,1))[:,None] for l in range(model.n_para)]
        for mode in range(model.n_modes_truncated)
        ]

    lambda_i = [
            torch.cat([torch.unsqueeze(Para_mode_Lists[m][l],dim=0) for m in range(model.n_modes_truncated)], dim=0)
            for l in range(model.n_para)
        ]

    # Compute the gradient space modes from the primal field space modes
    du_dx = [torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0] for u_x in Space_modes]
    du_dx = torch.cat(du_dx,dim=1)  

    # Calculate dx
    dx = x[1:] - x[:-1]

    TensorDecomposition = True                 # Enables using tensor decomposition as oppose to computing the full-order tensors
    if not TensorDecomposition:
        # # Intermediate 3rd order
        f_1_E_1 = torch.einsum('ik,jk->ij',f_1,E[0])
        f_2_E_2 = torch.einsum('ik,jk->ij',f_2,E[1])
        E_tensor = f_1_E_1[:,:,None]+f_2_E_2[:,None,:]
        du_dx = torch.einsum('ik,kj,kl->ijl',du_dx,lambda_i[0].view(model.n_modes_truncated,lambda_i[0].shape[1]),
                                lambda_i[1].view(model.n_modes_truncated,lambda_i[1].shape[1]))

        du_dx_E_tensor = (du_dx**2)*E_tensor

        # Vectorised calculation of the integral terms
        int_term1 = 0.25 * A * dx[:,None] * (du_dx_E_tensor[1:] + du_dx_E_tensor[:-1])
        int_term2 = 0.5 * dx[:,None] * (u[1:] * b[1:,None] + u[:-1] * b[:-1,None])

        # Vectorised calculation of the integral using the trapezoidal rule
        integral = torch.sum(torch.sum(torch.sum(int_term1 - int_term2,axis=0))/(E_tensor.shape[1]))/(E_tensor.shape[2])
    else:

        F = torch.cat((f_1,f_2),dim = 1)
        E1 = torch.cat((E[0],torch.ones(E[0].shape)),dim = 1)
        E2 = torch.cat((torch.ones(E[1].shape),E[1]),dim = 1)

        term1_contributions = 0.25 * A *(
            torch.einsum('im,mj...,ml...,iq,qj...,ql...,i...,ie,je,le->',du_dx[1:],lambda_i[0][:],lambda_i[1][:],du_dx[1:],lambda_i[0][:],lambda_i[1][:],dx,F[1:],E1,E2)+
            torch.einsum('im,mj...,ml...,iq,qj...,ql...,i...,ie,je,le->',du_dx[:-1],lambda_i[0][:],lambda_i[1][:],du_dx[:-1],lambda_i[0][:],lambda_i[1][:],dx,F[:-1],E1,E2)
    )
        term2_contributions = 0.5 * (torch.einsum('im...,mj...,mk...,i...,i...->',u_i[1:],lambda_i[0][:],lambda_i[1][:],dx,b[1:]) +
           torch.einsum('im...,mj...,mk...,i...,i...->',u_i[:-1],lambda_i[0][:],lambda_i[1][:],dx,b[:-1]) )
        integral = (term1_contributions-term2_contributions)/(E[0].shape[0]*E[1].shape[0])

    return integral

def PotentialEnergyVectorised(A, E, u, x, b):
    """Computes the potential energy of the Beam, which will be used as the loss of the HiDeNN"""
    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    # Calculate dx
    dx = x[1:] - x[:-1]
    # y = 0.5*A * E * du_dx**2 - u*b
    # integral = torch.trapezoid(y.view(-1),x.view(-1))

    # Vectorised calculation of the integral terms
    int_term1 = 0.25 * A * E * dx * (du_dx[1:]**2 + du_dx[:-1]**2)
    int_term2 = 0.5 * dx * (u[1:] * b[1:] + u[:-1] * b[:-1])

    # Vectorised calculation of the integral using the trapezoidal rule
    integral = torch.sum(int_term1 - int_term2)

    return integral


def AlternativePotentialEnergy(A,E,u,x,b):
    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    f_x = 0.5*(A*E*du_dx**2) - u*b
    f_x = f_x.view(-1)
    dx = torch.diff(x.view(-1))
    av = 0.5*(f_x[1:]+f_x[:-1])*dx
    return torch.sum(av)

def Derivative(u,x):
    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    return du_dx

def AnalyticSolution(A,E,x,u0=0,uL=0):
    lin = u0 + ((uL-u0)/10)*x
    out = (1/(A*E)*(torch.exp(-np.pi*(x-2.5)**2)-np.exp(-6.25*np.pi))) \
        + (2/(A*E)*(torch.exp(-np.pi*(x-7.5)**2)-np.exp(-56.25*np.pi))) \
            - (x/(10*A*E))*(np.exp(-6.25*np.pi) - np.exp(-56.25*np.pi))
    return out+lin

def AnalyticParametricSolution(A,E,x,u0=0,uL=0):
    E = E[0] # to simple parametric solution
    lin = u0 + ((uL-u0)/10)*x
    E = E.T
    out = (1/(A*E)*(torch.exp(-np.pi*(x-2.5)**2)-np.exp(-6.25*np.pi))) \
        + (2/(A*E)*(torch.exp(-np.pi*(x-7.5)**2)-np.exp(-56.25*np.pi))) \
            - (x/(10*A*E))*(np.exp(-6.25*np.pi) - np.exp(-56.25*np.pi))
    return out+lin

def AnalyticBiParametricSolution(A,E,x,u0=0,uL=0):
    E1 = E[0] 
    E2 = E[1]
    out = (1/(A*E1)*(torch.exp(-np.pi*(x-2.5)**2)-np.exp(-6.25*np.pi))) \
        + (2/(A*E2)*(torch.exp(-np.pi*(x-7.5)**2)-np.exp(-56.25*np.pi))) \
            - (x/(10*A*E1))*(np.exp(-6.25*np.pi)) + (x/(10*A*E2))*(np.exp(-56.25*np.pi))
    return out

def AnalyticGradientSolution(A,E,x):
    out = (2/(A*E)*((-np.pi)*(x-2.5)*torch.exp(-np.pi*(x-2.5)**2))) \
        + (4/(A*E)*((-np.pi)*(x-7.5)*torch.exp(-np.pi*(x-7.5)**2))) \
            - (1/(10*A*E))*(np.exp(-6.25*np.pi) - np.exp(-56.25*np.pi))
    return out

def MixedFormulation_Loss(A, E, u, du, x, b):

    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    du_dx_dx = torch.autograd.grad(du, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    res_constit = (du - du_dx)**2
    res_eq = (du_dx_dx + b/(A*E))**2

    #assert (res_constit.shape) == x.shape
    #assert (res_eq.shape) == x.shape

    return torch.mean(res_eq), torch.mean(res_constit)

def InternalEnergy_2D(u,x,lmbda, mu):
    eps =  Strain(u,x)
    sigma =  torch.stack(Stress(eps[:,0], eps[:,1], eps[:,2], lmbda, mu),dim=1)
    W_e = torch.sum(torch.stack([eps[:,0], eps[:,1], 2*eps[:,2]],dim=1)*sigma,dim=1)
    return W_e

def Stress(ep_11, ep_22, ep_12, lmbda, mu):
    tr_epsilon = ep_11 + ep_22
    return tr_epsilon*lmbda + 2*mu*ep_11, tr_epsilon*lmbda + 2*mu*ep_22, 2*mu*ep_12
def Strain(u,x):
    du = torch.autograd.grad(u[0,:], x, grad_outputs=torch.ones_like(u[0,:]), create_graph=True)[0]
    dv = torch.autograd.grad(u[1,:], x, grad_outputs=torch.ones_like(u[1,:]), create_graph=True)[0]
    return torch.stack([du[:,0], dv[:,1], 0.5*(du[:,1] + dv[:,0])],dim=1)

def Mixed_2D_loss(u_pred, v_pred, s11_pred, s22_pred, s12_pred, x, lmbda, mu):

    #print("u_pred = ", u_pred.shape)
    #print("x = ", x.shape)

    du = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    dv = torch.autograd.grad(v_pred, x, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0]
    
    s_11, s_22, s_12 = Stress(du[:,0], dv[:,1], 0.5*(du[:,1] + dv[:,0]), lmbda, mu)

    d_s11 = torch.autograd.grad(s11_pred, x, grad_outputs=torch.ones_like(s11_pred), create_graph=True)[0]
    d_s22 = torch.autograd.grad(s22_pred, x, grad_outputs=torch.ones_like(s22_pred), create_graph=True)[0]
    d_s12 = torch.autograd.grad(s12_pred, x, grad_outputs=torch.ones_like(s12_pred), create_graph=True)[0]

    
    res_eq = (d_s11[:,0] + d_s12[:,1])**2 + (d_s12[:,0] + d_s22[:,1])**2
    res_constit = (s_11 - s11_pred)**2 + (s_22 - s22_pred)**2 + (s_12 - s12_pred)**2


    assert sum(res_constit.shape) == x.shape[0]
    assert sum(res_eq.shape) == x.shape[0]

    return torch.mean(res_eq), torch.mean(res_constit), s_11, s_22, s_12
    
def Mixed_2D_loss_Displacement_based(model_u, model_du, x, lmbda, mu):

    du = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    dv = torch.autograd.grad(v_pred, x, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0]
    
    s_11, s_22, s_12 = Stress(du[:,0], dv[:,1], 0.5*(du[:,1] + dv[:,0]), lmbda, mu)

    d_s11 = torch.autograd.grad(s11_pred, x, grad_outputs=torch.ones_like(s11_pred), create_graph=True)[0]
    d_s22 = torch.autograd.grad(s22_pred, x, grad_outputs=torch.ones_like(s22_pred), create_graph=True)[0]
    d_s12 = torch.autograd.grad(s12_pred, x, grad_outputs=torch.ones_like(s12_pred), create_graph=True)[0]

    
    res_eq = (d_s11[:,0] + d_s12[:,1])**2 + (d_s12[:,0] + d_s22[:,1])**2
    res_constit = (s_11 - s11_pred)**2 + (s_22 - s22_pred)**2 + (s_12 - s12_pred)**2


    assert sum(res_constit.shape) == x.shape[0]
    assert sum(res_eq.shape) == x.shape[0]

    return torch.mean(res_eq), torch.mean(res_constit), s_11, s_22, s_12
    


def Neumann_BC_rel(Model):

    for i in range(len(Model.relation_BC_node_IDs)):

        nodes = Model.relation_BC_node_IDs[i]
        normals = Model.relation_BC_normals[i]
        value = Model.relation_BC_values[i]

        s_11 = Model.nodal_values[0]
        s_22 = Model.nodal_values[1]
        s_12 = Model.nodal_values[2]

        if len(value)>1:
            for j in range(nodes.shape[0]):

                ID = nodes[j]
                normal = normals[j]

                if np.isclose(normal[0],0.0, atol=1e-06):
                    s_12[ID] = torch.nn.Parameter(torch.tensor([value[0]/normal[1]]))
                    s_22[ID] = torch.nn.Parameter(torch.tensor([value[1]/normal[1]]))
                elif np.isclose(normal[1],0.0, atol=1e-06):
                    s_11[ID] = torch.nn.Parameter(torch.tensor([value[0]/normal[0]]))
                    s_12[ID] = torch.nn.Parameter(torch.tensor([value[1]/normal[0]]))
                else:
                    s_11[ID] = (value[0] - s_12[ID]*normal[1])/normal[0]
                    s_22[ID] = (value[1] - s_12[ID]*normal[0])/normal[1]
                    #s_12[ID] = (value[0] - s_11[ID]*normal[0])/normal[1]

        elif len(value)==1:
            for j in range(nodes.shape[0]):

                ID = nodes[j]
                normal = normals[j]

                #print(ID.item())
                #print(normal)

                if np.isclose(normal[0],0.0, atol=1e-06):
                    s_12[ID] = torch.nn.Parameter(torch.tensor([0*value[0]]))
                    s_22[ID] = torch.nn.Parameter(torch.tensor([value[0]]))
                elif np.isclose(normal[1],0.0, atol=1e-06):
                    s_11[ID] = torch.nn.Parameter(torch.tensor([value[0]]))
                    s_12[ID] = torch.nn.Parameter(torch.tensor([0*value[0]]))
                else:
                    #s_12[ID] = (value[0]*normal[0] - s_11[ID]*normal[0])/normal[1]
                    s_11[ID] = (value[0]*normal[0] - s_12[ID]*normal[1])/normal[0]
                    s_22[ID] = (value[0]*normal[1] - s_12[ID]*normal[0])/normal[1]

def Constitutive_BC(model_u, model_du, constit_point_coord_all, constit_cell_IDs_u_all, lmbda, mu):

    #print(" * Constit BC")

    NN_s_11 = model_du.nodal_values[0]
    NN_s_22 = model_du.nodal_values[1]
    NN_s_12 = model_du.nodal_values[2]

    all_node_IDs = model_du.constit_BC_node_IDs

    for i in range(len(all_node_IDs)):

        constit_point_coord = constit_point_coord_all[i]
        constit_cell_IDs_u = constit_cell_IDs_u_all[i]

        #print("     constit_cell_IDs_u = ", constit_cell_IDs_u )
        #print("     constit_point_coord = ", constit_point_coord)
        #print()

        u_pred = model_u(constit_point_coord, constit_cell_IDs_u)

        du = torch.autograd.grad(u_pred[0,:], constit_point_coord, grad_outputs=torch.ones_like(u_pred[0,:]), create_graph=True)[0]
        dv = torch.autograd.grad(u_pred[1,:], constit_point_coord, grad_outputs=torch.ones_like(u_pred[1,:]), create_graph=True)[0]
        
        s_11, s_22, s_12 = Stress(du[:,0], dv[:,1], 0.5*(du[:,1] + dv[:,0]), lmbda, mu)

        node_IDs = all_node_IDs[i]
        #print("         node_IDs : ", node_IDs)

        for j in range(len(node_IDs)):

                ID = node_IDs[j]
                NN_s_11[ID] = torch.nn.Parameter(torch.tensor([s_11[j]]))
                NN_s_22[ID] = torch.nn.Parameter(torch.tensor([s_22[j]]))
                NN_s_12[ID] = torch.nn.Parameter(torch.tensor([s_12[j]]))


def GetRealCoord(model, mesh, cell_ids, ref_coord):

    #for c in cell_ids:
    node_ids = mesh.Connectivity[cell_ids,:]

    node1_coord =  torch.cat([model.coordinates[int(row)-1] for row in node_ids[:,0]])
    node2_coord =  torch.cat([model.coordinates[int(row)-1] for row in node_ids[:,1]])
    node3_coord =  torch.cat([model.coordinates[int(row)-1] for row in node_ids[:,2]])

    coord = ref_coord[:,0].view(-1, 1)*node1_coord + ref_coord[:,1].view(-1, 1)*node2_coord + ref_coord[:,2].view(-1, 1)*node3_coord

    # print("cell ids : ", cell_ids[0])
    # print("node ids : ", node_ids[0,:])

    # print("n1 : ", node1_coord[0])
    # print("n2 : ", node2_coord[0])
    # print("n3 : ", node3_coord[0])
    # print("x : ", coord[0])
    # print("check cell ID : ", mesh.GetCellIds(coord)[0])
    # print()
    return coord.clone().detach().requires_grad_(True), cell_ids, cell_ids

