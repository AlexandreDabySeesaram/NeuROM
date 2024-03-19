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
    Space_modes = [model.Space_modes[l](x) for l in range(model.n_modes)]
    u_i = torch.cat(Space_modes,dim=1)  
    E = E[0] # From the multi-parametric version to the simple parameter one
    for mode in range(model.n_modes):
        Para_mode_List = [model.Para_modes[mode][l](E)[:,None] for l in range(model.n_para)]
        if mode == 0:
            lambda_i = torch.unsqueeze(torch.cat(Para_mode_List,dim=1), dim=0)
        else:
            New_mode = torch.unsqueeze(torch.cat(Para_mode_List,dim=1), dim=0)
            lambda_i = torch.vstack((lambda_i,New_mode))
    du_dx = [torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0] for u_x in Space_modes]
    du_dx = torch.cat(du_dx,dim=1)  
    du_dx = torch.matmul(du_dx,lambda_i.view(model.n_modes,lambda_i.shape[1]))
    # Calculate dx
    dx = x[1:] - x[:-1]

    # Vectorised calculation of the integral terms
    int_term1_x = 0.25 * A * dx * (du_dx[1:]**2 + du_dx[:-1]**2)
    int_term1 = torch.einsum('ik,kj->ik',int_term1_x,E)
    int_term2 = 0.5 * dx * (u[1:] * b[1:] + u[:-1] * b[:-1])

    # Vectorised calculation of the integral using the trapezoidal rule
    integral = torch.sum(torch.sum(int_term1 - int_term2,axis=0))/(E.shape[0])

    return integral

def EnhanceGramSchmidt(B,L):

    import copy
    lambda_2 = copy.deepcopy(L.data)

    # orth_basis = torch.zeros(B.shape)
    # norm = torch.norm(B[:,0])
    # orth_basis[:,0] = (B[:,0])/norm

    # lambda_updated = torch.zeros(L[:,:,0,0].shape)
    # lambda_updated[0,:] = L[0,:,0,0].clone()*norm

    # for m in range(1,B.shape[1]):
    #     v = B[:,m]    
    #     lambda_m = L[m,:,0,0]
    #     proj = torch.einsum('k,ki,li->l',v,orth_basis,orth_basis)
    #     norm = torch.norm(v - proj)
    #     if norm <1e-6: # to remove the normalisation of noise
    #         orth_basis[:,m] = 0*(v - proj)
    #         norm = 0
    #     else:
    #         orth_basis[:,m] = (v - proj)/norm
    #     proj_coef = torch.einsum('k,ki->i',v,orth_basis)
    #     lambda_updated = lambda_updated + proj_coef.unsqueeze(dim=1)*lambda_m
    #     lambda_m = lambda_m*norm
    #     lambda_updated[m,:] = lambda_m

# # Debug without vectorisation (for some reason the impact of normalising noise is not important here)

    orth_basis2 = torch.zeros(B.shape)
    norm = torch.norm(B[:,0])
    orth_basis2[:,0] = (B[:,0])/norm
    lambda_2[0,:,0,0] = lambda_2[0,:,0,0]*norm
    for m in range(1,B.shape[1]):
        v = B[:,m]    
        proj = 0
        for i in range(m):
            proj += (orth_basis2[:,i]@v)*orth_basis2[:,i]
            proj_coef = (orth_basis2[:,i]@v)
            lambda_2[i,:,0,0] += proj_coef*lambda_2[m,:,0,0]

        norm = torch.norm(v - proj)
        orth_basis2[:,m] = (v - proj)/norm
        lambda_2[m,:,0,0] = lambda_2[m,:,0,0]*norm


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


    # If we want the basis to be orthonormal
    # for n in range(n_mode):
    #     un = orth_basis[:, n].clone()
    #     if un.norm()>1e-6:
    #         orth_basis[:, n] = un / un.norm()
    #     else:
    #         orth_basis[:, n] = un * 0
    return orth_basis, L_updated
    # return orth_basis,lambda_updated






def PotentialEnergyVectorisedBiParametric(model,A, E, u, x, b):
    """Computes the potential energy of the Beam, which will be used as the loss of the HiDeNN"""
    with torch.no_grad(): #No derivatino of heavy side
        f_1 = torch.heaviside(x-5,torch.tensor(1, dtype = torch.float32))
        f_2 = 1-torch.heaviside(x-5,torch.tensor(1, dtype = torch.float32))

    Space_modes = [model.Space_modes[l](x) for l in range(model.n_modes)]
    u_i = torch.cat(Space_modes,dim=1)  

    Para_mode_Lists = [
        [model.Para_modes[mode][l](E[l][:,0].view(-1,1))[:,None] for l in range(model.n_para)]
        for mode in range(model.n_modes)
        ]

    lambda_i = [
            torch.cat([torch.unsqueeze(Para_mode_Lists[m][l],dim=0) for m in range(model.n_modes)], dim=0)
            for l in range(model.n_para)
        ]

    du_dx = [torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0] for u_x in Space_modes]
    du_dx = torch.cat(du_dx,dim=1)  

    TensorDecomposition = False                 # Enables using tensor decomposition as oppose to computing the full-order tensors
    if not TensorDecomposition:
        # # Intermediate 3rd order
        f_1_E_1 = torch.einsum('ik,jk->ij',f_1,E[0])
        f_2_E_2 = torch.einsum('ik,jk->ij',f_2,E[1])
        E_tensor = f_1_E_1[:,:,None]+f_2_E_2[:,None,:]
        du_dx = torch.einsum('ik,kj,kl->ijl',du_dx,lambda_i[0].view(model.n_modes,lambda_i[0].shape[1]),
                                lambda_i[1].view(model.n_modes,lambda_i[1].shape[1]))
        # Calculate dx
        dx = x[1:] - x[:-1]
        du_dx_E_tensor = (du_dx**2)*E_tensor

        # Vectorised calculation of the integral terms
        int_term1 = 0.25 * A * dx[:,None] * (du_dx_E_tensor[1:] + du_dx_E_tensor[:-1])
        int_term2 = 0.5 * dx[:,None] * (u[1:] * b[1:,None] + u[:-1] * b[:-1,None])

        # Vectorised calculation of the integral using the trapezoidal rule
        integral = torch.sum(torch.sum(torch.sum(int_term1 - int_term2,axis=0))/(E_tensor.shape[1]))/(E_tensor.shape[2])
    else:
        #### Without 3rd order /!\ Only if orthogonal u_i !!!
        # Gram-Schmidt
        u_i, lambda_i[0] = EnhanceGramSchmidt(u_i,lambda_i[0])
        # Calculate dx
        dx = x[1:] - x[:-1]
        f_1_E_1 = torch.einsum('ik,jk->ij',f_1,E[0])
        f_2_E_2 = torch.einsum('ik,jk->ij',f_2,E[1])
        term1_contributions = 0.25 * A *(
            torch.einsum('ip,i...,...im,mp...,mq...->',f_1_E_1[1:],dx, du_dx[1:]**2, lambda_i[0][:]**2, lambda_i[1][:]**2) +
            torch.einsum('iq,i...,...im,mp...,mq...->',f_2_E_2[1:],dx, du_dx[1:]**2, lambda_i[0][:]**2, lambda_i[1][:]**2) +
                    torch.einsum('ip,i...,...im,mp...,mq...->',f_1_E_1[:-1],dx, du_dx[:-1]**2, lambda_i[0][:]**2, lambda_i[1][:]**2) +
            torch.einsum('iq,i...,...im,mp...,mq...->',f_2_E_2[:-1],dx, du_dx[:-1]**2, lambda_i[0][:]**2, lambda_i[1][:]**2) 
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
