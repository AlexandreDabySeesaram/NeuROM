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
        f_1 = torch.heaviside(x-5,torch.tensor(1, dtype = model.float_config.dtype, device = model.float_config.device))
        f_2 = 1-torch.heaviside(x-5,torch.tensor(1, dtype = model.float_config.dtype, device = model.float_config.device))

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
        E1 = torch.cat((E[0],torch.ones(E[0].shape, dtype = model.float_config.dtype, device = model.float_config.device)),dim = 1)
        E2 = torch.cat((torch.ones(E[1].shape, dtype = model.float_config.dtype, device = model.float_config.device),E[1]),dim = 1)
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

def AnalyticSolution(A,E,x,u0=0,uL=0.005):
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

def AnalyticGradientSolution(A,E,x,u0=0,uL=0.005):
    lin = ((uL-u0)/10)
    out = (2/(A*E)*((-np.pi)*(x-2.5)*torch.exp(-np.pi*(x-2.5)**2))) \
        + (4/(A*E)*((-np.pi)*(x-7.5)*torch.exp(-np.pi*(x-7.5)**2))) \
            - (1/(10*A*E))*(np.exp(-6.25*np.pi) - np.exp(-56.25*np.pi))
    return out + lin

def MixedFormulation_Loss(A, E, u, du, x, b):

    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    du_dx_dx = torch.autograd.grad(du, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    res_constit = (du - du_dx)**2
    res_eq = (du_dx_dx + b/(A*E))**2

    #assert (res_constit.shape) == x.shape
    #assert (res_eq.shape) == x.shape

    return torch.mean(res_eq), torch.mean(res_constit)


def AnalyticSolution1(A,E,x,u0=0,uL=0.005):
    C =( uL - 10**3/(3*A*E))/10
    return 1/(A*E)*(x**3/3) + C*x

def AnalyticGradientSolution1(A,E,x,u0=0,uL=0.005):
    C =( uL - 10**3/(3*A*E))/10

    return 1/(A*E)*(x**2) + C

def RHS1(x):
    return  2*x

def InternalEnergy_1D(u,x,A,E):

    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    W_e = 0.5*A*E*du_dx*du_dx - RHS(x)*u
    
    return W_e

def WeakEquilibrium_1D(u, u_test, du_test_dx, x, A, E):

    du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    prod = A*E*du_dx*du_test_dx - RHS(x)*u_test
    return prod


def InternalEnergy_2D(u,x,lmbda, mu):

    eps =  Strain(u,x)

    sigma =  torch.stack(Stress(eps[:,0], eps[:,1], eps[:,2], lmbda, mu),dim=1)

    W_e = torch.sum(torch.stack([eps[:,0], eps[:,1], 2*eps[:,2]],dim=1)*sigma,dim=-1)
    
    return W_e

def Gravity(theta,rho = 1e-9):
    g = 9.81*1e3                            #m/s^2
    return (rho*g*torch.tensor([[torch.sin(theta)],[-torch.cos(theta)]], dtype=torch.float64))

def Gravity_vect(theta,rho = 1e-9):
    """Should be unified with Gravity defined above, require chaging torch.tensor(0*torch.pi/2) to torch.tensor([0*torch.pi/2]) in 
    functions referencing the latter."""
    g = 9.81*1e3                            #m/s^2
    return (rho*g*torch.stack([torch.sin(theta),-torch.cos(theta)]))

def VolumeForcesEnergy_2D(u,theta, rho):
    fv = Gravity(theta,rho).to(u.dtype).to(u.device)
    W_e = u.t()@fv
    return torch.squeeze(W_e)

def Stress(ep_11, ep_22, ep_12, lmbda, mu):
    tr_epsilon = ep_11 + ep_22
    return tr_epsilon*lmbda + 2*mu*ep_11, tr_epsilon*lmbda + 2*mu*ep_22, 2*mu*ep_12

def VonMises(sigma, lmbda, mu):
    
    # Accounts for sigma_zz = (lmbda/(2(mu+lmda)))*(sigma_xx+sigma_yy) != 0 if in plain strain
    two = torch.tensor(2,dtype=torch.float64)
    two2threeD = torch.tensor([[1, 0, 0], [0, 1, 0],[lmbda/(2*(mu+lmbda)),lmbda/(2*(mu+lmbda)),0],[0, 0, 1]],dtype=sigma.dtype, device=sigma.device)
    sigma_3D = torch.einsum('ij,ej->ei',two2threeD,sigma)
    VM = torch.tensor([[2/3, -1/3, -1/3, 0],[-1/3, 2/3,-1/3, 0],[-1/3, -1/3,2/3, 0],[0, 0, 0, torch.sqrt(two)]],dtype=sigma.dtype, device=sigma.device)
    sigma_dev = torch.einsum('ij,ej->ei',VM,sigma_3D) # in voigt notation 
    sigma_VM = torch.einsum('ei,ei->e',sigma_dev,sigma_dev) # in voigt notation
    return torch.sqrt((3/2)*sigma_VM)

def VonMises_plain_strain(sigma, lmbda, mu):
    # Accounts for sigma_zz = (lmbda/(2(mu+lmda)))*(sigma_xx+sigma_yy) != 0 if in plain strain
    two = torch.tensor(2,dtype=torch.float64)
    two2threeD = torch.tensor([[1, 0, 0], [0, 1, 0],[lmbda/(2*(mu+lmbda)),lmbda/(2*(mu+lmbda)),0],[0, 0, 1]],dtype=sigma.dtype, device=sigma.device)
    sigma_3D = torch.einsum('ij,ej->ei',two2threeD,sigma)
    VM = torch.tensor([[2/3, -1/3, -1/3, 0],[-1/3, 2/3,-1/3, 0],[-1/3, -1/3,2/3, 0],[0, 0,0, torch.sqrt(two)]],dtype=sigma.dtype, device=sigma.device)
    sigma_dev = torch.einsum('ij,ej->ei',VM,sigma_3D) # in voigt notation 
    sigma_VM = torch.einsum('ei,ei->e',sigma_dev,sigma_dev) # in voigt notation
    return torch.sqrt((3/2)*sigma_VM)

def Stress_tensor(eps, lmbda, mu):
    K = torch.tensor([[2*mu+lmbda, lmbda, 0],[lmbda, 2*mu+lmbda, 0],[0, 0, 2*mu]],dtype=eps.dtype, device=eps.device)
    sigma = torch.einsum('ij,ej->ei',K,eps)
    return sigma

def InternalEnergy_2D_einsum(u,x,lmbda, mu):
    eps =  Strain_sqrt(u,x)
    K = torch.tensor([[2*mu+lmbda, lmbda, 0],[lmbda, 2*mu+lmbda, 0],[0, 0, 2*mu]],dtype=eps.dtype, device=eps.device)
    W_e = torch.einsum('ij,ej,ei->e',K,eps,eps)
    return W_e

def InternalResidual(u,x,u_star,x_star,lmbda, mu):
    eps =  Strain_sqrt(u,x)
    eps_star = Strain_sqrt(u_star,x_star)
    K = torch.tensor([[2*mu+lmbda, lmbda, 0],[lmbda, 2*mu+lmbda, 0],[0, 0, 2*mu]],dtype=eps.dtype, device=eps.device)
    W_e = torch.einsum('ij,ej,ei->e',K,eps,eps_star)
    return W_e

def InternalResidual_precomputed(eps,eps_star,lmbda, mu):
    # eps =  Strain_sqrt(u,x)
    # eps_star = Strain_sqrt(u_star,x_star)
    K = torch.tensor([[2*mu+lmbda, lmbda, 0],[lmbda, 2*mu+lmbda, 0],[0, 0, 2*mu]],dtype=eps.dtype, device=eps.device)
    W_e = torch.einsum('ij,ej,ei->e',K,eps,eps_star)
    return W_e

def InternalEnergy_2D_einsum_para(model,lmbda, mu,E):
    # Space_modes = [model.Space_modes[l]()[0] for l in range(model.n_modes_truncated)]
    # # Need to extract the u_i and xg simultaneously to keep the linkj
    # xg_modes = [model.Space_modes[l]()[1] for l in range(model.n_modes_truncated)]
    # detJ_modes = [model.Space_modes[l]()[2] for l in range(model.n_modes_truncated)]
    Space_modes = []
    xg_modes = []
    detJ_modes = []
    for i in range(model.n_modes_truncated):
        u_k,xg_k,detJ_k = model.Space_modes[i]()
        Space_modes.append(u_k)
        xg_modes.append(xg_k)
        detJ_modes.append(detJ_k)


    u_i = torch.stack(Space_modes,dim=2)
    xg_i = torch.stack(xg_modes,dim=2) 
    detJ_i = torch.stack(detJ_modes,dim=1)  

    eps_list = [Strain_sqrt(Space_modes[i],xg_modes[i]) for i in range(model.n_modes_truncated)]
    eps_i = torch.stack(eps_list,dim=2)  
    K = torch.tensor([[2*mu+lmbda, lmbda, 0],[lmbda, 2*mu+lmbda, 0],[0, 0, 2*mu]],dtype=model.float_config.dtype, device=model.float_config.device)
    Para_mode_Lists = [
        [model.Para_modes[mode][l](E[l][:,0].view(-1,1))[:,None] for l in range(model.n_para)]
        for mode in range(model.n_modes_truncated)
        ]
    lambda_i = [
            torch.cat([torch.unsqueeze(Para_mode_Lists[m][l],dim=0) for m in range(model.n_modes_truncated)], dim=0)
            for l in range(model.n_para)
        ]    
    E_float = E[0][:,0]

    W_int = torch.einsum('ij,ejm,eil,em,mp...,lp...,p->',K,eps_i,eps_i,torch.abs(detJ_i),lambda_i[0],lambda_i[0],E_float)

    # W_int = torch.einsum('ij,ejm,eim,em,mp...,p->',K,eps_i,eps_i,torch.abs(detJ_i),lambda_i[0].to(torch.float64),E[0][:,0].to(torch.float64))
    W_ext_e = [VolumeForcesEnergy_2D(Space_modes[i],theta = torch.tensor(0*torch.pi/2), rho = 1e-9) for i in range(model.n_modes_truncated)]
    W_ext_e = torch.stack(W_ext_e,dim=1)
    W_ext = torch.einsum('em,em,mp...->',W_ext_e,torch.abs(detJ_i),lambda_i[0])
    return (0.5*W_int - W_ext)/(E[0].shape[0])
    # return (0.5*W_int)/(E[0].shape[0])

def InternalEnergy_2D_einsum_Bipara(model,lmbda, mu,E):

    Space_modes = []
    xg_modes = []
    detJ_modes = []
    for i in range(model.n_modes_truncated):
        u_k,xg_k,detJ_k = model.Space_modes[i]()
        Space_modes.append(u_k)
        xg_modes.append(xg_k)
        detJ_modes.append(detJ_k)

 
    u_i = torch.stack(Space_modes,dim=2)
    xg_i = torch.stack(xg_modes,dim=2) 
    detJ_i = torch.stack(detJ_modes,dim=1)  

    eps_list = [Strain_sqrt(Space_modes[i],xg_modes[i]) for i in range(model.n_modes_truncated)]
    eps_i = torch.stack(eps_list,dim=2)  
    K = torch.tensor([[2*mu+lmbda, lmbda, 0],[lmbda, 2*mu+lmbda, 0],[0, 0, 2*mu]],dtype=model.float_config.dtype, device=model.float_config.device)
    Para_mode_Lists = [
        [model.Para_modes[mode][l](E[l][:,0].view(-1,1))[:,None] for l in range(model.n_para)]
        for mode in range(model.n_modes_truncated)
        ]
    lambda_i = [
            torch.cat([torch.unsqueeze(Para_mode_Lists[m][l],dim=0) for m in range(model.n_modes_truncated)], dim=0)
            for l in range(model.n_para)
        ]    
    E_float = E[0][:,0]
    theta_float = E[1][:,0]

    W_int = torch.einsum('ij,ejm,eil,em,mp...,lp...,mt...,lt...,p->',K,eps_i,eps_i,torch.abs(detJ_i),lambda_i[0],lambda_i[0],lambda_i[1],lambda_i[1],E_float)

    Gravity_force = Gravity_vect(theta_float,rho = 1e-9).to(model.float_config.dtype).to(model.float_config.device)
    W_ext = torch.einsum('iem,it,mp...,mt...,em->',u_i,Gravity_force,lambda_i[0],lambda_i[1],torch.abs(detJ_i))

    return (0.5*W_int - W_ext)/(E[0].shape[0])
    # return (0.5*W_int)/(E[0].shape[0])

def InternalEnergy_2D_einsum_BiStiffness(model,lmbda, mu,E):

    Space_modes = []
    xg_modes = []
    detJ_modes = []
    for i in range(model.n_modes_truncated):
        u_k,xg_k,detJ_k = model.Space_modes[i]()
        Space_modes.append(u_k)
        xg_modes.append(xg_k)
        detJ_modes.append(detJ_k)

 
    u_i = torch.stack(Space_modes,dim=2)
    xg_i = torch.stack(xg_modes,dim=2) 
    detJ_i = torch.stack(detJ_modes,dim=1)  

    eps_list = [Strain_sqrt(Space_modes[i],xg_modes[i]) for i in range(model.n_modes_truncated)]
    eps_i = torch.stack(eps_list,dim=2)  
    K = torch.tensor([[2*mu+lmbda, lmbda, 0],[lmbda, 2*mu+lmbda, 0],[0, 0, 2*mu]],dtype=torch.float64)
    Para_mode_Lists = [
        [model.Para_modes[mode][l](E[l][:,0].view(-1,1))[:,None] for l in range(model.n_para)]
        for mode in range(model.n_modes_truncated)
        ]
    lambda_i = [
            torch.cat([torch.unsqueeze(Para_mode_Lists[m][l],dim=0) for m in range(model.n_modes_truncated)], dim=0).to(torch.float64)
            for l in range(model.n_para)
        ]    

    # To be replaced with the decoder in the full auto-encoder framework
    # support = (1+torch.tanh(xg_k[:,None,1] - E[1][None,:,0]))*0.5
    with torch.no_grad(): #No derivatino of the heavyside function
        support = (torch.heaviside(xg_k[:,None,0] - E[1][None,:,0],torch.tensor(1, dtype = torch.float64)))


    E_1 = E[0][0,0].to(torch.float64)
    Delta_E_float = E[0][:,0].to(torch.float64)

    W_int = E_1*torch.einsum('ij,ejm,eil,em,mp...,lp...,mt...,lt...->',K,eps_i,eps_i,torch.abs(detJ_i),lambda_i[0],lambda_i[0],lambda_i[1],lambda_i[1]) +  \
            torch.einsum('ij,ejm,eil,em,mp...,lp...,mt...,lt...,p,et->',K,eps_i,eps_i,torch.abs(detJ_i),lambda_i[0],lambda_i[0],lambda_i[1],lambda_i[1],Delta_E_float,support)

    # Gravity_force = Gravity_vect(theta_float,rho = 1e-9)
    Gravity_force = Gravity_vect(torch.tensor(0*torch.pi).to(torch.float64),rho = 1e-9).to(model.float_config.dtype).to(model.float_config.device)

    W_ext = torch.einsum('iem,i,mp...,mt...,em->',u_i,Gravity_force,lambda_i[0],lambda_i[1],torch.abs(detJ_i))

    return (0.5*W_int - W_ext)/(E[0].shape[0])
    # return (0.5*W_int)/(E[0].shape[0])

def PotentialEnergyVectorisedParametric_Gauss(model,A, E):
    """Computes the potential energy of the Beam, which will be used as the loss of the HiDeNN"""

    Space_modes = []
    xg_modes = []
    detJ_modes = []
    for i in range(model.n_modes_truncated):
        u_k,xg_k,detJ_k = model.Space_modes[i]()
        Space_modes.append(u_k)
        xg_modes.append(xg_k)
        detJ_modes.append(detJ_k)

    u_i = torch.stack(Space_modes,dim=2)
    xg_i = torch.stack(xg_modes,dim=2) 
    detJ_i = torch.stack(detJ_modes,dim=1) 
    b = RHS(xg_modes[0])
    E = E[0] # From the multi-parametric version to the simple parameter one
    for mode in range(model.n_modes_truncated):
        Para_mode_List = [model.Para_modes[mode][l](E)[:,None] for l in range(model.n_para)]
        if mode == 0:
            lambda_i = torch.unsqueeze(torch.cat(Para_mode_List,dim=1), dim=0)
        else:
            New_mode = torch.unsqueeze(torch.cat(Para_mode_List,dim=1), dim=0)
            lambda_i = torch.vstack((lambda_i,New_mode))
    
    du_dx = [torch.autograd.grad(Space_modes[i], xg_modes[i], grad_outputs=torch.ones_like(Space_modes[i]), create_graph=True)[0] for i in range(model.n_modes_truncated)]
    du_dx = torch.stack(du_dx,dim=2)  
    Wint = 0.5*A*torch.einsum('egm,egk,emg,mp...,kp...,p...->', du_dx, du_dx, torch.abs(detJ_i),lambda_i.to(torch.float64),lambda_i.to(torch.float64),E.to(torch.float64) )
    Wext = torch.einsum('egm,mp...,emg,eg->', u_i,lambda_i.to(torch.float64),torch.abs(detJ_i),b)
    integral = (Wint-Wext)/(E.shape[0])
    return integral

def PotentialEnergyVectorisedBiParametric_Gauss(model,A, E):
    """Computes the potential energy of the Beam, which will be used as the loss of the HiDeNN"""
    with torch.no_grad(): #No derivatino of the heavyside function
        f_1 = torch.heaviside(x-5,torch.tensor(1, dtype = torch.float32))
        f_2 = 1-torch.heaviside(x-5,torch.tensor(1, dtype = torch.float32))

        # HERE x DEPENDS ON THE MODES SO F_xg additional dimension m 


    Space_modes = []
    xg_modes = []
    detJ_modes = []
    for i in range(model.n_modes_truncated):
        u_k,xg_k,detJ_k = model.Space_modes[i]()
        Space_modes.append(u_k)
        xg_modes.append(xg_k)
        detJ_modes.append(detJ_k)

    u_i = torch.stack(Space_modes,dim=2)
    xg_i = torch.stack(xg_modes,dim=2) 
    detJ_i = torch.stack(detJ_modes,dim=1)  

    Para_mode_Lists = [
        [model.Para_modes[mode][l](E[l][:,0].view(-1,1))[:,None] for l in range(model.n_para)]
        for mode in range(model.n_modes_truncated)
        ]

    lambda_i = [
            torch.cat([torch.unsqueeze(Para_mode_Lists[m][l],dim=0) for m in range(model.n_modes_truncated)], dim=0)
            for l in range(model.n_para)
        ]
 
    du_dx = [torch.autograd.grad(Space_modes[i], xg_modes[i], grad_outputs=torch.ones_like(Space_modes[i]), create_graph=True)[0] for i in range(model.n_modes_truncated)]
    du_dx = torch.stack(du_dx,dim=2)  

    # Calculate dx
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

def Strain_sqrt(u,x):
    """ Return the Scientific voigt notation  of the strain [eps_xx eps_yy sqrt(2)eps_xy]"""
    du = torch.autograd.grad(u[0,:], x, grad_outputs=torch.ones_like(u[0,:]), create_graph=True)[0]
    dv = torch.autograd.grad(u[1,:], x, grad_outputs=torch.ones_like(u[1,:]), create_graph=True)[0]
    return torch.stack([du[:,0], dv[:,1], (1/torch.sqrt(torch.tensor(2)))*(du[:,1] + dv[:,0])],dim=1)

def Strain(u,x):
    """ Return the vector strain [eps_xx eps_yy eps_xy]"""
    du = torch.autograd.grad(u[0,:], x, grad_outputs=torch.ones_like(u[0,:]), create_graph=True)[0]
    dv = torch.autograd.grad(u[1,:], x, grad_outputs=torch.ones_like(u[1,:]), create_graph=True)[0]

    return torch.stack([du[...,0], dv[...,1], 0.5*(du[...,1] + dv[...,0])],dim=1)





def Mixed_2D_loss(u_pred, v_pred, s11_pred, s22_pred, s12_pred, x, lmbda, mu):

    du = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    dv = torch.autograd.grad(v_pred, x, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0]
    
    s_11, s_22, s_12 = Stress(du[:,0], dv[:,1], 0.5*(du[:,1] + dv[:,0]), lmbda, mu)

    d_s11 = torch.autograd.grad(s11_pred, x, grad_outputs=torch.ones_like(s11_pred), create_graph=True)[0]
    d_s22 = torch.autograd.grad(s22_pred, x, grad_outputs=torch.ones_like(s22_pred), create_graph=True)[0]
    d_s12 = torch.autograd.grad(s12_pred, x, grad_outputs=torch.ones_like(s12_pred), create_graph=True)[0]

    # d_s11_u = torch.autograd.grad(s_11, x, grad_outputs=torch.ones_like(s_11), create_graph=True)[0]
    # d_s22_u = torch.autograd.grad(s_22, x, grad_outputs=torch.ones_like(s_22), create_graph=True)[0]
    # d_s12_u = torch.autograd.grad(s_12, x, grad_outputs=torch.ones_like(s_12), create_graph=True)[0]

    res_eq = (d_s11[:,0] + d_s12[:,1])**2 + (d_s12[:,0] + d_s22[:,1])**2
    res_constit = (s_11 - s11_pred)**2 + (s_22 - s22_pred)**2 + (s_12 - s12_pred)**2
    # res_eq_u = (d_s11_u[:,0] + d_s12_u[:,1])**2 + (d_s12_u[:,0] + d_s22_u[:,1])**2

    assert sum(res_constit.shape) == x.shape[0]
    assert sum(res_eq.shape) == x.shape[0]

    return torch.mean(res_eq), torch.mean(res_constit), s_11, s_22, s_12 


def Mixed_2D_loss_Strain(u_pred, v_pred, dudx_pred, dvdy_pred, dudy_pred, x, lmbda, mu):

    du = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    dv = torch.autograd.grad(v_pred, x, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0]
    
    s_11, s_22, s_12 = Stress(dudx_pred, dvdy_pred, dudy_pred, lmbda, mu)

    d_s11 = torch.autograd.grad(s_11, x, grad_outputs=torch.ones_like(s_11), create_graph=True)[0]
    d_s22 = torch.autograd.grad(s_22, x, grad_outputs=torch.ones_like(s_12), create_graph=True)[0]
    d_s12 = torch.autograd.grad(s_12, x, grad_outputs=torch.ones_like(s_22), create_graph=True)[0]

    res_eq = (d_s11[:,0] + d_s12[:,1])**2 + (d_s12[:,0] + d_s22[:,1])**2
    res_constit = (du[:,0] - dudx_pred)**2 + (dv[:,1] - dvdy_pred)**2 + (0.5*(du[:,1] + dv[:,0]) - dudy_pred)**2

    assert sum(res_constit.shape) == x.shape[0]
    assert sum(res_eq.shape) == x.shape[0]

    return torch.mean(res_eq), torch.mean(res_constit), s_11, s_22, s_12

def Neumann_BC_rel(Model):
    #print(" * Neumann_BC_rel")

    s_11 = Model.nodal_values[0]
    s_22 = Model.nodal_values[1]
    s_12 = Model.nodal_values[2]

    for i in range(len(Model.relation_BC_node_IDs)):

        nodes = Model.relation_BC_node_IDs[i]
        normals = Model.relation_BC_normals[i]
        value = Model.relation_BC_values[i]

        if len(value)>1:
            for j in range(nodes.shape[0]):

                ID = nodes[j]
                normal = normals[j]

                if np.isclose(normal[0],0.0, atol=1.0e-8):
                    s_12[ID] = torch.nn.Parameter(torch.tensor([value[0]/normal[1]]))
                    s_22[ID] = torch.nn.Parameter(torch.tensor([value[1]/normal[1]]))
                elif np.isclose(normal[1],0.0, atol=1.0e-8):
                    s_11[ID] = torch.nn.Parameter(torch.tensor([value[0]/normal[0]]))
                    s_12[ID] = torch.nn.Parameter(torch.tensor([value[1]/normal[0]]))
                else:
                    #s_12[ID] = (value[1] - s_22[ID]*normal[1])/normal[0]
                    s_11[ID] = (value[0] - s_12[ID]*normal[1])/normal[0]
                    s_22[ID] = (value[1] - s_12[ID]*normal[0])/normal[1]

        elif len(value)==1:
            for j in range(nodes.shape[0]):

                ID = nodes[j]
                normal = normals[j]

                if np.isclose(normal[0],0.0, atol=1.0e-8):
                    s_12[ID] = torch.nn.Parameter(torch.tensor([0*value[0]]))
                    s_22[ID] = torch.nn.Parameter(torch.tensor([value[0]]))
                elif np.isclose(normal[1],0.0, atol=1.0e-8):
                    s_11[ID] = torch.nn.Parameter(torch.tensor([value[0]]))
                    s_12[ID] = torch.nn.Parameter(torch.tensor([0*value[0]]))
                else:
                    #s_12[ID] = (value[0]*normal[0] - s_11[ID]*normal[0])/normal[1]
                    s_11[ID] = (value[0]*normal[0] - s_12[ID]*normal[1])/normal[0]
                    s_22[ID] = (value[0]*normal[1] - s_12[ID]*normal[0])/normal[1]

def CheckNeumann_BC_rel(Model):
    #print(" * Neumann_BC_rel")

    s_11 = Model.nodal_values[0]
    s_22 = Model.nodal_values[1]
    s_12 = Model.nodal_values[2]

    for i in range(len(Model.relation_BC_node_IDs)):

        nodes = Model.relation_BC_node_IDs[i]
        normals = Model.relation_BC_normals[i]
        value = Model.relation_BC_values[i]

        if len(value)>1:
            for j in range(nodes.shape[0]):

                ID = nodes[j]
                normal = normals[j]

                if np.isclose(normal[0],0.0, atol=1.0e-8):
                    print("nx ==0")
                    print(s_12[ID] - torch.nn.Parameter(torch.tensor([value[0]/normal[1]])))
                    print(s_22[ID] - torch.nn.Parameter(torch.tensor([value[1]/normal[1]])))
                elif np.isclose(normal[1],0.0, atol=1.0e-8):
                    print("ny == 0")
                    print(s_11[ID] - torch.nn.Parameter(torch.tensor([value[0]/normal[0]])))
                    print(s_12[ID] - torch.nn.Parameter(torch.tensor([value[1]/normal[0]])))
                else:
                    print("n general")
                    print(s_11[ID] - (value[0] - s_12[ID]*normal[1])/normal[0])
                    print(s_22[ID] - (value[1] - s_12[ID]*normal[0])/normal[1])
                    print()
        elif len(value)==1:
            for j in range(nodes.shape[0]):

                ID = nodes[j]
                normal = normals[j]

                if np.isclose(normal[0],0.0, atol=1.0e-8):
                    s_12[ID] = torch.nn.Parameter(torch.tensor([0*value[0]]))
                    s_22[ID] = torch.nn.Parameter(torch.tensor([value[0]]))
                elif np.isclose(normal[1],0.0, atol=1.0e-8):
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

    return coord.clone().detach().requires_grad_(True), cell_ids, cell_ids


def Update_Coordinates(Model_u, Model_du, Mesh_u, Mesh_du):

    cell_nodes_IDs_u = Mesh_u.Connectivity
    cell_nodes_IDs_du = Mesh_du.Connectivity

    node1_coord =  torch.cat([Model_du.coordinates[int(row)-1] for row in cell_nodes_IDs_du[:,0]])
    node2_coord =  torch.cat([Model_du.coordinates[int(row)-1] for row in cell_nodes_IDs_du[:,1]])
    node3_coord =  torch.cat([Model_du.coordinates[int(row)-1] for row in cell_nodes_IDs_du[:,2]])

    for j in range(len(cell_nodes_IDs_u)):
        # print(Model_du.coordinates[int(cell_nodes_IDs_du[j,0])-1])
        # print(node1_coord[j].unsqueeze(0))
        # print()
        Model_u.coordinates[int(cell_nodes_IDs_u[j,0])-1] = node1_coord[j].unsqueeze(0)
        Model_u.coordinates[int(cell_nodes_IDs_u[j,1])-1] = node2_coord[j].unsqueeze(0)
        Model_u.coordinates[int(cell_nodes_IDs_u[j,2])-1] = node3_coord[j].unsqueeze(0)


def CopyStress(Model_u, Model_du, Domain_mesh_u, lmbda, mu):
    stress_all_coord = [(Model_du.coordinates[i]).clone().detach().requires_grad_(True) for i in range(len(Model_du.coordinates))]

    stress_all_cell_IDs = torch.tensor([torch.tensor(Domain_mesh_u.GetCellIds(i),dtype=torch.int)[0] for i in stress_all_coord])
    stress_all_coord = (torch.cat(stress_all_coord)).clone().detach().requires_grad_(True)

    u_pred = Model_u(stress_all_coord, stress_all_cell_IDs)
    
    du = torch.autograd.grad(u_pred[0,:], stress_all_coord, grad_outputs=torch.ones_like(u_pred[0,:]), create_graph=True)[0]
    dv = torch.autograd.grad(u_pred[1,:], stress_all_coord, grad_outputs=torch.ones_like(u_pred[1,:]), create_graph=True)[0]
    
    s_11, s_22, s_12 = Stress(du[:,0], dv[:,1], 0.5*(du[:,1] + dv[:,0]), lmbda, mu)

    node_IDs_s11 = []
    node_IDs_s22 = []
    node_IDs_s12 = []

    for j in range(len(Model_du.nodal_values[0])):
        if Model_du.nodal_values[0][j].requires_grad == True:
            node_IDs_s11.append(j)
        if Model_du.nodal_values[1][j].requires_grad == True:
            node_IDs_s22.append(j)
        if Model_du.nodal_values[2][j].requires_grad == True:
            node_IDs_s12.append(j)
              
    for j in node_IDs_s11:
        Model_du.nodal_values[0][j] = torch.nn.Parameter(torch.tensor([s_11[j]]))
    for j in node_IDs_s22:
        Model_du.nodal_values[1][j] = torch.nn.Parameter(torch.tensor([s_22[j]]))
    for j in node_IDs_s12:
        Model_du.nodal_values[2][j] = torch.nn.Parameter(torch.tensor([s_12[j]]))

