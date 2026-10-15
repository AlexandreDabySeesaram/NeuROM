import numpy as np
import random
import torch
import torch.utils.checkpoint
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

def Gravity_vect(theta,rho = 1e-9, dim = 2, n_angle = 1):
    """Should be unified with Gravity defined above, require chaging torch.tensor(0*torch.pi/2) to torch.tensor([0*torch.pi/2]) in 
    functions referencing the latter."""
    g = 9.81*1e3                            #m/s^2
    match dim:
        case 2:
            return (rho*g*torch.stack([torch.sin(theta),-torch.cos(theta)]))
        case 3:
            if n_angle == 1:
                return (rho*g*torch.stack([torch.sin(theta),-torch.cos(theta), 0*torch.sin(theta)]))
            else:
                theta_angle = theta[0]
                phi         = theta[1]

                # Compute the components using broadcasting

                sin_theta   = torch.sin(theta_angle).unsqueeze(1)  
                cos_theta   = torch.cos(theta_angle).unsqueeze(1)  
                cos_phi     = torch.cos(phi).unsqueeze(0)      
                sin_phi     = torch.sin(phi).unsqueeze(0) 
   

                component_2 = -cos_theta * cos_phi           
                component_1 = sin_theta * torch.ones_like(component_2)  
                component_3 = cos_theta * sin_phi           

                return (rho*g*torch.stack([component_1, component_2, component_3], dim=0))


def VolumeForcesEnergy_2D(u,theta, rho):
    fv = Gravity(theta,rho).to(u.dtype).to(u.device)
    W_e = u.t()@fv
    return torch.squeeze(W_e)

def Stress(eps, lmbda, mu):
    # Function used only for export to vtk, ordering compatible with Paraview ( 3D: XX, YY, ZZ, XY, YZ, XZ)
    components = eps.shape[1]
    match components:
        case 3:     # 2D : [eps_xx eps_yy eps_xy]
            ep_11=eps[:,0]
            ep_22=eps[:,1]
            ep_12=eps[:,2]
            tr_eps = ep_11 + ep_22
            return tr_eps*lmbda + 2*mu*ep_11, tr_eps*lmbda + 2*mu*ep_22, 2*mu*ep_12
        case 6:     # 3D : [eps_xx eps_yy eps_zz eps_xy eps_yz eps_xz]
            # Input eps: [xx, yy, zz, xy, yz, xz]
            ep_xx = eps[:,0]
            ep_yy = eps[:,1]
            ep_zz = eps[:,2]
            ep_xy = eps[:,3]
            ep_yz = eps[:,4]
            ep_xz = eps[:,5]

            tr_eps = ep_xx + ep_yy + ep_zz
            return tr_eps*lmbda + 2*mu*ep_xx, tr_eps*lmbda + 2*mu*ep_yy, tr_eps*lmbda + 2*mu*ep_zz,\
                2*mu*ep_xy, 2*mu*ep_yz, 2*mu*ep_xz

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
    print(lmbda, mu)
    # Accounts for sigma_zz = (lmbda/(2(mu+lmda)))*(sigma_xx+sigma_yy) != 0 if in plain strain
    two = torch.tensor(2,dtype=torch.float64)
    two2threeD = torch.tensor([[1, 0, 0], [0, 1, 0],[lmbda/(2*(mu+lmbda)),lmbda/(2*(mu+lmbda)),0],[0, 0, 1]],dtype=sigma.dtype, device=sigma.device)
    if sigma.shape[1] == 3:                                     # 2D: sigma_xx, sigma_yy, sigma_xy
        sigma_3D = torch.einsum('ij,ej->ei',two2threeD,sigma)   # 2D: sigma_xx, sigma_yy, sigma_zz, sigma_xy
        VM = torch.tensor([[2/3, -1/3, -1/3, 0],[-1/3, 2/3,-1/3, 0],[-1/3, -1/3,2/3, 0],[0, 0,0, torch.sqrt(two)]],dtype=sigma.dtype, device=sigma.device)
    else:                                                       # 3D: sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz
        sigma_3D = sigma
        VM = torch.tensor([[2/3, -1/3, -1/3, 0, 0, 0],[-1/3, 2/3,-1/3, 0, 0, 0],[-1/3, -1/3,2/3, 0, 0, 0],[0, 0,0, torch.sqrt(two), 0, 0], [0, 0, 0, 0, torch.sqrt(two), 0], [0, 0, 0, 0, 0, torch.sqrt(two)]],dtype=sigma.dtype, device=sigma.device)
    sigma_dev = torch.einsum('ij,ej...->ei',VM,sigma_3D) # in voigt notation 
    sigma_VM = torch.einsum('ei,ei->e',sigma_dev,sigma_dev) # in voigt notation
    return torch.sqrt((3/2)*sigma_VM)

def Stress_tensor(eps, lmbda, mu):
    K = torch.tensor([[2*mu+lmbda, lmbda, 0],[lmbda, 2*mu+lmbda, 0],[0, 0, 2*mu]],dtype=eps.dtype, device=eps.device)
    sigma = torch.einsum('ij,ej->ei',K,eps)
    return sigma

def InternalEnergy_2_3D_einsum(u,x,lmbda, mu, dim = 2, grad_u = None):
    match dim:
        case 2:
            eps =  Strain_sqrt(u,x, dim=2, grad_u=grad_u)
            K = torch.tensor([[2*mu+lmbda, lmbda, 0],[lmbda, 2*mu+lmbda, 0],[0, 0, 2*mu]],dtype=eps.dtype, device=eps.device)
            if eps.ndim == 3:
                W_e = torch.einsum('ij,ejg,eig->eg',K,eps,eps)
            else:
                W_e = torch.einsum('ij,ej,ei->e',K,eps,eps)
            return W_e
        case 3:
            eps =  Strain_sqrt(u,x, dim = dim, grad_u=grad_u)
            K = torch.tensor([[2*mu+lmbda, lmbda, lmbda, 0, 0, 0],[lmbda, 2*mu+lmbda, lmbda, 0, 0, 0], [lmbda, lmbda, 2*mu+lmbda, 0, 0, 0],[0, 0, 0, 2*mu, 0, 0],[0, 0, 0, 0, 2*mu, 0],[0, 0, 0, 0, 0, 2*mu]],dtype=eps.dtype, device=eps.device)
            if eps.ndim == 3:
                W_e = torch.einsum('ij,ejg,eig->eg',K,eps,eps)
            else:
                W_e = torch.einsum('ij,ej,ei->e',K,eps,eps)
            return W_e

InternalEnergy_2D_einsum = InternalEnergy_2_3D_einsum

def InternalEnergy_2D_einsum_NeoHookean(u,x,lmbda, mu):
    grad_u =  grad_u_2D(u,x)
    Green_lagrange_tensor = Green_lagrange(grad_u)
    Id = torch.eye(2,dtype=Green_lagrange_tensor.dtype, device=Green_lagrange_tensor.device)
    F = grad_u + Id
    C = torch.einsum('eki,ekj->eij',F,F)
    J = torch.linalg.det(F)
    tr = torch.einsum('eii->e',C)

    penalty = torch.relu(-J)

    W_e = 0.5*mu*(tr-2-2*torch.log(torch.abs(J))) + 0.5*(lmbda*(J-1)**2) + 1000*penalty

    return W_e

def InternalEnergy_2D_einsum_SaintVenantKirchhoff(u,x,lmbda, mu):
    grad_u =  grad_u_2D(u,x)
    Green_lagrange_tensor = Green_lagrange(grad_u)
    eps_green = torch.stack([Green_lagrange_tensor[:,0,0], Green_lagrange_tensor[:,1,1], (1/torch.sqrt(torch.tensor(2)))*(Green_lagrange_tensor[:,1,0])],dim=1)
    K = torch.tensor([[2*mu+lmbda, lmbda, 0],[lmbda, 2*mu+lmbda, 0],[0, 0, 2*mu]],dtype=Green_lagrange_tensor.dtype, device=Green_lagrange_tensor.device)
    W_e = torch.einsum('ij,ej,ei->e',K,eps_green,eps_green)
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

def InternalEnergy_2_3D_einsum_Bipara(model,lmbda, mu,E):

    Space_modes = []
    xg_modes = []
    detJ_modes = []
    for i in range(model.n_modes_truncated):
        u_k,xg_k,detJ_k = model.Space_modes[i]()
        Space_modes.append(u_k)
        xg_modes.append(xg_k)
        detJ_modes.append(detJ_k)

 

    detJ_i = torch.stack(detJ_modes,dim=1)  

    match model.Space_modes[0].mesh.dim:
        case 2:
            u_i = torch.stack(Space_modes,dim=2)
            xg_i = torch.stack(xg_modes,dim=2) 
            eps_list    = [Strain_sqrt(Space_modes[i],xg_modes[i]) for i in range(model.n_modes_truncated)]
            K = torch.tensor([[2*mu+lmbda, lmbda, 0],[lmbda, 2*mu+lmbda, 0],[0, 0, 2*mu]],dtype=model.float_config.dtype, device=model.float_config.device)

        case 3:
            # print(Space_modes[0].shape)
            u_i = torch.stack(Space_modes,dim=2)
            xg_i = torch.stack(xg_modes,dim=2) 
            eps_list    = [Strain_sqrt(Space_modes[i],xg_modes[i], model.Space_modes[0].mesh.dim) for i in range(model.n_modes_truncated)]
            K = torch.tensor([[2*mu+lmbda, lmbda, lmbda, 0, 0, 0],[lmbda, 2*mu+lmbda, lmbda, 0, 0, 0], [lmbda, lmbda, 2*mu+lmbda, 0, 0, 0],[0, 0, 0, 2*mu, 0, 0],[0, 0, 0, 0, 2*mu, 0],[0, 0, 0, 0, 0, 2*mu]],dtype=model.float_config.dtype, device=model.float_config.device)

    eps_i       = torch.stack(eps_list,dim=2)  
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

    # print(f"eps_i shape is {eps_i.shape}")#DEBUG
    W_int = torch.einsum('ij,ejm...,eil...,em,mp...,lp...,mt...,lt...,p->',K,eps_i,eps_i,torch.abs(detJ_i),lambda_i[0],lambda_i[0],lambda_i[1],lambda_i[1],E_float)

    Gravity_force = Gravity_vect(theta_float,rho = 1e-9, dim = model.Space_modes[0].mesh.dim).to(model.float_config.dtype).to(model.float_config.device)
    W_ext = torch.einsum('iem,it,mp...,mt...,em->',u_i,Gravity_force,lambda_i[0],lambda_i[1],torch.abs(detJ_i))

    return (0.5*W_int - W_ext)/(E[0].shape[0])

def InternalEnergy_2_3D_einsum_Tripara(model,lmbda, mu,E):

    Space_modes = []
    xg_modes = []
    detJ_modes = []
    for i in range(model.n_modes_truncated):
        u_k,xg_k,detJ_k = model.Space_modes[i]()
        Space_modes.append(u_k)
        xg_modes.append(xg_k)
        detJ_modes.append(detJ_k)

 

    detJ_i = torch.stack(detJ_modes,dim=1)  

    match model.Space_modes[0].mesh.dim:
        case 2:
            u_i = torch.stack(Space_modes,dim=2)
            xg_i = torch.stack(xg_modes,dim=2) 
            eps_list    = [Strain_sqrt(Space_modes[i],xg_modes[i]) for i in range(model.n_modes_truncated)]
            K = torch.tensor([[2*mu+lmbda, lmbda, 0],[lmbda, 2*mu+lmbda, 0],[0, 0, 2*mu]],dtype=model.float_config.dtype, device=model.float_config.device)

        case 3:
            # print(Space_modes[0].shape)
            u_i = torch.stack(Space_modes,dim=2)
            xg_i = torch.stack(xg_modes,dim=2) 
            eps_list    = [Strain_sqrt(Space_modes[i],xg_modes[i], model.Space_modes[0].mesh.dim) for i in range(model.n_modes_truncated)]
            K = torch.tensor([[2*mu+lmbda, lmbda, lmbda, 0, 0, 0],[lmbda, 2*mu+lmbda, lmbda, 0, 0, 0], [lmbda, lmbda, 2*mu+lmbda, 0, 0, 0],[0, 0, 0, 2*mu, 0, 0],[0, 0, 0, 0, 2*mu, 0],[0, 0, 0, 0, 0, 2*mu]],dtype=model.float_config.dtype, device=model.float_config.device)

    eps_i       = torch.stack(eps_list,dim=2)  
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
    phi_float = E[2][:,0]

    angles = [theta_float,phi_float]

    # print(f"eps_i shape is {eps_i.shape}")#DEBUG
    W_int = torch.einsum('ij,ejm...,eil...,em,mp...,lp...,mt...,lt...,ms...,ls...,p->',K,eps_i,eps_i,torch.abs(detJ_i),lambda_i[0],lambda_i[0],lambda_i[1],lambda_i[1],lambda_i[2],lambda_i[2],E_float)


    Gravity_force = Gravity_vect(angles,rho = 1e-9, dim = model.Space_modes[0].mesh.dim, n_angle=2).to(model.float_config.dtype).to(model.float_config.device)
    W_ext = torch.einsum('iem,its,mp...,mt...,ms...,em->',u_i,Gravity_force,lambda_i[0],lambda_i[1],lambda_i[2],torch.abs(detJ_i))

    return (0.5*W_int - W_ext)/(E[0].shape[0])

def InternalEnergy_StochasticShape(
    model, lmbda, mu, E,
    shape_indices=[1, 2, 3, 4],
    E_idx=0,
    N_mc_samples=32,
    BoundaryNormals=False
):
    orig_free = [model.Space_modes[i].coordinates['free'].clone() for i in range(model.n_modes_truncated)]
    orig_imposed = [model.Space_modes[i].coordinates['imposed'].clone() for i in range(model.n_modes_truncated)]
    
    total_loss = 0.0
    
    for mc in range(N_mc_samples):
        # Sample shapes from E
        s_vals = [random.choice(E[idx][:, 0].tolist()) for idx in shape_indices]
        
        # Warp the coordinates
        for i in range(model.n_modes_truncated):
            coord_free_mask = model.Space_modes[i].coord_free
            
            orig_all = torch.ones_like(model.Space_modes[i].coordinates_all)
            orig_all[coord_free_mask] = orig_free[i]
            orig_all[~coord_free_mask] = orig_imposed[i]
            
            warped_all = orig_all.clone()
            for k, s in enumerate(s_vals):
                warped_all = warped_all + s * model.shape_modes[k]
                
            model.Space_modes[i].coordinates['free'] = warped_all[coord_free_mask]
            model.Space_modes[i].coordinates['imposed'] = warped_all[~coord_free_mask]
            
        # Re-evaluate Spatial integral
        Space_modes = []
        xg_modes = []
        detJ_modes = []
        for i in range(model.n_modes_truncated):
            u_k,xg_k,detJ_k = model.Space_modes[i]()
            Space_modes.append(u_k)
            xg_modes.append(xg_k)
            detJ_modes.append(detJ_k)

        detJ_i = torch.stack(detJ_modes,dim=1)  

        match model.Space_modes[0].mesh.dim:
            case 2:
                u_i = torch.stack(Space_modes,dim=2)
                xg_i = torch.stack(xg_modes,dim=2) 
                eps_list = [Strain_sqrt(Space_modes[i],xg_modes[i]) for i in range(model.n_modes_truncated)]
                K = torch.tensor([[2*mu+lmbda, lmbda, 0],[lmbda, 2*mu+lmbda, 0],[0, 0, 2*mu]],dtype=model.float_config.dtype, device=model.float_config.device)

            case 3:
                u_i = torch.stack(Space_modes,dim=2)
                xg_i = torch.stack(xg_modes,dim=2) 
                eps_list = [Strain_sqrt(Space_modes[i],xg_modes[i], model.Space_modes[0].mesh.dim) for i in range(model.n_modes_truncated)]
                K = torch.tensor([[2*mu+lmbda, lmbda, lmbda, 0, 0, 0],[lmbda, 2*mu+lmbda, lmbda, 0, 0, 0], [lmbda, lmbda, 2*mu+lmbda, 0, 0, 0],[0, 0, 0, 2*mu, 0, 0],[0, 0, 0, 0, 2*mu, 0],[0, 0, 0, 0, 0, 2*mu]],dtype=model.float_config.dtype, device=model.float_config.device)

        eps_i = torch.stack(eps_list,dim=2)  
        
        I_ml = torch.einsum('ij,ejm,eil,e->ml', K, eps_i, eps_i, torch.abs(detJ_i[:, 0]))
        
        Para_mode_Lists = []
        for mode in range(model.n_modes_truncated):
            mode_list = []
            for l in range(model.n_para):
                if l in shape_indices:
                    s_tensor = torch.tensor([s_vals[shape_indices.index(l)]], dtype=model.float_config.dtype, device=model.float_config.device)
                    val = model.Para_modes[mode][l](s_tensor.view(-1, 1))[:, None]
                    mode_list.append(val)
                else:
                    val = model.Para_modes[mode][l](E[l][:, 0].view(-1, 1))[:, None]
                    mode_list.append(val)
            Para_mode_Lists.append(mode_list)
            
        lambda_i = [
            torch.cat([torch.unsqueeze(Para_mode_Lists[m][l],dim=0) for m in range(model.n_modes_truncated)], dim=0)
            for l in range(model.n_para)
        ]
        
        M_ml = torch.ones(model.n_modes_truncated, model.n_modes_truncated, device=u_i.device)
        for q in range(model.n_para):
            L_q = lambda_i[q][:, :, 0]
            if q == E_idx:
                E_float = E[q][:, 0]
                M_q = torch.einsum('p,mp,lp->ml', E_float, L_q, L_q)
            else:
                M_q = torch.einsum('mp,lp->ml', L_q, L_q)
            M_ml = M_ml * M_q
            
        W_int_sample = torch.sum(I_ml * M_ml)
        total_loss += W_int_sample

    for i in range(model.n_modes_truncated):
        model.Space_modes[i].coordinates['free'] = orig_free[i]
        model.Space_modes[i].coordinates['imposed'] = orig_imposed[i]

    denom = 1
    for i in range(model.n_para):
        if i not in shape_indices:
            denom = denom * E[i].shape[0]

    return 0.5 * (total_loss / N_mc_samples) / denom

class SymmetricQuadraticForm(torch.autograd.Function):
    """A = U K U^T for a constant symmetric sparse K, with an explicit backward (dL/dU = (G + G^T) U K), 
    which avoids differentiating through the sparse product"""
    @staticmethod
    def forward(ctx, U, K):
        KU = (K @ U.T).T
        ctx.save_for_backward(KU)
        return U @ KU.T
    @staticmethod
    def backward(ctx, G):
        KU, = ctx.saved_tensors
        return (G + G.T) @ KU, None

def LinearOperators3D(model, lmbda, mu, BoundaryNormals=True, z0=0.0, y0=0.0):
    """Sparse operators of the linear lung energy on the (frozen) P1 mesh, in the nodal layout [3*node + component]
    (cached on the mesh): stiffness K (E = 1), springs Ks (one point rule), nodal volumes (gravity) and pressure load vectors 
    f_P0, f_H, f_K (P = 1, z - z0, y - y0 at the facet centroids, one point rule). Same quadratures as the einsum path.

    The element matrices and vectors are not written by hand: they are generated by automatic differentiation of the element 
    energies (Hessian of the quadratic terms, gradient of the linear ones), vectorised over the elements with torch.func.vmap.
    Only the energy densities are written, with the same strain (Strain_sqrt convention) and stiffness matrix as the einsum path."""
    mesh = model.Space_modes[0].mesh
    key = (float(lmbda), float(mu), bool(BoundaryNormals), float(z0), float(y0), str(model.float_config.device))
    if getattr(mesh, 'linear_operators_key', None) != key:
        c = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in SVK_GeometricCache(model).items()}
        dtype, X, tets, dN, vol, tri, n, area = model.float_config.dtype, c["X"], c["tets"], c["dN"], c["vol"], c["tri"], c["n"], c["area"]
        n_dofs = 3 * len(X)
        C = torch.tensor([[2*mu+lmbda, lmbda, lmbda, 0, 0, 0],[lmbda, 2*mu+lmbda, lmbda, 0, 0, 0], [lmbda, lmbda, 2*mu+lmbda, 0, 0, 0],[0, 0, 0, 2*mu, 0, 0],[0, 0, 0, 0, 2*mu, 0],[0, 0, 0, 0, 0, 2*mu]], dtype=dtype)

        # Element energies, functions of the element nodal displacements u_e [nodes, 3]
        def strain_energy(u_e, dN_e, vol_e):                                            # 1/2 \int eps:C:eps dV (E = 1)
            g = u_e.T @ dN_e                                                            # grad u [i, j]
            eps = _voigt_sqrt(0.5 * (g + g.T))
            return 0.5 * vol_e * eps @ C @ eps
        def spring_energy(u_f, n_f, area_f):                                            # 1/2 \int (u.n)^2 dS, one point rule
            u_c = u_f.mean(dim=0)
            return 0.5 * area_f * (torch.dot(u_c, n_f)**2 if BoundaryNormals else torch.dot(u_c, u_c))
        def pressure_work(u_f, n_f, area_f, field_f):                                   # \int field u.n dS, one point rule
            return area_f * field_f * torch.dot(u_f.mean(dim=0), n_f)

        hessian = lambda f, k, *args: torch.func.vmap(torch.func.hessian(f), in_dims=(None,) + (0,)*len(args))(
                                        torch.zeros(k, 3, dtype=dtype), *args).reshape(-1, 3*k, 3*k)
        gradient = lambda f, k, *args: torch.func.vmap(torch.func.grad(f), in_dims=(None,) + (0,)*len(args))(
                                        torch.zeros(k, 3, dtype=dtype), *args).reshape(-1, 3*k)
        dofs_e = (3 * tets[:, :, None] + torch.arange(3)).reshape(-1, 12)
        dofs_f = (3 * tri[:, :, None] + torch.arange(3)).reshape(-1, 9)
        def assemble(dofs, Me):
            k = dofs.shape[1]
            idx = torch.stack([dofs.repeat_interleave(k, dim=1).reshape(-1), dofs.repeat(1, k).reshape(-1)])
            return torch.sparse_coo_tensor(idx, Me.reshape(-1), (n_dofs, n_dofs)).coalesce().to_sparse_csr()
        def assemble_vector(dofs, fe):
            return torch.zeros(n_dofs, dtype=dtype).index_add_(0, dofs.reshape(-1), fe.reshape(-1))

        xc = X[tri].mean(dim=1)
        ops = {"K":  assemble(dofs_e, hessian(strain_energy, 4, dN, vol)),
               "Ks": assemble(dofs_f, hessian(spring_energy, 3, n, area)),
               "nodal_volume": c["nodal_volume"]}
        for name, field in [("f_P0", torch.ones_like(area)), ("f_H", xc[:, 2] - z0), ("f_K", xc[:, 1] - y0)]:
            ops[name] = assemble_vector(dofs_f, gradient(pressure_work, 3, n, area, field))
        mesh.linear_operators = {k: v.to(model.float_config.device) for k, v in ops.items()}
        mesh.linear_operators_key = key
    return mesh.linear_operators

def InternalEnergy_2_3D_einsum_Multipara_PressureGravityBoundaryStiffness(
    model, lmbda, mu, E,
    E_idx=0,
    theta_idx=None, phi_idx=None,
    p0_idx=None, h_idx=None, k_idx=None,
    p0_val=0.0, h_val=0.0, k_val=0.0,
    k_spring=1e2, BoundaryNormals=True,
    z0=0.0, y0=0.0, rho=1e-9, fast_path=False
):
    r"""
    Parametric potential energy of a linear elastic body loaded by
        - gravity (direction given by the angles theta and phi),
        - a boundary pressure P(y, z) = P_0 + H*(z - z_0) + K*(y - y_0) acting along the outward normal,
        - normal (BoundaryNormals = True) or full boundary springs of stiffness k_spring (Robin BCs).

    Pi(u) = 1/2 \int eps:C(E):eps dV - \int rho g . u dV - \int P u.n dS + 1/2 \int k (u.n)^2 dS

    Every parameter (E, theta, phi, P_0, H, K) can either be an extra-coordinate of the tensor decomposition 
    (its index in E is given by the corresponding *_idx argument) or a constant (*_val argument, idx = None).
    The energy is averaged over the parametric grid, using the separability of the tensor decomposition so that 
    each parameter direction is only summed once.

    fast_path = True (3D, frozen mesh): the space integrals are computed with sparse operators assembled once
    (stiffness, springs, load vectors), I_ml = U_m^T K U_l, instead of re-evaluating every space mode, its geometry 
    and its strain (autograd) at each call. Same discrete energy (P1, same quadratures).
    """
    dim = model.Space_modes[0].mesh.dim
    n_m = model.n_modes_truncated

    if fast_path and dim == 3:
        # Space integrals with the assembled operators: U [m, nodes, 3] -> Uf [m, 3*nodes]
        ops     = LinearOperators3D(model, lmbda, mu, BoundaryNormals, z0, y0)
        U       = torch.stack([NodalValues3D(model.Space_modes[m]) for m in range(n_m)], dim=0)
        Uf      = U.reshape(n_m, -1)
        I_ml    = SymmetricQuadraticForm.apply(Uf, ops["K"])                # \int eps(U_m):C:eps(U_l) dV
        I_im    = torch.einsum('mnx,n->xm', U, ops["nodal_volume"])         # \int U_m dV
        I_b_ml  = SymmetricQuadraticForm.apply(Uf, ops["Ks"])               # \int (U_m.n)(U_l.n) dS
        I_m_pressure = [Uf @ ops[name] for name in ["f_P0", "f_H", "f_K"]]  # \int field U_m.n dS
        ref     = Uf
    else:
        # Space modes, in the volume (Gauss points) and on the boundary (one point per facet)
        u_modes, x_modes, detJ_modes = [], [], []
        u_b_modes, detJ_b_modes = [], []
        for i in range(n_m):
            u_k, xg_k, detJ_k, u_b_k, xg_b_k, detJ_b_k = model.Space_modes[i](return_boundary=True)
            u_modes.append(u_k)
            x_modes.append(xg_k)
            detJ_modes.append(detJ_k)
            u_b_modes.append(u_b_k)
            detJ_b_modes.append(detJ_b_k)

        u_i     = torch.stack(u_modes, dim=2)                                   # [x, e, m, g]
        detJ    = torch.abs(detJ_modes[0])                                      # [e, g] (identical for all modes, frozen mesh)
        u_b_i   = torch.stack(u_b_modes, dim=2)                                 # [x, f, m]
        detJ_b  = torch.abs(detJ_b_modes[0])                                    # [f]
        x_b     = xg_b_k                                                        # [f, x]

        eps_i   = torch.stack([Strain_sqrt(u_modes[i], x_modes[i], dim) for i in range(n_m)], dim=2)  # [e, voigt, m, g]
        match dim:
            case 2:
                K = torch.tensor([[2*mu+lmbda, lmbda, 0],[lmbda, 2*mu+lmbda, 0],[0, 0, 2*mu]],dtype=model.float_config.dtype, device=model.float_config.device)
            case 3:
                K = torch.tensor([[2*mu+lmbda, lmbda, lmbda, 0, 0, 0],[lmbda, 2*mu+lmbda, lmbda, 0, 0, 0], [lmbda, lmbda, 2*mu+lmbda, 0, 0, 0],[0, 0, 0, 2*mu, 0, 0],[0, 0, 0, 0, 2*mu, 0],[0, 0, 0, 0, 0, 2*mu]],dtype=model.float_config.dtype, device=model.float_config.device)

        I_ml    = torch.einsum('ij,ejmg,eilg,eg->ml', K, eps_i, eps_i, detJ)
        I_im    = torch.einsum('iemg,eg->im', u_i, detJ)                    # \int u_m dV
        un_i_b  = torch.einsum('xfm,fx->fm', u_b_i, BoundaryNormalsTensor(model).to(u_b_i.device).to(u_b_i.dtype))   # u.n [f, m]
        I_m_pressure = [torch.einsum('fm,f,f->m', un_i_b, field, detJ_b) for field in [torch.ones_like(detJ_b), x_b[:, 2] - z0, x_b[:, 1] - y0]]
        if BoundaryNormals:
            I_b_ml = torch.einsum('fm,fl,f->ml', un_i_b, un_i_b, detJ_b)
        else:
            I_b_ml = torch.einsum('xfm,xfl,f->ml', u_b_i, u_b_i, detJ_b)
        ref     = u_i

    # Parametric modes L_q[m, p], and their grid averages
    L = [
        torch.cat([model.Para_modes[m][q](E[q][:,0].view(-1,1)).view(1,-1) for m in range(n_m)], dim=0)
        for q in range(model.n_para)
        ]
    L_mean = [L_q.mean(dim=1) for L_q in L]                                 # [m]

    def mean_except(excluded):
        """Product over the parameters not in excluded of the grid average of the parametric modes"""
        S_m = torch.ones(n_m, dtype=ref.dtype, device=ref.device)
        for q in range(model.n_para):
            if q not in excluded:
                S_m = S_m * L_mean[q]
        return S_m

    def quadratic_para(weighted_idx=None):
        """Grid average of the product of two modes (with the stiffness weighting on the E direction) - [m, l]"""
        M_ml = torch.ones(n_m, n_m, dtype=ref.dtype, device=ref.device)
        for q in range(model.n_para):
            if q == weighted_idx:
                M_ml = M_ml * torch.einsum('p,mp,lp->ml', E[q][:,0], L[q], L[q]) / L[q].shape[1]
            else:
                M_ml = M_ml * torch.einsum('mp,lp->ml', L[q], L[q]) / L[q].shape[1]
        return M_ml

    #  Internal strain energy
    W_int   = torch.sum(I_ml * quadratic_para(E_idx))

    #  Gravity
    if phi_idx is not None:
        theta = E[theta_idx][:, 0] if theta_idx is not None else torch.zeros(1, dtype=ref.dtype, device=ref.device)
        g_vect = Gravity_vect([theta, E[phi_idx][:, 0]], rho=rho, dim=dim, n_angle=2).to(ref.dtype).to(ref.device)  # [x, t, s]
        L_theta = L[theta_idx] if theta_idx is not None else torch.ones(n_m, 1, dtype=ref.dtype, device=ref.device)
        S_g_m = torch.einsum('im,its,mt,ms->m', I_im, g_vect, L_theta, L[phi_idx]) / (L_theta.shape[1] * L[phi_idx].shape[1])
        W_ext_gravity = torch.sum(S_g_m * mean_except([theta_idx, phi_idx]))
    elif theta_idx is not None:
        g_vect = Gravity_vect(E[theta_idx][:, 0], rho=rho, dim=dim, n_angle=1).to(ref.dtype).to(ref.device)           # [x, t]
        S_g_m = torch.einsum('im,it,mt->m', I_im, g_vect, L[theta_idx]) / L[theta_idx].shape[1]
        W_ext_gravity = torch.sum(S_g_m * mean_except([theta_idx]))
    else:
        g_vect = Gravity_vect(torch.tensor([0.0], dtype=ref.dtype), rho=rho, dim=dim, n_angle=1).to(ref.device)[:, 0]  # [x]
        W_ext_gravity = torch.sum(torch.einsum('im,i->m', I_im, g_vect) * mean_except([]))

    #  Pressure P(y,z) = P_0 + H*(z - z_0) + K*(y - y_0)
    W_ext_pressure = 0
    for I_m, idx, val in zip(I_m_pressure, [p0_idx, h_idx, k_idx], [p0_val, h_val, k_val]):
        if idx is not None:
            W_ext_pressure = W_ext_pressure + torch.sum(I_m * torch.einsum('p,mp->m', E[idx][:,0], L[idx]) / L[idx].shape[1] * mean_except([idx]))
        elif val != 0:
            W_ext_pressure = W_ext_pressure + val * torch.sum(I_m * mean_except([]))

    #  Boundary springs (Robin)
    W_spring = k_spring * torch.sum(I_b_ml * quadratic_para())

    return 0.5 * W_int - W_ext_gravity - W_ext_pressure + 0.5 * W_spring


def BoundaryNormalsTensor(model):
    """Outward unit normals [f, x] of the boundary facets, ordered as the borders_nodes of the mesh (cached on the mesh)"""
    mesh = model.Space_modes[0].mesh
    n_facets = len(model.Space_modes[0].borders_nodes)
    if not hasattr(mesh, 'boundary_normals'):
        if hasattr(mesh, 'normals') and mesh.normals.shape[0] == n_facets:
            mesh.boundary_normals = mesh.normals.detach().clone()
        else:
            # Geometric normals, oriented outward using the tetrahedra the facets belong to
            nodes = torch.as_tensor(np.array(mesh.Nodes)[:, 1:])
            tri   = torch.as_tensor(np.array(model.Space_modes[0].borders_nodes), dtype=torch.long) - 1
            n = torch.linalg.cross(nodes[tri[:, 1]] - nodes[tri[:, 0]], nodes[tri[:, 2]] - nodes[tri[:, 0]], dim=1)
            n = n / torch.norm(n, dim=1, keepdim=True)
            tets = torch.as_tensor(np.array(mesh.Connectivity), dtype=torch.long) - 1
            facet_key = {tuple(sorted(t)): f for f, t in enumerate(tri.tolist())}
            for t in tets.tolist():
                for a in range(4):
                    face = tuple(sorted(t[:a] + t[a+1:]))
                    if face in facet_key:
                        f = facet_key[face]
                        if torch.dot(nodes[t[a]] - nodes[tri[f, 0]], n[f]) > 0:
                            n[f] = -n[f]
            mesh.boundary_normals = n
    return mesh.boundary_normals

#%% Saint Venant-Kirchhoff lung problem (gravity + follower pressure + normal springs)
#
# Separated representation u(x, p) = sum_m U_m(x) Lambda_m(p),   Lambda_m(p) = prod_q lambda_mq(p_q)
#
# Every quantity entering the energy is a polynomial of degree <= 2 in u, hence it has an exact separated representation
# on an "extended" set of modes  k in {0} U {m} U {(m,n), m <= n}, of parametric functions
#       Lambda_0 = 1,    Lambda_m,    Lambda_(m,n) = Lambda_m Lambda_n = prod_q lambda_mq lambda_nq
# e.g. the Green-Lagrange strain   E = sum_m eps(U_m) Lambda_m + sum_(m<=n) c_mn sym(grad U_m^T grad U_n) Lambda_(m,n)
# The energy being quadratic in these quantities (1/2 E:C:E, a.(x - c)/3, ...) every term reads
#       sum_(k,l) A_kl G_kl,       A_kl = space integral (computed once for all parameters),
#                                  G_kl = prod_q mean_p [w_q(p_q) mu_kq(p_q) mu_lq(p_q)]   (separable grid average)
# so the parametric solution is never evaluated on the (huge) parametric hypercube.

def ExtendedParametricModes(model, E):
    """Parametric factors mu_kq of the extended modes, for each parameter q: [K, P_q], K = 1 + M + M(M+1)/2"""
    n_m = model.n_modes_truncated
    mi, ni = torch.triu_indices(n_m, n_m, device=model.float_config.device)
    Mu = []
    for q in range(model.n_para):
        L_q = torch.cat([model.Para_modes[m][q](E[q][:,0].view(-1,1)).view(1,-1) for m in range(n_m)], dim=0)    # [M, P_q]
        Mu.append(torch.cat([torch.ones_like(L_q[:1]), L_q, L_q[mi]*L_q[ni]], dim=0))
    return Mu

def SeparableGram(Mu, E, weighted_idx=None):
    """G_kl = prod_q mean_p [w_q mu_kq mu_lq] with w_q = E[q] for q = weighted_idx and 1 otherwise"""
    G = 1
    for q, Mu_q in enumerate(Mu):
        w = E[q][:,0] if q == weighted_idx else torch.ones_like(E[q][:,0])
        G = G * torch.einsum('kp,p,lp->kl', Mu_q, w, Mu_q) / Mu_q.shape[1]
    return G

def PairModes(f, X, Y, mi, ni):
    """Pair (m <= n) coefficients of the bilinear form sum_(m,n) f(X_m, Y_n) Lambda_m Lambda_n (coefficient of Lambda_(m,n): f(X_m,Y_n) + f(X_n,Y_m), halved if m = n)"""
    out = f(X[mi], Y[ni]) + f(X[ni], Y[mi])
    return (1 - 0.5 * (mi == ni).to(out.dtype)).view(-1, *[1]*(out.dim()-1)) * out

def NodalValues3D(space_mode):
    """Nodal displacement [nodes, 3] of a 3D space mode"""
    if all(bool(getattr(space_mode, 'dofs_free_'+x).all()) for x in ['x', 'y', 'z']):          # no Dirichlet BCs
        return torch.stack([space_mode.nodal_values[x+'_free'] for x in ['x', 'y', 'z']], dim=1)
    u = torch.ones_like(space_mode.values)
    for c, x in enumerate(['x', 'y', 'z']):
        mask = getattr(space_mode, 'dofs_free_'+x)
        u[mask, c] = space_mode.nodal_values[x+'_free']
        u[~mask, c] = space_mode.nodal_values[x+'_imposed']
    return u

def SVK_GeometricCache(model):
    """Reference geometry of the (frozen) mesh: P1 shape function gradients, volumes, outward oriented facets (cached on the mesh)"""
    mesh = model.Space_modes[0].mesh
    device = model.float_config.device
    if not hasattr(mesh, 'svk_cache') or mesh.svk_cache["device"] != device:
        dtype = model.float_config.dtype
        X = torch.as_tensor(np.array(mesh.Nodes)[:, 1:], dtype=dtype)
        tets = torch.as_tensor(np.array(mesh.Connectivity), dtype=torch.long) - 1
        DN = torch.tensor([[1., 0, 0, -1], [0, 1., 0, -1], [0, 0, 1., -1]], dtype=dtype)
        J = torch.einsum('an,enx->eax', DN, X[tets])                                    # [e, xi, x]
        dN = torch.linalg.solve(J, DN.expand(len(tets), 3, 4)).transpose(1, 2)          # dN_a/dx [e, a, x]
        vol = torch.abs(torch.linalg.det(J)) / 6
        tri = torch.as_tensor(np.array(model.Space_modes[0].borders_nodes), dtype=torch.long) - 1
        n_out = BoundaryNormalsTensor(model).to(dtype).cpu()
        cr = torch.linalg.cross(X[tri[:, 1]] - X[tri[:, 0]], X[tri[:, 2]] - X[tri[:, 0]], dim=1)
        flip = torch.einsum('fx,fx->f', cr, n_out) < 0
        tri[flip] = tri[flip][:, [0, 2, 1]]                                             # (X2-X1)x(X3-X1) along the outward normal
        # Sparse gradient operator: grad u [e, j, i] = sum_n Grad[(e, j), n] u[n, i]
        rows = (3 * torch.arange(len(tets))[:, None, None] + torch.arange(3)[None, None, :]).expand(-1, 4, -1)
        cols = tets[:, :, None].expand(-1, -1, 3)
        Grad = torch.sparse_coo_tensor(torch.stack([rows.reshape(-1), cols.reshape(-1)]), dN.reshape(-1), 
                                       (3 * len(tets), len(X))).coalesce().to_sparse_csr()
        nodal_volume = torch.zeros(len(X), dtype=dtype).index_add_(0, tets.reshape(-1), vol.repeat_interleave(4) / 4)  # \int u dV = sum_n w_n u_n
        cache = {"X": X, "tets": tets, "dN": dN, "vol": vol, "tri": tri, "n": n_out.cpu(),
                 "area": 0.5 * torch.norm(cr, dim=1), "Grad": Grad, "nodal_volume": nodal_volume}
        if device.type == "mps":                                                        # no sparse tensors on MPS: gather path instead
            cache["Grad"] = None
        mesh.svk_cache = {k: (v.to(device) if v is not None else None) for k, v in cache.items()}  # computed on the CPU, used on the model device
        mesh.svk_cache["device"] = device
    return mesh.svk_cache

def _voigt_sqrt(S):
    """[..., 3, 3] symmetric -> [..., 6] (E11, E22, E33, sqrt2 E23, sqrt2 E13, sqrt2 E12), consistent with Strain_sqrt"""
    r2 = np.sqrt(2)
    return torch.stack([S[..., 0, 0], S[..., 1, 1], S[..., 2, 2], r2*S[..., 1, 2], r2*S[..., 0, 2], r2*S[..., 0, 1]], dim=-1)

def _SVK_stiffness_chunk(g, vol, C, mi, ni):
    """Space operator A_kl = int E_k : C : E_l over a chunk of elements, for the extended strain modes k,l in {m} U {(m,n)}
    g = grad U_m [m, e, i, j] on the chunk"""
    r2 = np.sqrt(2)
    E_lin = _voigt_sqrt(0.5 * (g + g.transpose(-1, -2)))                                # eps(U_m)  [m, e, 6]
    # Pair modes of 1/2 grad u^T grad u: 1/2 (grad U_m^T grad U_n + grad U_n^T grad U_m) (halved for m = n), component-wise
    T = g[mi].transpose(-1, -2) @ g[ni]                                                  # grad U_m^T grad U_n  [pairs, e, 3, 3]
    S = 0.5 * (T + T.transpose(-1, -2)) * (1 - 0.5 * (mi == ni).to(g.dtype))[:, None, None, None]
    E_quad = _voigt_sqrt(S)                                                             # [pairs, e, 6]
    E_k = torch.cat([E_lin, E_quad], dim=0)                                             # [K-1, e, 6]
    CE_k = torch.einsum('kev,vw,e->kew', E_k, C, vol)
    return E_k.reshape(E_k.shape[0], -1) @ CE_k.reshape(E_k.shape[0], -1).T             # single GEMM

def InternalEnergy_3D_einsum_Multipara_PressureGravityBoundaryStiffness_SVK(
    model, lmbda, mu, E,
    E_idx=0,
    theta_idx=None, phi_idx=None,
    p0_idx=None, h_idx=None, k_idx=None,
    p0_val=0.0, h_val=0.0, k_val=0.0,
    k_spring=1e2, BoundaryNormals=True,
    z0=0.0, y0=0.0, rho=1e-9, n_chunks=8, checkpoint=False
):
    r"""
    Parametric potential energy of a Saint Venant-Kirchhoff body (finite strains) loaded by
        - gravity (dead load, direction given by the angles theta and phi),
        - a follower pressure P(x) = P_0 + H*(z - z_0) + K*(y - y_0), function of the current position x, acting along the current outward normal,
        - normal springs (Robin BCs) on the reference boundary.

    Pi(u) = \int 1/2 E:C(E):E dV - \int rho g . u dV + 1/2 \int k (u.N)^2 dS - \int_{Omega_t} P(x) dv

    The follower pressure derives from the potential - \int_{Omega_t} P dv  (its variation is - \int_{d Omega_t} P du.n da).
    With Phi such that div(Phi) = P, i.e. Phi = (P_0 (x - c)/3,  K/2 (y - y_0)^2 + P_0 (y - c_y)/3,  H/2 (z - z_0)^2 + P_0 (z - c_z)/3),
    it is computed on the current (P1, hence polyhedral) boundary:
        \int_{Omega_t} P dv = sum_facets a_f . mean_f(Phi),     a_f = 1/2 (x_2 - x_1) x (x_3 - x_1),
    which is exact (mean_f of the quadratic Phi is exact) and only involves the boundary facets.

    All the terms are written with the extended modes described above (see ExtendedParametricModes), the parametric
    grid averages being separable, so that the parametric solution is never evaluated on the full hypercube.
    """
    n_m     = model.n_modes_truncated
    mi, ni  = torch.triu_indices(n_m, n_m, device=model.float_config.device)
    cache   = SVK_GeometricCache(model)
    dtype   = model.float_config.dtype
    U       = torch.stack([NodalValues3D(model.Space_modes[m]) for m in range(n_m)], dim=0)      # [m, nodes, 3]
    Mu      = ExtendedParametricModes(model, E)
    G_1     = SeparableGram(Mu, E)                                                               # plain grid averages

    #  Internal energy 1/2 \int E:C:E (weighted by the stiffness parameter), chunked over the elements 
    #  (checkpoint = True recomputes the chunks during the backward pass: less memory, slower)
    K = torch.tensor([[2*mu+lmbda, lmbda, lmbda, 0, 0, 0],[lmbda, 2*mu+lmbda, lmbda, 0, 0, 0], [lmbda, lmbda, 2*mu+lmbda, 0, 0, 0],[0, 0, 0, 2*mu, 0, 0],[0, 0, 0, 0, 2*mu, 0],[0, 0, 0, 0, 0, 2*mu]],dtype=dtype, device=U.device)
    n_e = len(cache["tets"])
    if cache["Grad"] is not None:
        g = (cache["Grad"] @ U.permute(1, 0, 2).reshape(U.shape[1], -1)).view(n_e, 3, n_m, 3).permute(2, 0, 3, 1)   # grad U_m [m, e, i, j]
    else:
        g = torch.einsum('meai,eaj->meij', U[:, cache["tets"]], cache["dN"])
    A = 0
    for ids in torch.chunk(torch.arange(n_e), n_chunks):
        if checkpoint:
            A = A + torch.utils.checkpoint.checkpoint(_SVK_stiffness_chunk, g[:, ids], cache["vol"][ids], K, mi, ni, use_reentrant=False)
        else:
            A = A + _SVK_stiffness_chunk(g[:, ids], cache["vol"][ids], K, mi, ni)
    W_int = torch.sum(A * SeparableGram(Mu, E, E_idx)[1:, 1:])

    #  Gravity (dead load, linear in u)
    L       = [Mu_q[1:n_m+1] for Mu_q in Mu]
    L_mean  = [L_q.mean(dim=1) for L_q in L]
    def mean_except(excluded):
        S_m = torch.ones(n_m, dtype=dtype, device=U.device)
        for q in range(model.n_para):
            if q not in excluded:
                S_m = S_m * L_mean[q]
        return S_m
    I_im = torch.einsum('mnx,n->xm', U, cache["nodal_volume"])                                  # \int U_m dV
    if phi_idx is not None:
        theta = E[theta_idx][:, 0] if theta_idx is not None else torch.zeros(1, dtype=dtype, device=U.device)
        g_vect = Gravity_vect([theta, E[phi_idx][:, 0]], rho=rho, dim=3, n_angle=2).to(dtype).to(U.device)
        L_theta = L[theta_idx] if theta_idx is not None else torch.ones(n_m, 1, dtype=dtype, device=U.device)
        S_g_m = torch.einsum('im,its,mt,ms->m', I_im, g_vect, L_theta, L[phi_idx]) / (L_theta.shape[1] * L[phi_idx].shape[1])
        W_ext_gravity = torch.sum(S_g_m * mean_except([theta_idx, phi_idx]))
    elif theta_idx is not None:
        g_vect = Gravity_vect(E[theta_idx][:, 0], rho=rho, dim=3, n_angle=1).to(dtype).to(U.device)
        S_g_m = torch.einsum('im,it,mt->m', I_im, g_vect, L[theta_idx]) / L[theta_idx].shape[1]
        W_ext_gravity = torch.sum(S_g_m * mean_except([theta_idx]))
    else:
        g_vect = Gravity_vect(torch.tensor([0.0], dtype=dtype), rho=rho, dim=3, n_angle=1).to(U.device)[:, 0]
        W_ext_gravity = torch.sum(torch.einsum('im,i->m', I_im, g_vect) * mean_except([]))

    #  Springs on the reference boundary, one point (centroid) quadrature as in the linear case
    tri     = cache["tri"]
    U_f     = U[:, tri]                                                                           # [m, f, 3 nodes, 3]
    u_c     = U_f.mean(dim=2)                                                                     # [m, f, 3]
    if BoundaryNormals:
        un = torch.einsum('mfx,fx->mf', u_c, cache["n"])
        I_b = torch.einsum('mf,lf,f->ml', un, un, cache["area"])
    else:
        I_b = torch.einsum('mfx,lfx,f->ml', u_c, u_c, cache["area"])
    W_spring = k_spring * torch.sum(I_b * G_1[1:n_m+1, 1:n_m+1])

    #  Follower pressure: - \int_{Omega_t} P dv = - sum_f a_f . mean_f(Phi), every factor written on the extended modes
    X_f = cache["X"][tri]                                                                         # [f, 3 nodes, 3]
    D, Ec = X_f[:, 1] - X_f[:, 0], X_f[:, 2] - X_f[:, 0]
    d, e = U_f[:, :, 1] - U_f[:, :, 0], U_f[:, :, 2] - U_f[:, :, 0]                               # [m, f, 3]
    cross = lambda a, b: torch.linalg.cross(a, b, dim=-1)
    a_k = torch.cat([   0.5 * cross(D, Ec)[None],
                        0.5 * (cross(d, Ec.expand_as(d)) + cross(D.expand_as(e), e)),
                        0.5 * PairModes(cross, d, e, mi, ni)], dim=0)                             # current area vectors [K, f, 3]
    zeros_pairs = torch.zeros(len(mi), *X_f.shape[:1], 3, dtype=dtype, device=U.device)
    c = torch.tensor([cache["X"][:, 0].mean().item(), y0, z0], dtype=dtype, device=U.device)
    Phi_P0 = torch.cat([((X_f.mean(dim=1) - c) / 3)[None], u_c / 3, zeros_pairs], dim=0)         # [K, f, 3]

    def half_mean_square(comp, ref):
        """Modes of 1/2 mean_f (x_comp - ref)^2, the facet mean of a quadratic being q(s,s) = (sum s_i^2 + (sum s_i)^2)/12"""
        S = X_f[:, :, comp] - ref                                                                 # [f, 3 nodes]
        w = U_f[..., comp]                                                                        # [m, f, 3 nodes]
        q = lambda a, b: (torch.sum(a * b, dim=-1) + torch.sum(a, dim=-1) * torch.sum(b, dim=-1)) / 12
        modes = torch.cat([q(S, S)[None], 2 * q(S[None], w), PairModes(q, w, w, mi, ni)], dim=0)  # [K, f]
        Phi = torch.zeros(*modes.shape, 3, dtype=dtype, device=U.device)
        Phi[..., comp] = 0.5 * modes
        return Phi
    Phi_H = half_mean_square(2, z0)
    Phi_K = half_mean_square(1, y0)

    no_constant = torch.ones_like(G_1)
    no_constant[0, 0] = 0                                                                         # drop the constant \int_{Omega_0} P dV
    W_ext_pressure = 0
    for Phi, idx, val in [(Phi_P0, p0_idx, p0_val), (Phi_H, h_idx, h_val), (Phi_K, k_idx, k_val)]:
        B = torch.einsum('kfx,lfx->kl', a_k, Phi)
        if idx is not None:
            W_ext_pressure = W_ext_pressure + torch.sum(B * SeparableGram(Mu, E, idx) * no_constant)
        elif val != 0:
            W_ext_pressure = W_ext_pressure + val * torch.sum(B * G_1 * no_constant)

    return 0.5 * W_int - W_ext_gravity + 0.5 * W_spring - W_ext_pressure


def InternalEnergy_2_3D_einsum_Tripara_BoundaryStiffness(model,lmbda, mu,E, k_spring=1e2, BoundaryNormals = False):

    Space_modes = []
    xg_modes = []
    detJ_modes = []
    for i in range(model.n_modes_truncated):
        u_k,xg_k,detJ_k = model.Space_modes[i]()
        Space_modes.append(u_k)
        xg_modes.append(xg_k)
        detJ_modes.append(detJ_k)

 

    detJ_i = torch.stack(detJ_modes,dim=1)  

    match model.Space_modes[0].mesh.dim:
        case 2:
            u_i = torch.stack(Space_modes,dim=2)
            xg_i = torch.stack(xg_modes,dim=2) 
            eps_list    = [Strain_sqrt(Space_modes[i],xg_modes[i]) for i in range(model.n_modes_truncated)]
            K = torch.tensor([[2*mu+lmbda, lmbda, 0],[lmbda, 2*mu+lmbda, 0],[0, 0, 2*mu]],dtype=model.float_config.dtype, device=model.float_config.device)

        case 3:
            # print(Space_modes[0].shape)
            u_i = torch.stack(Space_modes,dim=2)
            xg_i = torch.stack(xg_modes,dim=2) 
            eps_list    = [Strain_sqrt(Space_modes[i],xg_modes[i], model.Space_modes[0].mesh.dim) for i in range(model.n_modes_truncated)]
            K = torch.tensor([[2*mu+lmbda, lmbda, lmbda, 0, 0, 0],[lmbda, 2*mu+lmbda, lmbda, 0, 0, 0], [lmbda, lmbda, 2*mu+lmbda, 0, 0, 0],[0, 0, 0, 2*mu, 0, 0],[0, 0, 0, 0, 2*mu, 0],[0, 0, 0, 0, 0, 2*mu]],dtype=model.float_config.dtype, device=model.float_config.device)

    eps_i       = torch.stack(eps_list,dim=2)  
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
    phi_float = E[2][:,0]

    angles = [theta_float,phi_float]

    # print(f"eps_i shape is {eps_i.shape}")#DEBUG
    W_int = torch.einsum('ij,ejm...,eil...,em,mp...,lp...,mt...,lt...,ms...,ls...,p->',K,eps_i,eps_i,torch.abs(detJ_i),lambda_i[0],lambda_i[0],lambda_i[1],lambda_i[1],lambda_i[2],lambda_i[2],E_float)


    Gravity_force = Gravity_vect(angles,rho = 1e-9, dim = model.Space_modes[0].mesh.dim, n_angle=2).to(model.float_config.dtype).to(model.float_config.device)
    W_ext = torch.einsum('iem,its,mp...,mt...,ms...,em->',u_i,Gravity_force,lambda_i[0],lambda_i[1],lambda_i[2],torch.abs(detJ_i))


    # k = 1e5 # Boundary stiffness 

    if BoundaryNormals:
        W_Boundary = BoundaryStiffnessEnergy_para_normal(model, k_spring, E, u_ref=0)
    else: 
        W_Boundary = BoundaryStiffnessEnergy_para(model, k_spring, E, u_ref=0)

    return (0.5*W_int - W_ext + W_Boundary)/(E[0].shape[0])

InternalEnergy_2D_einsum_Bipara = InternalEnergy_2_3D_einsum_Bipara

def InternalEnergy_2D_einsum_Bipara_NeoHookean(model,lmbda, mu,E, kappa = 100):

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
    # grad_u_list = [grad_u_2D(Space_modes[i],xg_modes[i]) for i in range(model.n_modes_truncated)]
    grad_u_list = [grad_u_2_3D(Space_modes[i],xg_modes[i], dim = model.Space_modes[0].mesh.dim) for i in range(model.n_modes_truncated)]



    grad_u_i = torch.stack(grad_u_list,dim=3)  
    Para_mode_Lists = [
        [model.Para_modes[mode][l](E[l][:,0].view(-1,1))[:,None] for l in range(model.n_para)]
        for mode in range(model.n_modes_truncated)
        ]
    lambda_i = [
            torch.cat([torch.unsqueeze(Para_mode_Lists[m][l],dim=0) for m in range(model.n_modes_truncated)], dim=0)
            for l in range(model.n_para)
        ]    
    mu_lame = E[0][:,0]/3
    theta_float = E[1][:,0]


######################################### DEBUG #############################

    lambda_i = [torch.ones(1,1,1),torch.ones(1,1,1)]
    mu_lame = torch.ones_like(mu_lame)
# E_green_lagrange = Green_lagrange(grad_u_i[:,:,:,0])

# tr_E_E = torch.einsum("eij,eji->e",E_green_lagrange,E_green_lagrange)
# int_tr_E_E = torch.einsum("e, e->",tr_E_E,torch.abs(detJ_i)[:,0])


# tr_E_squared = torch.pow(torch.einsum("eii->e",E_green_lagrange),2)
# int_trE_squared = torch.einsum("e, e->",tr_E_squared,torch.abs(detJ_i)[:,0])


# tr_E = torch.einsum("eii->e",E_green_lagrange)
# tr_E_squared = torch.pow(tr_E,2)
# int_trE_squared = torch.einsum("e, e->",tr_E_squared,torch.abs(detJ_i)[:,0])



####Parametrised 


#     tr_gtg_e = torch.einsum('exym,exyl,mp...,lp...,mt...,lt...->e',grad_u_i,grad_u_i,lambda_i[0],lambda_i[0],lambda_i[1],lambda_i[1]) # trace of tgradu.gradu * mu
#     tr_g_e = torch.einsum('exxm,mp...,mt...->e',grad_u_i,lambda_i[0],lambda_i[1]) # trace of gradu * mu


#     tr_gtg_e_np = torch.einsum('exy,exy->e',grad_u_i[:,:,:,0],grad_u_i[:,:,:,0]) # trace of gradu * mu

#     tr_g_e_np = torch.einsum('exx->e',grad_u_i[:,:,:,0]) # trace of gradu * mu

#     int_tr_gtg_e = torch.einsum('e,e->',tr_gtg_e,torch.abs(detJ_i)[:,0])
#     int_tr_g_e = torch.einsum('e,e->',tr_g_e,torch.abs(detJ_i)[:,0])

# #####



#     tr_C = torch.einsum('eki,eki -> e',grad_u_i[:,:,:,0],grad_u_i[:,:,:,0])
#     tr_C_2 = tr_C-2 
#     int_tr_C_2 = torch.einsum("e, e->",tr_C_2,torch.abs(detJ_i)[:,0])
#     F = grad_u_i[:,:,:,0] + id_tensor[None,:,:]

#     C = torch.einsum('eki,ekj -> eij',F,F)
    

#     levi_civitas = torch.tensor([[0., 1.0],[-1.0, 0.0]], dtype = grad_u_i.dtype, device = grad_u_i.device)
#     id_tensor = torch.tensor([[1., 0.0],[0.0, 1.0]], dtype = grad_u_i.dtype, device = grad_u_i.device)

#     C_I = C - id_tensor[None,:,:]
#     tr_C_I = torch.einsum('exx -> e', C_I)
#     int_tr_C_I = torch.einsum("e, e->",tr_C_I,torch.abs(detJ_i)[:,0])


#     det_F_1 = 0.5*torch.einsum('ij,kl,eik,ejl -> e', levi_civitas, levi_civitas, F, F)-1
#     det_F_1_squarred = det_F_1**2

#     int_det_F_1_squarred = torch.einsum("e, e->",det_F_1_squarred,torch.abs(detJ_i)[:,0])
############################################################################


    tr_gtg = torch.einsum('exym,exyl,em,mp...,lp...,mt...,lt...,p->',grad_u_i,grad_u_i,torch.abs(detJ_i),lambda_i[0],lambda_i[0],lambda_i[1],lambda_i[1],mu_lame) # trace of tgradu.gradu * mu
    tr_g = torch.einsum('exxm,em,mp...,mt...,p->',grad_u_i,torch.abs(detJ_i),lambda_i[0],lambda_i[1],mu_lame) # trace of gradu * mu
    W_int = (tr_gtg+2*tr_g)

    # Gravity_force = Gravity_vect(theta_float,rho = 1e-9).to(model.float_config.dtype).to(model.float_config.device)
    Gravity_force = Gravity_vect(theta_float,rho = 1e-9, dim = model.Space_modes[0].mesh.dim).to(model.float_config.dtype).to(model.float_config.device)
    W_ext = torch.einsum('iem,it,mp...,mt...,em->',u_i,Gravity_force,lambda_i[0],lambda_i[1],torch.abs(detJ_i))

    if kappa == 0: 
         Constraint = torch.tensor(kappa)
    else:
        C1 = torch.einsum('ei,ei,ip...,it...,ej,jp...,jt...,ek,kp...,kt...,el,lp...,lt...->', torch.abs(detJ_i),grad_u_i[:,0,0,:],lambda_i[0],lambda_i[1],grad_u_i[:,1,1,:],lambda_i[0],lambda_i[1],grad_u_i[:,0,0,:],lambda_i[0],lambda_i[1],grad_u_i[:,1,1,:],lambda_i[0],lambda_i[1])
        C2 = 2*torch.einsum('ei,ei,ip...,it...,ej,jp...,jt...,ek,kp...,kt...->', torch.abs(detJ_i),grad_u_i[:,0,0,:],lambda_i[0],lambda_i[1],grad_u_i[:,1,1,:],lambda_i[0],lambda_i[1],grad_u_i[:,0,0,:],lambda_i[0],lambda_i[1])
        C3 = 2*torch.einsum('ei,ei,ip...,it...,ej,jp...,jt...,ek,kp...,kt...->', torch.abs(detJ_i),grad_u_i[:,0,0,:],lambda_i[0],lambda_i[1],grad_u_i[:,1,1,:],lambda_i[0],lambda_i[1],grad_u_i[:,1,1,:],lambda_i[0],lambda_i[1])
        C4 = -2*torch.einsum('ei,ei,ip...,it...,ej,jp...,jt...,ek,kp...,kt...,el,lp...,lt...->',torch.abs(detJ_i),grad_u_i[:,0,0,:],lambda_i[0],lambda_i[1],grad_u_i[:,1,1,:],lambda_i[0],lambda_i[1], grad_u_i[:,0,1,:],lambda_i[0],lambda_i[1],grad_u_i[:,1,0,:],lambda_i[0],lambda_i[1])

        C5 = torch.einsum('ei,ei,ip...,it...,ej,jp...,jt...->', torch.abs(detJ_i),grad_u_i[:,0,0,:],lambda_i[0],lambda_i[1],grad_u_i[:,0,0,:],lambda_i[0],lambda_i[1])
        C6 = 2*torch.einsum('ei,ei,ip...,it...,ej,jp...,jt...->', torch.abs(detJ_i),grad_u_i[:,0,0,:],lambda_i[0],lambda_i[1],grad_u_i[:,1,1,:],lambda_i[0],lambda_i[1])
        C7 = -2*torch.einsum('ek,ek,kp...,kt...,ei,ip...,it...,ej,jp...,jt...->',torch.abs(detJ_i),grad_u_i[:,0,0,:],lambda_i[0],lambda_i[1],grad_u_i[:,0,1,:],lambda_i[0],lambda_i[1],grad_u_i[:,1,0,:],lambda_i[0],lambda_i[1])

        C8 = torch.einsum('ei,ei,ip...,it...,ej,jp...,jt...->', torch.abs(detJ_i),grad_u_i[:,1,1,:],lambda_i[0],lambda_i[1],grad_u_i[:,1,1,:],lambda_i[0],lambda_i[1])
        C9 = -2*torch.einsum('ek,ek,kp...,kt...,ei,ip...,it...,ej,jp...,jt...->', torch.abs(detJ_i),grad_u_i[:,1,1,:],lambda_i[0],lambda_i[1], grad_u_i[:,0,1,:],lambda_i[0],lambda_i[1],grad_u_i[:,1,0,:],lambda_i[0],lambda_i[1])

        C10 = torch.einsum('ei,ei,ip...,it...,ej,jp...,jt...,ek,kp...,kt...,el,lp...,lt...->', torch.abs(detJ_i),grad_u_i[:,0,1,:],lambda_i[0],lambda_i[1],grad_u_i[:,1,0,:],lambda_i[0],lambda_i[1],grad_u_i[:,0,1,:],lambda_i[0],lambda_i[1],grad_u_i[:,1,0,:],lambda_i[0],lambda_i[1])
        Constraint = (C1+C2+C3+C4+C5+C6+C7+C8+C9+C10)


    return (0.5*W_int - W_ext)/(E[0].shape[0]*E[1].shape[0]), Constraint/(E[0].shape[0]*E[1].shape[0])
    # return (0.5*W_int)/(E[0].shape[0])



def InternalEnergy_2D_einsum_Bipara_KirchhoffSaintVenant(model,lmbda, mu,E):

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
    # grad_u_list = [grad_u_2D(Space_modes[i],xg_modes[i]) for i in range(model.n_modes_truncated)]
    grad_u_list = [grad_u_2_3D(Space_modes[i],xg_modes[i], dim = model.Space_modes[0].mesh.dim) for i in range(model.n_modes_truncated)]


    grad_u_i = torch.stack(grad_u_list,dim=3)  
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

######################################### DEBUG #############################

    # lambda_i = [torch.ones(1,1,1),torch.ones(1,1,1)]

    # E_green_lagrange = Green_lagrange(grad_u_i[:,:,:,0])

    # tr_E_E = torch.einsum("eij,eji->e",E_green_lagrange,E_green_lagrange)
    # int_tr_E_E = torch.einsum("e, e->",tr_E_E,torch.abs(detJ_i)[:,0])


    # tr_E_squared = torch.pow(torch.einsum("eii->e",E_green_lagrange),2)
    # int_trE_squared = torch.einsum("e, e->",tr_E_squared,torch.abs(detJ_i)[:,0])


    # tr_E = torch.einsum("eii->e",E_green_lagrange)
    # tr_E_squared = torch.pow(tr_E,2)
    # int_trE_squared = torch.einsum("e, e->",tr_E_squared,torch.abs(detJ_i)[:,0])


############################################################################
    # print(f"shape grad_u_i {grad_u_i.shape}")#DEBUG
    import time
    t_0 = time.time()
    tr_EE_A = 0.5*torch.einsum('em,exym,mp...,mt...,eyxl,lp...,lt...,p->',torch.abs(detJ_i),grad_u_i,lambda_i[0],lambda_i[1],grad_u_i,lambda_i[0],lambda_i[1],E_float) 
    t_end = time.time()
    # print(f'Duration C_1 (ms): {1000*(t_end-t_0)}')
    t_0 = time.time()

    tr_EE_B = 0.5*torch.einsum('em,exym,mp...,mt...,exyl,lp...,lt...,p->',torch.abs(detJ_i),grad_u_i,lambda_i[0],lambda_i[1],grad_u_i,lambda_i[0],lambda_i[1],E_float) 
    t_end = time.time()
    # print(f'Duration C_2 (ms): {1000*(t_end-t_0)}')
    t_0 = time.time()

    tr_EE_C = torch.einsum('em,ezxm,mp...,mt...,ezyl,lp...,lt...,eyxn,np...,nt...,p->',torch.abs(detJ_i),grad_u_i,lambda_i[0],lambda_i[1],grad_u_i,lambda_i[0],lambda_i[1],grad_u_i,lambda_i[0],lambda_i[1],E_float) 
    t_end = time.time()
    # print(f'Duration C_3 (ms): {1000*(t_end-t_0)}')
    t_0 = time.time()

    tr_EE_D = 0.25*torch.einsum('em,ezxm,mp...,mt...,ezyl,lp...,lt...,ewyn,np...,nt...,ewxo,op...,ot...,p->',torch.abs(detJ_i),grad_u_i,lambda_i[0],lambda_i[1],grad_u_i,lambda_i[0],lambda_i[1],grad_u_i,lambda_i[0],lambda_i[1],grad_u_i,lambda_i[0],lambda_i[1],E_float) 
    t_end = time.time()
    # print(f'Duration C_4 (ms): {1000*(t_end-t_0)}')
    t_0 = time.time()
 
    int_trEE_para = tr_EE_A+tr_EE_B+tr_EE_C+tr_EE_D


    tr_E_squared_E = torch.einsum('em,exxm,mp...,mt...,eXXl,lp...,lt...,p->',torch.abs(detJ_i),grad_u_i,lambda_i[0],lambda_i[1],grad_u_i,lambda_i[0],lambda_i[1],E_float) 
    tr_E_squared_F = torch.einsum('em,exxm,mp...,mt...,eXYl,lp...,lt...,eXYn,np...,nt...,p->',torch.abs(detJ_i),grad_u_i,lambda_i[0],lambda_i[1],grad_u_i,lambda_i[0],lambda_i[1],grad_u_i,lambda_i[0],lambda_i[1],E_float) 
    tr_E_squared_G = 0.25*torch.einsum('em,eyxm,mp...,mt...,eyxl,lp...,lt...,eYXn,np...,nt...,eYXo,op...,ot...,p->',torch.abs(detJ_i),grad_u_i,lambda_i[0],lambda_i[1],grad_u_i,lambda_i[0],lambda_i[1],grad_u_i,lambda_i[0],lambda_i[1],grad_u_i,lambda_i[0],lambda_i[1],E_float) 
    int_trE_squared_para = tr_E_squared_E+tr_E_squared_F+tr_E_squared_G



    W_int = mu*int_trEE_para + 0.5*lmbda*int_trE_squared_para

    # Gravity_force = Gravity_vect(theta_float,rho = 1e-9).to(model.float_config.dtype).to(model.float_config.device)
    Gravity_force = Gravity_vect(theta_float,rho = 1e-9, dim = model.Space_modes[0].mesh.dim).to(model.float_config.dtype).to(model.float_config.device)
    W_ext = torch.einsum('iem,it,mp...,mt...,em->',u_i,Gravity_force,lambda_i[0],lambda_i[1],torch.abs(detJ_i))

    return (W_int - W_ext)/(E[0].shape[0]*E[1].shape[0])




def InternalEnergy_2D_einsum_Tripara_KirchhoffSaintVenant(model,lmbda, mu,E):

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
    # grad_u_list = [grad_u_2D(Space_modes[i],xg_modes[i]) for i in range(model.n_modes_truncated)]
    grad_u_list = [grad_u_2_3D(Space_modes[i],xg_modes[i], dim = model.Space_modes[0].mesh.dim) for i in range(model.n_modes_truncated)]


    grad_u_i = torch.stack(grad_u_list,dim=3)  
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
    phi_float = E[2][:,0]

    angles = [theta_float,phi_float]


    tr_EE_A = 0.5*torch.einsum('em,exym,mp...,mt...,ms...,eyxl,lp...,lt...,ls...,p->',torch.abs(detJ_i),grad_u_i,lambda_i[0],lambda_i[1],lambda_i[2],grad_u_i,lambda_i[0],lambda_i[1],lambda_i[2],E_float) 


    tr_EE_B = 0.5*torch.einsum('em,exym,mp...,mt...,ms...,exyl,lp...,lt...,ls...,p->',torch.abs(detJ_i),grad_u_i,lambda_i[0],lambda_i[1],lambda_i[2],grad_u_i,lambda_i[0],lambda_i[1],lambda_i[2],E_float) 


    tr_EE_C = torch.einsum('em,ezxm,mp...,mt...,ms...,ezyl,lp...,lt...,ls...,eyxn,np...,nt...,ns...,p->',torch.abs(detJ_i),grad_u_i,lambda_i[0],lambda_i[1],lambda_i[2],grad_u_i,lambda_i[0],lambda_i[1],lambda_i[2],grad_u_i,lambda_i[0],lambda_i[1],lambda_i[2],E_float) 


    tr_EE_D = 0.25*torch.einsum('em,ezxm,mp...,mt...,ms...,ezyl,lp...,lt...,ls...,ewyn,np...,nt...,ns...,ewxo,op...,ot...,os...,p->',torch.abs(detJ_i),grad_u_i,lambda_i[0],lambda_i[1],lambda_i[2],grad_u_i,lambda_i[0],lambda_i[1],lambda_i[2],grad_u_i,lambda_i[0],lambda_i[1],lambda_i[2],grad_u_i,lambda_i[0],lambda_i[1],lambda_i[2],E_float) 

 
    int_trEE_para = tr_EE_A+tr_EE_B+tr_EE_C+tr_EE_D


    tr_E_squared_E = torch.einsum('em,exxm,mp...,mt...,ms...,eXXl,lp...,lt...,ls...,p->',torch.abs(detJ_i),grad_u_i,lambda_i[0],lambda_i[1],lambda_i[2],grad_u_i,lambda_i[0],lambda_i[1],lambda_i[2],E_float) 
    tr_E_squared_F = torch.einsum('em,exxm,mp...,mt...,ms...,eXYl,lp...,lt...,ls...,eXYn,np...,nt...,ns...,p->',torch.abs(detJ_i),grad_u_i,lambda_i[0],lambda_i[1],lambda_i[2],grad_u_i,lambda_i[0],lambda_i[1],lambda_i[2],grad_u_i,lambda_i[0],lambda_i[1],lambda_i[2],E_float) 
    tr_E_squared_G = 0.25*torch.einsum('em,eyxm,mp...,mt...,ms...,eyxl,lp...,lt...,ls...,eYXn,np...,nt...,ns...,eYXo,op...,ot...,os...,p->',torch.abs(detJ_i),grad_u_i,lambda_i[0],lambda_i[1],lambda_i[2],grad_u_i,lambda_i[0],lambda_i[1],lambda_i[2],grad_u_i,lambda_i[0],lambda_i[1],lambda_i[2],grad_u_i,lambda_i[0],lambda_i[1],lambda_i[2],E_float) 
    int_trE_squared_para = tr_E_squared_E+tr_E_squared_F+tr_E_squared_G



    W_int = mu*int_trEE_para + 0.5*lmbda*int_trE_squared_para

    # Gravity_force = Gravity_vect(theta_float,rho = 1e-9).to(model.float_config.dtype).to(model.float_config.device)
    Gravity_force = Gravity_vect(angles,rho = 1e-9, dim = model.Space_modes[0].mesh.dim, n_angle=2).to(model.float_config.dtype).to(model.float_config.device)
    W_ext = torch.einsum('iem,its,mp...,mt...,ms...,em->',u_i,Gravity_force,lambda_i[0],lambda_i[1],lambda_i[2],torch.abs(detJ_i))

    return (W_int - W_ext)/(E[0].shape[0]*E[1].shape[0])













def InternalEnergy_2_3D_einsum_BiStiffness(model,lmbda, mu,E):

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

    # eps_list = [Strain_sqrt(Space_modes[i],xg_modes[i]) for i in range(model.n_modes_truncated)]

    match model.Space_modes[0].mesh.dim:
        case 2:
            u_i = torch.stack(Space_modes,dim=2)
            xg_i = torch.stack(xg_modes,dim=2) 
            eps_list    = [Strain_sqrt(Space_modes[i],xg_modes[i]) for i in range(model.n_modes_truncated)]
            K = torch.tensor([[2*mu+lmbda, lmbda, 0],[lmbda, 2*mu+lmbda, 0],[0, 0, 2*mu]],dtype=model.float_config.dtype, device=model.float_config.device)

            # To be replaced with the decoder in the full auto-encoder framework
            support = (1+torch.tanh(xg_k[:,None,1] - E[1][None,:,0]))*0.5
            # with torch.no_grad(): #No derivatino of the heavyside function
            #     support = (torch.heaviside(xg_k[:,None,0] - E[1][None,:,0],torch.tensor(1, dtype = torch.float64)))

        case 3:
            # print(Space_modes[0].shape)
            u_i = torch.stack(Space_modes,dim=2)
            xg_i = torch.stack(xg_modes,dim=2) 
            eps_list    = [Strain_sqrt(Space_modes[i],xg_modes[i], model.Space_modes[0].mesh.dim) for i in range(model.n_modes_truncated)]
            K = torch.tensor([[2*mu+lmbda, lmbda, lmbda, 0, 0, 0],[lmbda, 2*mu+lmbda, lmbda, 0, 0, 0], [lmbda, lmbda, 2*mu+lmbda, 0, 0, 0],[0, 0, 0, 2*mu, 0, 0],[0, 0, 0, 0, 2*mu, 0],[0, 0, 0, 0, 0, 2*mu]],dtype=model.float_config.dtype, device=model.float_config.device)
            # print(xg_k.shape) #DEBUG
            support = (1+torch.tanh(xg_k[:,:,2] - E[1][None,:,0]))*0.5


    eps_i = torch.stack(eps_list,dim=2)  
    # K = torch.tensor([[2*mu+lmbda, lmbda, 0],[lmbda, 2*mu+lmbda, 0],[0, 0, 2*mu]],dtype=torch.float64)
    Para_mode_Lists = [
        [model.Para_modes[mode][l](E[l][:,0].view(-1,1))[:,None] for l in range(model.n_para)]
        for mode in range(model.n_modes_truncated)
        ]
    lambda_i = [
            torch.cat([torch.unsqueeze(Para_mode_Lists[m][l],dim=0) for m in range(model.n_modes_truncated)], dim=0).to(torch.float64)
            for l in range(model.n_para)
        ]    


    E_1 = E[0][0,0].to(torch.float64)
    Delta_E_float = E[0][:,0].to(torch.float64)

    W_int = E_1*torch.einsum('ij,ejm...,eil...,em,mp...,lp...,mt...,lt...->',K,eps_i,eps_i,torch.abs(detJ_i),lambda_i[0],lambda_i[0],lambda_i[1],lambda_i[1]) +  \
            torch.einsum('ij,ejm...,eil...,em,mp...,lp...,mt...,lt...,p,et->',K,eps_i,eps_i,torch.abs(detJ_i),lambda_i[0],lambda_i[0],lambda_i[1],lambda_i[1],Delta_E_float,support)

    # Gravity_force = Gravity_vect(theta_float,rho = 1e-9)
    Gravity_force = Gravity_vect(torch.tensor(0*torch.pi).to(torch.float64),rho = 1e-9, dim = model.Space_modes[0].mesh.dim).to(model.float_config.dtype).to(model.float_config.device)

    W_ext = torch.einsum('iem,i,mp...,mt...,em->',u_i,Gravity_force,lambda_i[0],lambda_i[1],torch.abs(detJ_i))

    return (0.5*W_int - W_ext)/(E[0].shape[0])
    # return (0.5*W_int)/(E[0].shape[0])


InternalEnergy_2D_einsum_BiStiffness = InternalEnergy_2_3D_einsum_BiStiffness



def InternalEnergy_2_3D_einsum_BiStiffnessBiangle(model,lmbda, mu,E):

    '''Parameters order: 
            - E_0
            - E_1
            - alpha
            - theta
            - phi''' 


    E_1_float       = E[0][:,0] # p
    E_2_float       = E[1][:,0] # t
    alpha           = E[2][:,0] # s
    theta_float     = E[3][:,0] # q
    phi_float       = E[4][:,0] # r


    hyper_length = [(tensor.max() - tensor.min()).item() for tensor in E]


    Delta_E_float   = E_1_float[:,None] - E_2_float[None,:]       # pt : N_p is E_1 discretisation N_t is E_2, cross linked for smooth connection

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


    match model.Space_modes[0].mesh.dim:
        case 2:
            u_i = torch.stack(Space_modes,dim=2)
            xg_i = torch.stack(xg_modes,dim=2) 
            eps_list    = [Strain_sqrt(Space_modes[i],xg_modes[i]) for i in range(model.n_modes_truncated)]
            K = torch.tensor([[2*mu+lmbda, lmbda, 0],[lmbda, 2*mu+lmbda, 0],[0, 0, 2*mu]],dtype=model.float_config.dtype, device=model.float_config.device)
            support = (1+torch.tanh(xg_k[:,None,1] - E[1][None,:,0]))*0.5


        case 3:
            u_i = torch.stack(Space_modes,dim=2)
            xg_i = torch.stack(xg_modes,dim=2) 
            eps_list    = [Strain_sqrt(Space_modes[i],xg_modes[i], model.Space_modes[0].mesh.dim) for i in range(model.n_modes_truncated)]
            K = torch.tensor([[2*mu+lmbda, lmbda, lmbda, 0, 0, 0],[lmbda, 2*mu+lmbda, lmbda, 0, 0, 0], [lmbda, lmbda, 2*mu+lmbda, 0, 0, 0],[0, 0, 0, 2*mu, 0, 0],[0, 0, 0, 0, 2*mu, 0],[0, 0, 0, 0, 0, 2*mu]],dtype=model.float_config.dtype, device=model.float_config.device)
            # support = (1+torch.tanh(xg_k[:,:,2] - E[1][None,:,0]))*0.5
            support = (1+torch.tanh(xg_k[:,:,2] - alpha[None,:]))*0.5 #DEBUG Check shapes
            # print(f" E[2][None,:,0] size is { E[2][None,:,0].shape}")
            # print(f" alpha size is { alpha.shape}")
            # print(f" alpha[None,:] size is { alpha[None,:].shape}")

    eps_i = torch.stack(eps_list,dim=2)  

    Para_mode_Lists = [
        [model.Para_modes[mode][l](E[l][:,0].view(-1,1))[:,None] for l in range(model.n_para)]
        for mode in range(model.n_modes_truncated)
        ]
    lambda_i = [
            torch.cat([torch.unsqueeze(Para_mode_Lists[m][l],dim=0) for m in range(model.n_modes_truncated)], dim=0).to(torch.float64)
            for l in range(model.n_para)
        ]    


    # print(f"shape lambda_i[0] is {lambda_i[0].shape} ")#DEBUG
    # print(f"shape lambda_i[1] is {lambda_i[1].shape} ")#DEBUG
    # print(f"shape lambda_i[2] is {lambda_i[2].shape} ")#DEBUG
    # print(f"shape lambda_i[3] is {lambda_i[3].shape} ")#DEBUG
    # print(f"shape lambda_i[4] is {lambda_i[4].shape} ")#DEBUG
    # print(f"shape Delta_E_float is {Delta_E_float.shape} ")#DEBUG

    angles = [theta_float,phi_float]


    #DEBUG
    E_1 = E_2_float[0]
    # W_int = torch.einsum('ij,ejm...,eil...,em,mp...,lp...,mt...,lt...,ms...,ls...,mq...,lq...,mr...,lr...,t,es->',K,eps_i,eps_i,torch.abs(detJ_i),lambda_i[0],lambda_i[0],lambda_i[1],lambda_i[1],lambda_i[2],lambda_i[2],lambda_i[3],lambda_i[3],lambda_i[4],lambda_i[4],E_2_float,support) + \
    #         E_1*torch.einsum('ij,ejm...,eil...,em,mp...,lp...,mt...,lt...,ms...,ls...,mq...,lq...,mr...,lr...->',K,eps_i,eps_i,torch.abs(detJ_i),lambda_i[0],lambda_i[0],lambda_i[1],lambda_i[1],lambda_i[2],lambda_i[2],lambda_i[3],lambda_i[3],lambda_i[4],lambda_i[4])

    #DEBUBG OK
    # W_int = E_1*torch.einsum('ij,ejm...,eil...,em,mp...,lp...,mt...,lt...->',K,eps_i,eps_i,torch.abs(detJ_i),lambda_i[0],lambda_i[0],lambda_i[2],lambda_i[2]) +  \
    #     torch.einsum('ij,ejm...,eil...,em,mp...,lp...,mt...,lt...,p,et->',K,eps_i,eps_i,torch.abs(detJ_i),lambda_i[0],lambda_i[0],lambda_i[2],lambda_i[2],E_1_float,support)

    #DEBUBG 4para
    W_int = torch.einsum('ij,ejm...,eil...,em,mp...,lp...,ms...,ls...,mq...,lq...,mr...,lr...,p,es->',K,eps_i,eps_i,torch.abs(detJ_i),lambda_i[0],lambda_i[0],lambda_i[2],lambda_i[2],lambda_i[3],lambda_i[3],lambda_i[4],lambda_i[4],E_1_float,support) + \
        E_1*torch.einsum('ij,ejm...,eil...,em,mp...,lp...,ms...,ls...,mq...,lq...,mr...,lr...->',K,eps_i,eps_i,torch.abs(detJ_i),lambda_i[0],lambda_i[0],lambda_i[2],lambda_i[2],lambda_i[3],lambda_i[3],lambda_i[4],lambda_i[4])

    # W_int = torch.einsum('ij,ejm...,eil...,em,mp...,lp...,mt...,lt...,ms...,ls...,mq...,lq...,mr...,lr...,t,es->',K,eps_i,eps_i,torch.abs(detJ_i),lambda_i[0],lambda_i[0],lambda_i[1],lambda_i[1],lambda_i[2],lambda_i[2],lambda_i[3],lambda_i[3],lambda_i[4],lambda_i[4],E_2_float,support) + \
    #         torch.einsum('ij,ejm...,eil...,em,mp...,lp...,mt...,lt...,ms...,ls...,mq...,lq...,mr...,lr...,p->',   K,eps_i,eps_i,torch.abs(detJ_i),lambda_i[0],lambda_i[0],lambda_i[1],lambda_i[1],lambda_i[2],lambda_i[2],lambda_i[3],lambda_i[3],lambda_i[4],lambda_i[4],E_1_float)


    # Gravity_force = Gravity_vect(angles,rho = 1e-9, dim = model.Space_modes[0].mesh.dim, n_angle=2).to(model.float_config.dtype).to(model.float_config.device)
    # W_ext = torch.einsum('iem,iqr,mp...,mt...,ms...,mq...,mr...,em->',u_i,Gravity_force,lambda_i[0],lambda_i[1],lambda_i[2],lambda_i[3],lambda_i[4],torch.abs(detJ_i))

    #DEBUG 4para
    Gravity_force = Gravity_vect(angles,rho = 1e-9, dim = model.Space_modes[0].mesh.dim, n_angle=2).to(model.float_config.dtype).to(model.float_config.device)
    W_ext = torch.einsum('iem,iqr,mp...,ms...,mq...,mr...,em->',u_i,Gravity_force,lambda_i[0],lambda_i[2],lambda_i[3],lambda_i[4],torch.abs(detJ_i))


    return (0.5*W_int - W_ext)/(E[0].shape[0]*E[2].shape[0])




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

_grad_outputs_cache = {}

def _get_grad_outputs(dim, x_shape, dtype, device):
    if isinstance(x_shape, int):
        x_shape = (x_shape,)
    key = (dim, x_shape, dtype, device)
    if key not in _grad_outputs_cache:
        v = torch.eye(dim, dtype=dtype, device=device)
        for _ in range(len(x_shape) - 1):
            v = v.unsqueeze(-1)
        v = v.expand((dim, dim) + x_shape[:-1])
        _grad_outputs_cache[key] = v
    return _grad_outputs_cache[key]

def Strain_sqrt(u,x, dim = 2, grad_u = None):
    """ Return the Scientific voigt notation  of the strain [eps_xx eps_yy sqrt(2)eps_xy]"""
    if grad_u is not None:
        match dim:
            case 2:
                eps_xx = grad_u[:, 0, 0]
                eps_yy = grad_u[:, 1, 1]
                eps_xy = grad_u[:, 0, 1] + grad_u[:, 1, 0]
                return torch.stack([eps_xx, eps_yy, (1/torch.sqrt(torch.tensor(2)))*eps_xy], dim=1)
            case 3:
                eps_xx = grad_u[:, 0, 0]
                eps_yy = grad_u[:, 1, 1]
                eps_zz = grad_u[:, 2, 2]
                eps_yz = grad_u[:, 1, 2] + grad_u[:, 2, 1]
                eps_xz = grad_u[:, 0, 2] + grad_u[:, 2, 0]
                eps_xy = grad_u[:, 0, 1] + grad_u[:, 1, 0]
                return torch.stack([eps_xx, eps_yy, eps_zz,
                                    (1/torch.sqrt(torch.tensor(2)))*eps_yz,
                                    (1/torch.sqrt(torch.tensor(2)))*eps_xz,
                                    (1/torch.sqrt(torch.tensor(2)))*eps_xy], dim=1)

    # Compute gradients in one go
    if callable(u):
        grad_u_batch = torch.vmap(torch.func.jacrev(u))(x).permute(1, 0, 2)
    else:
        v = _get_grad_outputs(dim, x.shape, u.dtype, u.device)
        grad_u_batch = torch.autograd.grad(u, x, grad_outputs=v, create_graph=True, is_grads_batched=True)[0]
    
    match dim:
        case 2:
            du = grad_u_batch[0]
            dv = grad_u_batch[1]
            return torch.stack([du[...,0], dv[...,1], (1/torch.sqrt(torch.tensor(2)))*(du[...,1] + dv[...,0])],dim=1)

        case 3:
            du = grad_u_batch[0]
            dv = grad_u_batch[1]
            dw = grad_u_batch[2]
            return torch.stack([du[...,0], dv[...,1], dw[...,2],
                                (1/torch.sqrt(torch.tensor(2)))*(dv[...,2] + dw[...,1]), 
                                (1/torch.sqrt(torch.tensor(2)))*(du[...,2] + dw[...,0]), 
                                (1/torch.sqrt(torch.tensor(2)))*(du[...,1] + dv[...,0])],dim=1)

def grad_u_2_3D(u,x, dim = 2, grad_u = None):
    """ Return the gradient of u"""
    if grad_u is not None:
        return grad_u

    if callable(u):
        grad_u_batch = torch.vmap(torch.func.jacrev(u))(x).permute(1, 0, 2)
    else:
        v = _get_grad_outputs(dim, x.shape, u.dtype, u.device)
        grad_u_batch = torch.autograd.grad(u, x, grad_outputs=v, create_graph=True, is_grads_batched=True)[0]

    match dim:
        case 2:
            du = grad_u_batch[0]
            dv = grad_u_batch[1]
            return torch.stack([du, dv],dim=1)
        case 3:
            du = grad_u_batch[0]
            dv = grad_u_batch[1]
            dw = grad_u_batch[2]
            return torch.stack([du, dv, dw],dim=1)



grad_u_2D = grad_u_2_3D

def Green_lagrange(grad_u):
    grad_u_transpose = torch.einsum('exy->eyx',grad_u)
    quad_term = torch.einsum('ekx,eky->exy',grad_u,grad_u)
    return 0.5*(grad_u_transpose+grad_u+quad_term)

def Strain(u,x, grad_u=None):
    # Used only for visualisation purposes.
    # Ordering corresponds to Paraview.
    if grad_u is not None:
        dim = grad_u.shape[1]
        match dim:
            case 2:
                eps_xx = grad_u[:, 0, 0]
                eps_yy = grad_u[:, 1, 1]
                eps_xy = grad_u[:, 0, 1] + grad_u[:, 1, 0]
                return torch.stack([eps_xx, eps_yy, 0.5*eps_xy], dim=1)
            case 3:
                eps_xx = grad_u[:, 0, 0]
                eps_yy = grad_u[:, 1, 1]
                eps_zz = grad_u[:, 2, 2]
                eps_yz = grad_u[:, 1, 2] + grad_u[:, 2, 1]
                eps_xz = grad_u[:, 0, 2] + grad_u[:, 2, 0]
                eps_xy = grad_u[:, 0, 1] + grad_u[:, 1, 0]
                return torch.stack([eps_xx, eps_yy, eps_zz, 0.5*eps_xy, 0.5*eps_yz, 0.5*eps_xz], dim=1)

    if callable(u):
        grad_u_batch = torch.vmap(torch.func.jacrev(u))(x).permute(1, 0, 2)
        dim = grad_u_batch.shape[0]
    else:
        dim = u.shape[0]
        v = _get_grad_outputs(dim, x.shape, u.dtype, u.device)
        grad_u_batch = torch.autograd.grad(u, x, grad_outputs=v, create_graph=True, is_grads_batched=True)[0]

    match dim:
        case 2:
            """ Return the vector strain [eps_xx eps_yy eps_xy]"""
            du = grad_u_batch[0]
            dv = grad_u_batch[1]
            return torch.stack([du[...,0], dv[...,1], 0.5*(du[...,1] + dv[...,0])],dim=1)
        case 3:
            """ Return the vector strain [3D: XX, YY, ZZ, XY, YZ, XZ]"""
            du = grad_u_batch[0]
            dv = grad_u_batch[1]
            dw = grad_u_batch[2]
            return torch.stack([du[...,0], dv[...,1], dw[...,2],
                0.5*(du[...,1] + dv[...,0]), 0.5*(dv[...,2] + dw[...,1]), 0.5*(du[...,2] + dw[...,0])],dim=1)

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

def BoundaryStiffnessEnergy_para(model, k, E, u_ref=0):
    r"""
    Computes a parametric physical loss that adds a boundary stiffness 
    (like if the structure is held by surface springs).
    Evaluates \int_{\partial \Omega} k || u - u_ref ||^2 dS.
    """
    Space_modes_b = []
    detJ_modes_b = []
    for i in range(model.n_modes_truncated):
        res = model.Space_modes[i](return_boundary=True)
        Space_modes_b.append(res[3])
        detJ_modes_b.append(res[5])
        
    u_i_b = torch.stack(Space_modes_b, dim=2) 
    detJ_b_i = torch.stack(detJ_modes_b, dim=1) 
    
    Para_mode_Lists = [
        [model.Para_modes[mode][l](E[l][:,0].view(-1,1))[:,None] for l in range(model.n_para)]
        for mode in range(model.n_modes_truncated)
    ]
    lambda_i = [
        torch.cat([torch.unsqueeze(Para_mode_Lists[m][l],dim=0) for m in range(model.n_modes_truncated)], dim=0)
        for l in range(model.n_para)
    ]
    
    detJ_full = detJ_b_i[:, 0]
    is_u_ref_nonzero = isinstance(u_ref, torch.Tensor) or u_ref != 0
    
    if model.n_para == 1:
        L0 = lambda_i[0][:, :, 0]
        W_uu = k * torch.einsum('xem,xel,e,mp,lp->', u_i_b, u_i_b, torch.abs(detJ_full), L0, L0)
        
        if is_u_ref_nonzero:
            u_ref_t = u_ref * torch.ones(u_i_b.shape[0], device=u_i_b.device)
            W_u_ref = k * torch.einsum('xem,e,mp,x->', u_i_b, torch.abs(detJ_full), L0, u_ref_t)
            W_ref_ref = k * torch.sum(u_ref_t**2) * torch.sum(torch.abs(detJ_full)) * L0.shape[1]
            W = W_uu - 2 * W_u_ref + W_ref_ref
        else:
            W = W_uu
            
        return W / E[0].shape[0]
        
    elif model.n_para == 2:
        L0 = lambda_i[0][:, :, 0]
        L1 = lambda_i[1][:, :, 0]
        W_uu = k * torch.einsum('xem,xel,e,mp,lp,mt,lt->', u_i_b, u_i_b, torch.abs(detJ_full), L0, L0, L1, L1)
        
        if is_u_ref_nonzero:
            u_ref_t = u_ref * torch.ones(u_i_b.shape[0], device=u_i_b.device)
            W_u_ref = k * torch.einsum('xem,e,mp,mt,x->', u_i_b, torch.abs(detJ_full), L0, L1, u_ref_t)
            W_ref_ref = k * torch.sum(u_ref_t**2) * torch.sum(torch.abs(detJ_full)) * L0.shape[1] * L1.shape[1]
            W = W_uu - 2 * W_u_ref + W_ref_ref
        else:
            W = W_uu
            
        return W / (E[0].shape[0] * E[1].shape[0])
        
    elif model.n_para == 3:
        L0 = lambda_i[0][:, :, 0]
        L1 = lambda_i[1][:, :, 0]
        L2 = lambda_i[2][:, :, 0]
        W_uu = k * torch.einsum('xem,xel,e,mp...,lp...,mt...,lt...,ms...,ls...->', u_i_b, u_i_b, torch.abs(detJ_full), L0, L0, L1, L1, L2, L2)
        
        if is_u_ref_nonzero:
            u_ref_t = u_ref * torch.ones(u_i_b.shape[0], device=u_i_b.device)
            W_u_ref = k * torch.einsum('xem,e,mp,mt,ms,x->', u_i_b, torch.abs(detJ_full), L0, L1, L2, u_ref_t)
            W_ref_ref = k * torch.sum(u_ref_t**2) * torch.sum(torch.abs(detJ_full)) * L0.shape[1] * L1.shape[1] * L2.shape[1]
            W = W_uu - 2 * W_u_ref + W_ref_ref
        else:
            W = W_uu

    elif model.n_para == 4:
        L0 = lambda_i[0][:, :, 0]
        L1 = lambda_i[1][:, :, 0]
        L2 = lambda_i[2][:, :, 0]
        L3 = lambda_i[3][:, :, 0]
        W_uu = k * torch.einsum('xem,xel,e,mp...,lp...,mt...,lt...,ms...,ls...,mq...,lq...->', u_i_b, u_i_b, torch.abs(detJ_full), L0, L0, L1, L1, L2, L2, L3, L3)
        
        if is_u_ref_nonzero:
            u_ref_t = u_ref * torch.ones(u_i_b.shape[0], device=u_i_b.device)
            W_u_ref = k * torch.einsum('xem,e,mp,mt,ms,mq,x->', u_i_b, torch.abs(detJ_full), L0, L1, L2,L3, u_ref_t)
            W_ref_ref = k * torch.sum(u_ref_t**2) * torch.sum(torch.abs(detJ_full)) * L0.shape[1] * L1.shape[1] * L2.shape[1]
            W = W_uu - 2 * W_u_ref + W_ref_ref
        else:
            W = W_uu
            
        return W / (E[0].shape[0] * E[1].shape[0] * E[2].shape[0])

def BoundaryStiffnessEnergy_para_normal(model, k, E, u_ref=0, compute_normals_once=True):
    r"""
    Computes a parametric physical loss that adds a boundary stiffness 
    only in the normal direction (k * (u \cdot n - u_ref_n)^2).
    Evaluates \int_{\partial \Omega} k (u \cdot n - u_ref_n)^2 dS.
    """
    Space_modes_b = []
    detJ_modes_b = []
    for i in range(model.n_modes_truncated):
        res = model.Space_modes[i](return_boundary=True)
        Space_modes_b.append(res[3])
        detJ_modes_b.append(res[5])
        
    u_i_b = torch.stack(Space_modes_b, dim=2) 
    detJ_b_i = torch.stack(detJ_modes_b, dim=1) 
    
    Para_mode_Lists = [
        [model.Para_modes[mode][l](E[l][:,0].view(-1,1))[:,None] for l in range(model.n_para)]
        for mode in range(model.n_modes_truncated)
    ]
    lambda_i = [
        torch.cat([torch.unsqueeze(Para_mode_Lists[m][l],dim=0) for m in range(model.n_modes_truncated)], dim=0)
        for l in range(model.n_para)
    ]
    
    detJ_full = detJ_b_i[:, 0]
    
    # Try to use the user's normals or cached normals, otherwise compute them
    try:
        if compute_normals_once and hasattr(model.Space_modes[0].mesh, 'boundary_normals'):
            n = model.Space_modes[0].mesh.boundary_normals.to(u_i_b.device)
        else:
            # raise ValueError() #DEBUG Force computing normals
            n = model.Space_modes[0].mesh.normals.to(u_i_b.device)
            if n.shape[0] != u_i_b.shape[1]:
                print(f"Warning: mesh.normals shape {n.shape} does not match number of boundary elements {u_i_b.shape[1]}. Computing geometrically...")
                raise ValueError()
            if compute_normals_once:
                model.Space_modes[0].mesh.boundary_normals = n.detach()
    except:
        # Fallback to computing normals geometrically from the borders_nodes
        coords = model.Space_modes[0].coordinates_all.to(u_i_b.device)
        border_nodes = torch.tensor(model.Space_modes[0].borders_nodes, dtype=torch.long, device=u_i_b.device) - 1
        v0 = coords[border_nodes[:, 0]]
        v1 = coords[border_nodes[:, 1]]
        v2 = coords[border_nodes[:, 2]]
        n = torch.cross(v1 - v0, v2 - v0, dim=1)
        n = n / torch.norm(n, dim=1, keepdim=True)
        if compute_normals_once:
            model.Space_modes[0].mesh.boundary_normals = n.detach()
        
    # n has shape [num_boundary_elems, 3]. We transpose to [3, num_boundary_elems] for einsum
    n = n.t()
    
    is_u_ref_nonzero = isinstance(u_ref, torch.Tensor) or u_ref != 0
    if is_u_ref_nonzero:
        if isinstance(u_ref, torch.Tensor) and u_ref.ndim > 0:
            u_ref_n = torch.einsum('x,xe->e', u_ref, n)
        else:
            u_ref_n = u_ref * torch.ones(u_i_b.shape[1], device=u_i_b.device)
    else:
        u_ref_n = 0
        
    if isinstance(k, torch.Tensor) and k.ndim == 1:
        k = k.view(-1, *[1]*(model.n_para + 1))
        
    if model.n_para == 1:
        L0 = lambda_i[0][:, :, 0]
        # un_i_b = sum_x u_i_b[x,e,m] * n[x,e]  -> shape [e, m]
        un_i_b = torch.einsum('xem,xe->em', u_i_b, n)
        
        W_uu = k * torch.einsum('em,el,e,mp,lp->', un_i_b, un_i_b, torch.abs(detJ_full), L0, L0)
        
        if is_u_ref_nonzero:
            W_u_ref = k * torch.einsum('em,e,mp,e->', un_i_b, torch.abs(detJ_full), L0, u_ref_n)
            W_ref_ref = k * torch.sum(u_ref_n**2 * torch.abs(detJ_full)) * L0.shape[1]
            W = W_uu - 2 * W_u_ref + W_ref_ref
        else:
            W = W_uu
            
        return W / E[0].shape[0]
        
    elif model.n_para == 2:
        L0 = lambda_i[0][:, :, 0]
        L1 = lambda_i[1][:, :, 0]
        un_i_b = torch.einsum('xem,xe->em', u_i_b, n)
        
        W_uu = k * torch.einsum('em,el,e,mp,lp,mt,lt->', un_i_b, un_i_b, torch.abs(detJ_full), L0, L0, L1, L1)
        
        if is_u_ref_nonzero:
            W_u_ref = k * torch.einsum('em,e,mp,mt,e->', un_i_b, torch.abs(detJ_full), L0, L1, u_ref_n)
            W_ref_ref = k * torch.sum(u_ref_n**2 * torch.abs(detJ_full)) * L0.shape[1] * L1.shape[1]
            W = W_uu - 2 * W_u_ref + W_ref_ref
        else:
            W = W_uu
            
        return W / (E[0].shape[0] * E[1].shape[0])
        
    elif model.n_para == 3:
        L0 = lambda_i[0][:, :, 0]
        L1 = lambda_i[1][:, :, 0]
        L2 = lambda_i[2][:, :, 0]
        un_i_b = torch.einsum('xem,xe->em', u_i_b, n)
        
        W_uu = k * torch.einsum('em,el,e,mp...,lp...,mt...,lt...,ms...,ls...->', un_i_b, un_i_b, torch.abs(detJ_full), L0, L0, L1, L1, L2, L2)
        
        if is_u_ref_nonzero:
            W_u_ref = k * torch.einsum('em,e,mp,mt,ms,e->', un_i_b, torch.abs(detJ_full), L0, L1, L2, u_ref_n)
            W_ref_ref = k * torch.sum(u_ref_n**2 * torch.abs(detJ_full)) * L0.shape[1] * L1.shape[1] * L2.shape[1]
            W = W_uu - 2 * W_u_ref + W_ref_ref
        else:
            W = W_uu

    elif model.n_para == 4:
        L0 = lambda_i[0][:, :, 0]
        L1 = lambda_i[1][:, :, 0]
        L2 = lambda_i[2][:, :, 0]
        L3 = lambda_i[3][:, :, 0]
        un_i_b = torch.einsum('xem,xe->em', u_i_b, n)
        
        W_uu = k * torch.einsum('em,el,e,mp...,lp...,mt...,lt...,ms...,ls...,mq...,lq...->', un_i_b, un_i_b, torch.abs(detJ_full), L0, L0, L1, L1, L2, L2, L3, L3)
        
        if is_u_ref_nonzero:
            W_u_ref = k * torch.einsum('em,e,mp,mt,ms,mq,e->', un_i_b, torch.abs(detJ_full), L0, L1, L2,L3, u_ref_n)
            W_ref_ref = k * torch.sum(u_ref_n**2 * torch.abs(detJ_full)) * L0.shape[1] * L1.shape[1] * L2.shape[1]
            W = W_uu - 2 * W_u_ref + W_ref_ref
        else:
            W = W_uu
            
        # return W / (E[0].shape[0] * E[1].shape[0] * E[2].shape[0]) #DEBUG
        # return W / (E[0].shape[0]* E[1].shape[0])
        return W / (E[0].shape[0])

def BoundaryPressureEnergy_para_learned(model, E, p0_val=0.0, p0_idx=None, h_val=0.0, h_idx=None, k_val=0.0, k_idx=None, z0=0.0, y0=0.0, compute_normals_once=True):
    r"""
    Computes a parametric physical loss for external pressure where the pressure 
    parameters P_0, H, and K can either be constants or learned parametric dimensions 
    from the PGD hypercube.
    
    P(y, z) = P_0 + H*(z - z_0) + K*(y - y_0)
    """
    Space_modes_b = []
    detJ_modes_b = []
    x_g_modes_b = []
    for i in range(model.n_modes_truncated):
        res = model.Space_modes[i](return_boundary=True)
        Space_modes_b.append(res[3])
        x_g_modes_b.append(res[4])
        detJ_modes_b.append(res[5])
        
    u_i_b = torch.stack(Space_modes_b, dim=2) 
    detJ_b_i = torch.stack(detJ_modes_b, dim=1) 
    x_g_b_i = torch.stack(x_g_modes_b, dim=1)
    
    Para_mode_Lists = [
        [model.Para_modes[mode][l](E[l][:,0].view(-1,1))[:,None] for l in range(model.n_para)]
        for mode in range(model.n_modes_truncated)
    ]
    lambda_i = [
        torch.cat([torch.unsqueeze(Para_mode_Lists[m][l],dim=0) for m in range(model.n_modes_truncated)], dim=0)
        for l in range(model.n_para)
    ]
    
    detJ_full = detJ_b_i[:, 0]
    print("********************************** Pressure ")
    print(f"compute_normals_once {compute_normals_once}")
    print(f"hasattr(model.Space_modes[0].mesh, 'boundary_normals'){hasattr(model.Space_modes[0].mesh, 'boundary_normals')}")

    try:
        if compute_normals_once and hasattr(model.Space_modes[0].mesh, 'boundary_normals'):
            n = model.Space_modes[0].mesh.boundary_normals.to(u_i_b.device)
        else:
            # raise ValueError() #DEBUG Force computing normals
            n = model.Space_modes[0].mesh.normals.to(u_i_b.device)
            if n.shape[0] != u_i_b.shape[1]:
                print(f"Warning: mesh.normals shape {n.shape} does not match number of boundary elements {u_i_b.shape[1]}. Computing geometrically...")
                raise ValueError()
            if compute_normals_once:
                model.Space_modes[0].mesh.boundary_normals = n.detach()
    except:
        print("********************************** Normals are being computed !!! ")
        coords = model.Space_modes[0].coordinates_all.to(u_i_b.device)
        border_nodes = torch.tensor(model.Space_modes[0].borders_nodes, dtype=torch.long, device=u_i_b.device) - 1
        v0 = coords[border_nodes[:, 0]]
        v1 = coords[border_nodes[:, 1]]
        v2 = coords[border_nodes[:, 2]]
        n = torch.cross(v1 - v0, v2 - v0, dim=1)
        n = n / torch.norm(n, dim=1, keepdim=True)
        if compute_normals_once:
            model.Space_modes[0].mesh.boundary_normals = n.detach()
            
    n = n.t()
    
    x_g = x_g_b_i[:, 0, :]
    y = x_g[:, 1]
    z = x_g[:, 2]
    
    if p0_idx is not None: P0 = E[p0_idx][:, 0]
    else: P0 = p0_val
        
    if h_idx is not None: H = E[h_idx][:, 0]
    else: H = h_val
        
    if k_idx is not None: K = E[k_idx][:, 0]
    else: K = k_val
    
    un_i_b = torch.einsum('xem,xe->em', u_i_b, n)
    
    def compute_work_component(spatial_field, V, V_idx):
        I_m = torch.einsum('em,e,e->m', un_i_b, spatial_field, torch.abs(detJ_full))
        for i in range(model.n_para):
            L_i = lambda_i[i][:, :, 0]
            if i == V_idx:
                S_m = torch.einsum('p,mp...->m', V, L_i)
            else:
                S_m = torch.sum(L_i, dim=1).view(-1)
            I_m = I_m * S_m
        if V_idx is None:
            I_m = I_m * V
        return torch.sum(I_m)
        
    W_P0 = compute_work_component(torch.ones_like(z), P0, p0_idx)
    W_H = compute_work_component(z - z0, H, h_idx)
    W_K = compute_work_component(y - y0, K, k_idx)
    

    # W_P0_debug = P0*torch.einsum('em,mt...,ms...,mp...,e->', un_i_b, lambda_i[0], lambda_i[1], lambda_i[2],torch.abs(detJ_full))

    W_ext = W_P0 + W_H + W_K
    
    denom = 1
    # denom = E[0].shape[0]
    for i in range(model.n_para):
        denom = denom * E[i].shape[0]
        
    return W_ext / denom

def BoundaryPressureEnergy_para(model, P0, H, z0, K, y0, E, compute_normals_once=True):
    r"""
    Computes a parametric physical loss that adds the external work 
    done by a pressure field P(y, z) = P_0 + H*(z - z_0) + K*(y - y_0)
    Evaluates \int_{\partial \Omega} P (u \cdot n) dS.
    """
    Space_modes_b = []
    detJ_modes_b = []
    x_g_modes_b = []
    for i in range(model.n_modes_truncated):
        res = model.Space_modes[i](return_boundary=True)
        Space_modes_b.append(res[3])
        x_g_modes_b.append(res[4])
        detJ_modes_b.append(res[5])
        
    u_i_b = torch.stack(Space_modes_b, dim=2) 
    detJ_b_i = torch.stack(detJ_modes_b, dim=1) 
    x_g_b_i = torch.stack(x_g_modes_b, dim=1)
    
    Para_mode_Lists = [
        [model.Para_modes[mode][l](E[l][:,0].view(-1,1))[:,None] for l in range(model.n_para)]
        for mode in range(model.n_modes_truncated)
    ]
    lambda_i = [
        torch.cat([torch.unsqueeze(Para_mode_Lists[m][l],dim=0) for m in range(model.n_modes_truncated)], dim=0)
        for l in range(model.n_para)
    ]
    
    detJ_full = detJ_b_i[:, 0]
    
    try:
        if compute_normals_once and hasattr(model.Space_modes[0].mesh, 'boundary_normals'):
            n = model.Space_modes[0].mesh.boundary_normals.to(u_i_b.device)
        else:
            # raise ValueError() #DEBUG Force computing normals
            n = model.Space_modes[0].mesh.normals.to(u_i_b.device)
            if n.shape[0] != u_i_b.shape[1]:
                print(f"Warning: mesh.normals shape {n.shape} does not match number of boundary elements {u_i_b.shape[1]}. Computing geometrically...")
                raise ValueError()
            if compute_normals_once:
                model.Space_modes[0].mesh.boundary_normals = n.detach()
    except:
        coords = model.Space_modes[0].coordinates_all.to(u_i_b.device)
        border_nodes = torch.tensor(model.Space_modes[0].borders_nodes, dtype=torch.long, device=u_i_b.device) - 1
        v0 = coords[border_nodes[:, 0]]
        v1 = coords[border_nodes[:, 1]]
        v2 = coords[border_nodes[:, 2]]
        n = torch.cross(v1 - v0, v2 - v0, dim=1)
        n = n / torch.norm(n, dim=1, keepdim=True)
        if compute_normals_once:
            model.Space_modes[0].mesh.boundary_normals = n.detach()
            
    n = n.t()
    
    x_g = x_g_b_i[:, 0, 0, :]
    y = x_g[:, 1]
    z = x_g[:, 2]
    P = P0 + H*(z - z0) + K*(y - y0)
    
    un_i_b = torch.einsum('xem,xe->em', u_i_b, n)
    
    if model.n_para == 1:
        L0 = lambda_i[0][:, :, 0]
        W_ext = torch.einsum('em,e,e,mp...->', un_i_b, P, torch.abs(detJ_full), L0)
        return W_ext / E[0].shape[0]
        
    elif model.n_para == 2:
        L0 = lambda_i[0][:, :, 0]
        L1 = lambda_i[1][:, :, 0]
        W_ext = torch.einsum('em,e,e,mp...,mt...->', un_i_b, P, torch.abs(detJ_full), L0, L1)
        return W_ext / (E[0].shape[0] * E[1].shape[0])
        
    elif model.n_para == 3:
        L0 = lambda_i[0][:, :, 0]
        L1 = lambda_i[1][:, :, 0]
        L2 = lambda_i[2][:, :, 0]
        W_ext = torch.einsum('em,e,e,mp...,mt...,ms...->', un_i_b, P, torch.abs(detJ_full), L0, L1, L2)
        return W_ext / (E[0].shape[0] * E[1].shape[0] * E[2].shape[0])