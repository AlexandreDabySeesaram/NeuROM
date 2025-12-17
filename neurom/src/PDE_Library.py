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
                component_1 = sin_theta * torch.ones(component_2.shape)  
                component_3 = cos_theta * sin_phi           

                return (rho*g*torch.stack([component_1, component_2, component_3], dim=0))


def VolumeForcesEnergy_2D(u,theta, rho, mapping = None):
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

def InternalEnergy_2_3D_einsum(model, u,x,lmbda, mu, config, dim = 2, mapping = None):

    if 'parameters' in config:
        if 'x_0_x' in config['parameters']:
            x0 = torch.tensor((config["parameters"]["x_0_x"],config["parameters"]["x_0_y"]))
            eps_macro_2 = torch.tensor(((config["parameters"]["eps_xx"],config["parameters"]["eps_xy"],config["parameters"]["eps_xy"],config["parameters"]["eps_yy"])))
            eps_macro_4 = torch.tensor(((config["parameters"]["eps_xx"],config["parameters"]["eps_xy"]),(config["parameters"]["eps_xy"],config["parameters"]["eps_yy"])))

    match dim:
        case 2:
            eps =  Strain_sqrt(u,x)
            eps_full = Strain_full(u,x)

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            if not (mapping is None):

                if 'parameters' in config:
                    if 'x_0_x' in config['parameters']:
                        eps_full = eps_full - eps_macro_2


                list_F = mapping[1]
                F = torch.stack(list_F)  # shape: [N, 2, 2]
                F_inv = torch.linalg.inv(F)  # shape: [N, 2, 2]

                sqrt2 = eps.new_tensor(2.0).sqrt()

                grad_u = torch.stack([
                    torch.stack([eps_full[:,0], eps_full[:,1]], dim=1),  # first row
                    torch.stack([eps_full[:,2], eps_full[:,3]], dim=1),  # second row
                ], dim=1)
                eps_R = (grad_u @ F_inv  + (grad_u @ F_inv).transpose(1, 2))/2

                if 'parameters' in config:
                    if 'x_0_x' in config['parameters']:
                        eps_R = eps_R + eps_macro_4


                eps_R_voigt = torch.stack([
                    eps_R[:, 0, 0],                           # ε_xx
                    eps_R[:, 1, 1],                           # ε_yy
                    (eps_R[:, 0, 1]) * sqrt2             # ε_xy 
                ], dim=1)

                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
                eps = eps_R_voigt
            
            K = torch.tensor([[2*mu+lmbda, lmbda, 0],[lmbda, 2*mu+lmbda, 0],[0, 0, 2*mu]],dtype=eps.dtype, device=eps.device)
            W_e = torch.einsum('ij,ej,ei->e',K,eps,eps)

            return W_e
        case 3:
            eps =  Strain_sqrt(u,x, dim = dim)
            K = torch.tensor([[2*mu+lmbda, lmbda, lmbda, 0, 0, 0],[lmbda, 2*mu+lmbda, lmbda, 0, 0, 0], [lmbda, lmbda, 2*mu+lmbda, 0, 0, 0],[0, 0, 0, 2*mu, 0, 0],[0, 0, 0, 0, 2*mu, 0],[0, 0, 0, 0, 0, 2*mu]],dtype=eps.dtype, device=eps.device)
            W_e = torch.einsum('ij,ej...,ei...->e',K,eps,eps)
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

def InternalEnergy_2D_einsum_para(model,lmbda, mu, E):
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



def InternalEnergy_2D_einsum_hexa_para(model,lmbda, mu, h, config, list_F, list_J):
    
    if 'parameters' in config:
        if 'x_0_x' in config['parameters']:
            x0 = torch.tensor((config["parameters"]["x_0_x"],config["parameters"]["x_0_y"]))
            eps_macro = torch.tensor(((config["parameters"]["eps_xx"],config["parameters"]["eps_xy"]),(config["parameters"]["eps_xy"],config["parameters"]["eps_yy"])))
            eps_macro_vect = torch.tensor(((config["parameters"]["eps_xx"],config["parameters"]["eps_xy"],config["parameters"]["eps_xy"],config["parameters"]["eps_yy"])))

    Space_modes = []
    xg_modes = []
    detJ_modes = []
    
    Para_mode_Lists = [
        [model.Para_modes[mode][l](h[l][:,0].view(-1,1))[:,None] for l in range(model.n_para)]
        for mode in range(model.n_modes_truncated)
        ]
    
    lambda_i = [
            torch.cat([torch.unsqueeze(Para_mode_Lists[m][l],dim=0) for m in range(model.n_modes_truncated)], dim=0)
            for l in range(model.n_para)        ][0]    
    # lambda_i shape (N_modes, N_h, 1, 1)

    K = torch.tensor([[2*mu+lmbda, lmbda, 0],[lmbda, 2*mu+lmbda, 0],[0, 0, 2*mu]],dtype=model.float_config.dtype, device=model.float_config.device)
   

    for i in range(model.n_modes_truncated):
        u_k,xg_k,detJ_k = model.Space_modes[i]()
        Space_modes.append(u_k)
        xg_modes.append(xg_k)
        detJ_modes.append(detJ_k)

    J_x = torch.abs(detJ_modes[0]) 

    eps_full_list = [Strain_full(Space_modes[i],xg_modes[i]) for i in range(model.n_modes_truncated)]


    eps_list = [Strain_sqrt(Space_modes[i],xg_modes[i]) for i in range(model.n_modes_truncated)]
    eps_i = torch.stack(eps_list,dim=2)  

    eps_full_i = torch.stack(eps_full_list,dim=2)  # ---> [N_x, 4, N_modes]

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    F = torch.stack([torch.stack(F_in) for F_in in list_F])         # list ---> tensor shape: [N_xg, N_h, 2, 2])
    F_inv = torch.linalg.inv(F)                                     # shape: [N_xg, N_h, 2, 2])

    sqrt2 = eps_i.new_tensor(2.0).sqrt()

    grad_u = torch.stack([                                                                      # shape [N_x, 2, 2, 2]
            torch.stack([eps_full_i[:,0,:], eps_full_i[:,1,:]], dim=1),  # first row
            torch.stack([eps_full_i[:,2,:], eps_full_i[:,3,:]], dim=1),  # second row
    ], dim=1)

    # grad u: [N_x, 2, 2, N_m] !!!
    grad_u_2 = grad_u.permute(0, 3, 1, 2)   # -----> [N_x, N_modes, 2, 2]

    # F: [N_x, N_h, 2, 2]
    F_exp = F_inv.unsqueeze(2)       # -> [N_x, N_h, 1, 2, 2]
    u_exp = grad_u_2.unsqueeze(1)      # -> [N_x, 1, N_m, 2, 2]

    eps_R = (torch.matmul(F_exp, u_exp) + torch.matmul(F_exp, u_exp).transpose(-1,-2))/2
    
    eps_R_voigt = torch.stack([
        eps_R[:, :, :, 0, 0],                           # ε_xx
        eps_R[:, :, :, 1, 1],                           # ε_yy
        (eps_R[:,:, :, 0, 1]) * sqrt2             # ε_xy 
    ], dim=-2)

    eps_i = eps_R_voigt    


    # eps_i: [Nx, Nh, 3, Nm] = [3660, 5, 3, 4]
    # lambda_i: [Nm, Nh, 1, 1]
    # J_x: [Nx]
    # K: [3,3]
    # list_J: list of Nx lists of length Nh  → Jm(x,h)


    # 0) list of lists to tensor
    Jm = torch.abs(torch.tensor(list_J, dtype=eps_i.dtype, device=eps_i.device))   # --->  [Nx, Nh]

    # ----------------------------------------------
    # 1) prep
    lambda_i = lambda_i.squeeze(-1).squeeze(-1)      # [Nm, Nh]
    lambda_exp = lambda_i.T.unsqueeze(0).unsqueeze(2)

    # 2) Microstrain: eps_u(x,h) = Σ_m λ_m(h) eps_m(x,h)
    # ----------------------------------------------
    eps_u = (eps_i * lambda_exp).sum(dim=-1)         # [Nx, Nh, 3], 3 components of Voight notation strain

    # 3) Quadratic form: eps_u : K : eps_u
    # --------------------------------------------------------------
    prod = torch.einsum("xhi,ij,xhj->xh", eps_u, K, eps_u)   # [Nx, Nh]

    # 4) Final integral:
    # W = 1/2 Σ_x Σ_h prod(x,h) * J(x) * Jm(x,h)
    # ----------------------------------------------
    W_in = 0.5 * (prod * J_x[:, None] *Jm).sum()

    return W_in


# def InternalEnergy_2D_einsum_hexa_para_v2(model,lmbda, mu, h, config, list_F, list_J):
    
#     if 'parameters' in config:
#         if 'x_0_x' in config['parameters']:
#             x0 = torch.tensor((config["parameters"]["x_0_x"],config["parameters"]["x_0_y"]))
#             eps_macro = torch.tensor(((config["parameters"]["eps_xx"],config["parameters"]["eps_xy"]),(config["parameters"]["eps_xy"],config["parameters"]["eps_yy"])))
#             eps_macro_vect = torch.tensor(((config["parameters"]["eps_xx"],config["parameters"]["eps_xy"],config["parameters"]["eps_xy"],config["parameters"]["eps_yy"])))

#     Space_modes = []
#     xg_modes = []
#     detJ_modes = []
    
#     Para_mode_Lists = [
#         [model.Para_modes[mode][l](h[l][:,0].view(-1,1))[:,None] for l in range(model.n_para)]
#         for mode in range(model.n_modes_truncated)
#         ]
    
#     lambda_i = [
#             torch.cat([torch.unsqueeze(Para_mode_Lists[m][l],dim=0) for m in range(model.n_modes_truncated)], dim=0)
#             for l in range(model.n_para)        ][0]    
#     # lambda_i shape (N_modes, N_h, 1, 1)

#     K = torch.tensor([[2*mu+lmbda, lmbda, 0],[lmbda, 2*mu+lmbda, 0],[0, 0, 2*mu]],dtype=model.float_config.dtype, device=model.float_config.device)
   

#     for i in range(model.n_modes_truncated):
#         u_k,xg_k,detJ_k = model.Space_modes[i]()
#         Space_modes.append(u_k)
#         xg_modes.append(xg_k)
#         detJ_modes.append(detJ_k)

#     J_x = torch.abs(detJ_modes[0]) 

    
#     # if 'parameters' in config:
#     #     if 'x_0_x' in config['parameters']:
#     #         # Affine part of u
#     #         # print("     eps_macro = ", eps_macro)

#     #         u_affine = []
#     #         for i in range(model.n_modes_truncated):
#     #             # x_aff = torch.unsqueeze(xg_modes[i] - x0,-1)
#     #             # u_aff = torch.matmul(eps_macro, x_aff )
#     #             # u_aff = torch.squeeze(u_aff,-1)

#     #             u_aff = (eps_macro @ (xg_modes[i] - x0).T)
#     #             u_affine.append(u_aff)
#     #             # print("     mode = ",i, ", u = ", max(Space_modes[i][0,:]).item(), max(Space_modes[i][1,:]).item() )


#     #         u_affine = torch.stack(u_affine, dim=0)
#     #         # eps_full_list = [Strain_full(Space_modes[i] + u_affine[i,:,:], xg_modes[i]) for i in range(model.n_modes_truncated)]
#     #         eps_full_list = [Strain_full(Space_modes[i], xg_modes[i]) for i in range(model.n_modes_truncated)]
#     #     else:

#     eps_full_list = [Strain_full(Space_modes[i],xg_modes[i]) for i in range(model.n_modes_truncated)]


#     eps_list = [Strain_sqrt(Space_modes[i],xg_modes[i]) for i in range(model.n_modes_truncated)]
#     eps_i = torch.stack(eps_list,dim=2)  

#     eps_full_i = torch.stack(eps_full_list,dim=2)  # ---> [N_x, 4, N_modes]

#     # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

#     F = torch.stack([torch.stack(F_in) for F_in in list_F])         # list ---> tensor shape: [N_xg, N_h, 2, 2])
#     F_inv = torch.linalg.inv(F)                                     # shape: [N_xg, N_h, 2, 2])

#     sqrt2 = eps_i.new_tensor(2.0).sqrt()

#     grad_u = torch.stack([                                                                      # shape [N_x, 2, 2, 2]
#             torch.stack([eps_full_i[:,0,:], eps_full_i[:,1,:]], dim=1),  # first row
#             torch.stack([eps_full_i[:,2,:], eps_full_i[:,3,:]], dim=1),  # second row
#     ], dim=1)

#     # grad u: [N_x, 2, 2, N_m] !!!
#     grad_u_2 = grad_u.permute(0, 3, 1, 2)   # -----> [N_x, N_modes, 2, 2]

#     # F: [N_x, N_h, 2, 2]
#     F_exp = F_inv.unsqueeze(2)       # -> [N_x, N_h, 1, 2, 2]
#     u_exp = grad_u_2.unsqueeze(1)      # -> [N_x, 1, N_m, 2, 2]

#     eps_R = (torch.matmul(F_exp, u_exp) + torch.matmul(F_exp, u_exp).transpose(-1,-2))/2
    
#     eps_R_voigt = torch.stack([
#         eps_R[:, :, :, 0, 0],                           # ε_xx
#         eps_R[:, :, :, 1, 1],                           # ε_yy
#         (eps_R[:,:, :, 0, 1]) * sqrt2             # ε_xy 
#     ], dim=-2)

#     eps_i = eps_R_voigt    


#     # eps_i: [Nx, Nh, 3, Nm] = [3660, 5, 3, 4]
#     # lambda_i: [Nm, Nh, 1, 1]
#     # J_x: [Nx]
#     # K: [3,3]
#     # list_J: list of Nx lists of length Nh  → Jm(x,h)


#     # 0) list of lists to tensor
#     Jm = torch.abs(torch.tensor(list_J, dtype=eps_i.dtype, device=eps_i.device))   # --->  [Nx, Nh]

#     # ----------------------------------------------
#     # 1) prep
#     lambda_i = lambda_i.squeeze(-1).squeeze(-1)      # [Nm, Nh]
#     lambda_exp = lambda_i.T.unsqueeze(0).unsqueeze(2)

#     # 2) Microstrain: eps_u(x,h) = Σ_m λ_m(h) eps_m(x,h)
#     # ----------------------------------------------
#     eps_u = (eps_i * lambda_exp).sum(dim=-1)         # [Nx, Nh, 3], 3 components of Voight notation strain


#     if 'parameters' in config:
#         if 'x_0_x' in config['parameters']:
#             sqrt2 = eps_i.new_tensor(2.0).sqrt()

#             eps_macro_voigt = torch.stack([
#                 eps_macro[0,0],              # ε_xx
#                 eps_macro[1,1],              # ε_yy
#                 eps_macro[0,1] * sqrt2       # ε_xy
#             ], dim=0)                        # [3]

#             # reshape for broadcasting
#             eps_macro_voigt = eps_macro_voigt.view(1,1,3)       # [1,1,3]

#             eps_u = eps_u + eps_macro_voigt

#     # 3) Quadratic form: eps_u : K : eps_u
#     # --------------------------------------------------------------
#     prod = torch.einsum("xhi,ij,xhj->xh", eps_u, K, eps_u)   # [Nx, Nh]

#     # 4) Final integral:
#     # W = 1/2 Σ_x Σ_h prod(x,h) * J(x) * Jm(x,h)
#     # ----------------------------------------------
#     W_in = 0.5 * (prod * J_x[:, None] *Jm).sum()

#     return W_in

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

def Strain_sqrt(u,x, dim = 2):
    """ Return the Scientific voigt notation  of the strain [eps_xx eps_yy sqrt(2)eps_xy]"""
    # print(f"shape u is {u.shape}")#DEBUG
    # print(f"shape x is {x.shape}")#DEBUG

    match dim:
        case 2:
            du = torch.autograd.grad(u[0,:], x, grad_outputs=torch.ones_like(u[0,:]), create_graph=True)[0]
            dv = torch.autograd.grad(u[1,:], x, grad_outputs=torch.ones_like(u[1,:]), create_graph=True)[0]
            return torch.stack([du[:,0], dv[:,1], (1/torch.sqrt(torch.tensor(2)))*(du[:,1] + dv[:,0])],dim=1)

        case 3:
            du = torch.autograd.grad(u[0,:], x, grad_outputs=torch.ones_like(u[0,:]), create_graph=True)[0]
            dv = torch.autograd.grad(u[1,:], x, grad_outputs=torch.ones_like(u[1,:]), create_graph=True)[0]
            dw = torch.autograd.grad(u[2,:], x, grad_outputs=torch.ones_like(u[1,:]), create_graph=True)[0]
            return torch.stack([du[:,:,0], dv[:,:,1], dw[:,:,2],(1/torch.sqrt(torch.tensor(2)))*(dv[:,:,2] + dw[:,:,1]), (1/torch.sqrt(torch.tensor(2)))*(du[:,:,2] + dw[:,:,0]), (1/torch.sqrt(torch.tensor(2)))*(du[:,:,1] + dv[:,:,0])],dim=1)

def Strain_full(u,x, dim = 2):

    match dim:
        case 2:
            du = torch.autograd.grad(u[0,:], x, grad_outputs=torch.ones_like(u[0,:]), create_graph=True)[0]
            dv = torch.autograd.grad(u[1,:], x, grad_outputs=torch.ones_like(u[1,:]), create_graph=True)[0]
            return torch.stack([du[:,0], du[:,1], dv[:,0], dv[:,1]],dim=1)
        


def grad_u_2_3D(u,x, dim = 2):
    """ Return the gradient of u"""
    # print(f"shape u is {u.shape}")#DEBUG
    # print(f"shape x is {x.shape}")#DEBUG

    match dim:
        case 2:
            du = torch.autograd.grad(u[0,:], x, grad_outputs=torch.ones_like(u[0,:]), create_graph=True)[0]
            dv = torch.autograd.grad(u[1,:], x, grad_outputs=torch.ones_like(u[1,:]), create_graph=True)[0]
            return torch.stack([du[:,:], dv[:,:]],dim=1)
        case 3:
            du = torch.autograd.grad(u[0,:], x, grad_outputs=torch.ones_like(u[0,:]), create_graph=True)[0]
            dv = torch.autograd.grad(u[1,:], x, grad_outputs=torch.ones_like(u[1,:]), create_graph=True)[0]
            dw = torch.autograd.grad(u[2,:], x, grad_outputs=torch.ones_like(u[1,:]), create_graph=True)[0]
            return torch.stack([du[:,0,:], dv[:,0,:], dw[:,0,:]],dim=1)



grad_u_2D = grad_u_2_3D

def Green_lagrange(grad_u):
    grad_u_transpose = torch.einsum('exy->eyx',grad_u)
    quad_term = torch.einsum('ekx,eky->exy',grad_u,grad_u)
    return 0.5*(grad_u_transpose+grad_u+quad_term)

def Strain(u,x):
    # Used only for visualisation purposes.
    # Ordering corresponds to Paraview.
    dim = u.shape[0]
    match dim:
        case 2:
            """ Return the vector strain [eps_xx eps_yy eps_xy]"""
            du = torch.autograd.grad(u[0,:], x, grad_outputs=torch.ones_like(u[0,:]), create_graph=True)[0]
            dv = torch.autograd.grad(u[1,:], x, grad_outputs=torch.ones_like(u[1,:]), create_graph=True)[0]

            return torch.stack([du[...,0], dv[...,1], 0.5*(du[...,1] + dv[...,0])],dim=1)
        case 3:
            """ Return the vector strain [3D: XX, YY, ZZ, XY, YZ, XZ]"""
            du = torch.autograd.grad(u[0,:], x, grad_outputs=torch.ones_like(u[0,:]), create_graph=True)[0]
            dv = torch.autograd.grad(u[1,:], x, grad_outputs=torch.ones_like(u[1,:]), create_graph=True)[0]
            dw = torch.autograd.grad(u[2,:], x, grad_outputs=torch.ones_like(u[2,:]), create_graph=True)[0]

            return torch.stack([du[...,0], dv[...,1], dw[...,2], \
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



# # # # # # # # # # # # # # # # # # # # # # # #
#           HEXA mapping functions            #
# # # # # # # # # # # # # # # # # # # # # # # #

def line_point_and_slope(P,m):
    c = P[1] -m*P[0]
    a = m
    b = -1
    return a,b,c

def lines_intersection(a1, b1, c1, a2, b2, c2):
    # Returns the intersection point of two lines

    determinant = a1 * b2 - a2 * b1
    if (determinant == 0):
        # The lines are parallel. This is simplified
        # by returning a pair of (10.0)**19
        return [(10.0)**19, (10.0)**19]
    else:
        x = (b1 * c2 - b2 * c1)/determinant
        y = (-a1 * c2 + a2 * c1)/determinant
        return torch.tensor([x, y])

def line_2_points(P, Q):
    # Returns the line parameters between two points    
    m = (Q[1] - P[1])/(Q[0] - P[0])
    c = P[1] - m * P[0]
    return m, c

def line_btw_points(P, Q):
    # Returns the line parameters between two points
    
    a = -(Q[1] - P[1])
    b = Q[0] - P[0]
    c = -(b * P[1] + a * P[0])
    return a, b, c

def lines_intersection_v2(a1, b1, c1, a2, b2, c2, coord):
    # Returns the intersection point of two lines

    determinant = a1 * b2 - a2 * b1

    x = (b1 * c2 - b2 * c1)/determinant
    y = (-a1 * c2 + a2 * c1)/determinant

    return x, y

def Hexa_identify_regions(coord, important_points, center_point):


    [P_u_l, P_u_r, P_u_middle, centre_u_l, centre_u_r, P_d_l, P_d_r, P_d_middle, centre_d_l, centre_d_r] = important_points


    m,c = line_2_points(center_point,coord )
    angle = (torch.arctan2(coord[1]-center_point[1],coord[0]-center_point[0]))/np.pi*180
    if angle >-30 and angle<=30:  
        regions=1
    elif angle >30 and angle<=90:  
        a, b, c = line_btw_points(P_u_r,P_u_middle)
        if torch.sign(a*center_point[0]+b*center_point[1]+c) == torch.sign(a*coord[0]+b*coord[1]+c):
            regions=2
        else:
            angle = (torch.arctan2(coord[1]-centre_u_r[1],coord[0]-centre_u_r[0]))/np.pi*180
            if angle > -150 and angle <=-90:  
                regions=7
            else:
                regions=8

    elif angle >90 and angle<=150: 
        a, b, c = line_btw_points(P_u_l,P_u_middle)
        if torch.sign(a*center_point[0]+b*center_point[1]+c) == torch.sign(a*coord[0]+b*coord[1]+c):
            regions=3
        else:
            angle = (torch.arctan2(coord[1]-centre_u_l[1],coord[0]-centre_u_l[0]))/np.pi*180
            if angle > -30:  
                regions=9
            else:
                regions=10

    elif (angle >150 and angle<=180) or (angle >=-180 and angle<=-150):  
        regions=4
    elif angle >-150 and angle<=-90:  
        a, b, c = line_btw_points(P_d_l,P_d_middle)
        if torch.sign(a*center_point[0]+b*center_point[1]+c) == torch.sign(a*coord[0]+b*coord[1]+c):
            regions=5
        else:
            angle = (torch.arctan2(coord[1]-centre_d_l[1],coord[0]-centre_d_l[0]))/np.pi*180
            if angle > 30:  
                regions=11
            else:
                regions=12

    elif angle >-90 and angle<=-30:  
        a, b, c = line_btw_points(P_d_r,P_d_middle)
        if torch.sign(a*center_point[0]+b*center_point[1]+c) == torch.sign(a*coord[0]+b*coord[1]+c):
            regions=6
        else:
            angle = (torch.arctan2(coord[1]-centre_d_r[1],coord[0]-centre_d_r[0]))/np.pi*180
            if abs(angle) > 150:  
                regions=13
            else:
                regions=14

    return regions

def Hexa_domain_constants(Mesh_object):

    tags_unique = np.arange(151,169)

    # Inner edges 
    inner_edges = [np.where(tags_unique==167)[0][0], np.where(tags_unique==168)[0][0], np.where(tags_unique==163)[0][0],\
                np.where(tags_unique==164)[0][0], np.where(tags_unique==165)[0][0], np.where(tags_unique==166)[0][0],\
                np.where(tags_unique==158)[0][0], np.where(tags_unique==157)[0][0],
                np.where(tags_unique==155)[0][0], np.where(tags_unique==154)[0][0],
                np.where(tags_unique==152)[0][0], np.where(tags_unique==151)[0][0],
                np.where(tags_unique==161)[0][0], np.where(tags_unique==160)[0][0]
                ]

    top_bottom_edges = [np.where(tags_unique==162)[0][0], np.where(tags_unique==156)[0][0]]

    domain_y = 1.0
    domain_x = domain_y * np.sqrt(3)/1.5/2 

    y_min = min(node[2] for node in Mesh_object.Nodes)
    y_max = max(node[2] for node in Mesh_object.Nodes)

    x_min = min(node[1] for node in Mesh_object.Nodes)
    x_max = max(node[1] for node in Mesh_object.Nodes)

    edge_right_a, edge_right_b, edge_right_c = line_btw_points([domain_x,0], [domain_x,domain_y])        
    edge_left_a, edge_left_b, edge_left_c = line_btw_points([0,0], [0,domain_y])        
    edge_middle_a, edge_middle_b, edge_middle_c = line_btw_points([domain_x/2,0], [domain_x/2,domain_y])  

    edge_bottom_a, edge_bottom_b, edge_bottom_c = line_btw_points([x_min,y_min], [x_max,y_min])        
    edge_top_a, edge_top_b, edge_top_c = line_btw_points([x_min,y_max], [x_max,y_max])        

    # # # # # # # # # # # # # # # # # #
    # NOTE: the center of the hexagon is not neceserily in the center of the RVE!
    center_edges = np.where(tags_unique==164)[0][0]
    # print(center_edges)

    center_ids = np.unique(Mesh_object.borders_nodes[center_edges])
    coord_y = []
    for c_i in center_ids:
        coord_y.append(Mesh_object.Nodes[int(c_i-1)][2])
    center_y = (np.mean(coord_y))

    center_point = torch.tensor((domain_x/2, center_y ), requires_grad=True)
    # print("center_point : ", center_point)
    # print("y min = ", y_min)
    # print("y max = ", y_max)
    # print()
    # # # # # # # # # # # # # # # # # #


    # print(max(node[2] for node in Mesh_object.Nodes) - min(node[2] for node in Mesh_object.Nodes))
    # # # # ----- UP ------ # # #
    # print("UP")
    line_a, line_b, line_c = line_point_and_slope(center_point,np.tan(np.pi/180*30))
    # print(line_a, line_b, line_c)
    P_u_r = lines_intersection(line_a, line_b, line_c, edge_right_a, edge_right_b, edge_right_c).data
    # print(" P right = ", P_u_r)

    line_a, line_b, line_c = line_point_and_slope(center_point,np.tan(np.pi/180*150))
    P_u_l = lines_intersection(line_a, line_b, line_c, edge_left_a, edge_left_b, edge_left_c).data
    # print(" P left = ", P_u_l)

    line_a, line_b, line_c = line_point_and_slope(P_u_l,np.tan(np.pi/180*30))
    P_u_middle = lines_intersection(line_a, line_b, line_c,edge_middle_a, edge_middle_b, edge_middle_c).data
    # print(" P middle = ", P_u_middle)

    centre_u_r = lines_intersection(line_a, line_b, line_c, edge_right_a, edge_right_b, edge_right_c).data
    # print(" centre neighbour right = ", centre_u_r)

    centre_u_r_c = lines_intersection(line_a, line_b, line_c, edge_top_a, edge_top_b, edge_top_c).data
    # print(" centre neighbour right center = ", centre_u_r_c)

    centre_u_l = np.array([0, centre_u_r[1]])
    # print(" centre neighbour = ", centre_u_l)

    centre_u_l_c = np.array([ - (centre_u_r_c[0] - x_max), y_max])
    # print(" centre neighbour left center = ", centre_u_l_c)
    # print()


    # # # # ---- DOWN ----- # # #
    # print("DOWN")

    line_a, line_b, line_c = line_point_and_slope(center_point,np.tan(np.pi/180*150))
    P_d_r = lines_intersection(line_a, line_b, line_c, edge_right_a, edge_right_b, edge_right_c).data
    # print(" P right = ", P_d_r)

    line_a, line_b, line_c = line_point_and_slope(center_point,np.tan(np.pi/180*30))
    P_d_l = lines_intersection(line_a, line_b, line_c, edge_left_a, edge_left_b, edge_left_c).data
    # print(" P left = ", P_d_l)



    line_a, line_b, line_c = line_point_and_slope(P_d_r,np.tan(np.pi/180*30))
    P_d_middle = lines_intersection(line_a, line_b, line_c, edge_middle_a, edge_middle_b, edge_middle_c).data
    # print(" P middle = ", P_d_middle)

    centre_d_l = lines_intersection(line_a, line_b, line_c, edge_left_a, edge_left_b, edge_left_c).data
    # print(" centre neighbour left = ", centre_d_l)

    centre_d_l_c = lines_intersection(line_a, line_b, line_c, edge_bottom_a, edge_bottom_b, edge_bottom_c).data
    # print(" centre neighbour left c = ", centre_d_l_c)


    centre_d_r = np.array([domain_x, centre_d_l[1]])
    # print(" centre neighbour right = ", centre_d_r)

    centre_d_r_c = np.array([domain_x - centre_d_l_c[0], centre_d_l_c[1]])

    # print(" centre neighbour right c = ", centre_d_r_c)

        # region    1               2           3           4               5           6           7           8
    P1 = [[domain_x,0],         P_u_r,      P_u_l,      [0,0],          P_d_l,      P_d_r,      P_u_r,      [domain_x/2,0],\
        #     9               10          11         12               13              14
            [domain_x/2,0],     P_u_l,      P_d_l,     [domain_x/2,0],  [domain_x/2,0], P_d_r ]

    # region    1               2           3           4               5           6           7           8
    P2 = [[domain_x,domain_y],  P_u_middle, P_u_middle, [0,domain_y],   P_d_middle, P_d_middle, P_u_middle, [domain_x/2,domain_y],\
            #     9                 10           11          12                    13                       14
            [domain_x/2,domain_y], P_u_middle, P_d_middle, [domain_x/2,domain_y], [domain_x/2,domain_y], P_d_middle]

    centers = [center_point, center_point, center_point, center_point, center_point, center_point, centre_u_r, centre_u_r_c,\
            centre_u_l_c, centre_u_l, centre_d_l, centre_d_l_c, centre_d_r_c, centre_d_r ]
    
    impo_points = [P_u_l, P_u_r, P_u_middle, centre_u_l, centre_u_r, P_d_l, P_d_r, P_d_middle, centre_d_l, centre_d_r]

    return P1, P2, centers, impo_points, inner_edges, center_point

def Hexa_scaling_new_coord(center_point, coord, end_a, end_b, end_c, start_a, start_b, start_c, new_h, current_h):
    # end point = inner edge of the microstructure ~ centerline
    # start point = actual outer edge (given by boundary tag)
    
    eps = 1.0e-16
    line_a, line_b, line_c = line_btw_points(center_point, coord)


    end_point_x, end_point_y  = lines_intersection_v2(line_a, line_b, line_c, end_a, end_b, end_c, coord)
    start_point_x, start_point_y = lines_intersection_v2(line_a, line_b, line_c, start_a, start_b, start_c, coord)
    
    r_x = (end_point_x - coord[0] )/(end_point_x - start_point_x + eps)  
    r_y = (end_point_y - coord[1] )/(end_point_y - start_point_y + eps)  

    k_new_x = (new_h/current_h)*r_x
    k_new_y = (new_h/current_h)*r_y

    new_coord_x = (1-k_new_x)*end_point_x + (k_new_x)*start_point_x
    new_coord_y = (1-k_new_y)*end_point_y + (k_new_y)*start_point_y

    return new_coord_x, new_coord_y


def Hexa_mapping(mesh, xg, current_h, new_h):

    P1, P2, centers, impo_points, inner_edges, center_point = Hexa_domain_constants(mesh)

    F = []
    u = []
    J = []
    J_export = []
    xg_modes = []

    for i in range(len(xg)):
        coord = torch.tensor(xg[i],requires_grad=True)

        region = Hexa_identify_regions(coord, impo_points, center_point.data)
        end_a, end_b, end_c = line_btw_points(P1[region-1], P2[region-1])   

        n = len(mesh.borders_nodes[inner_edges[region-1]])
        if n==2:
            idx_a, idx_b = mesh.borders_nodes[inner_edges[region-1]][0], mesh.borders_nodes[inner_edges[region-1]][-1]
        else:
            mid = n // 2
            idx_a = mesh.borders_nodes[inner_edges[region-1]][mid - 1]
            idx_b = mesh.borders_nodes[inner_edges[region-1]][mid + 1] if mid + 1 < n else mesh.borders_nodes[inner_edges[region-1]][mid]

        A = mesh.Nodes[int(idx_a[0])-1][1:3]
        B = mesh.Nodes[int(idx_b[0])-1][1:3]
        start_a, start_b, start_c = line_btw_points(A, B)   

        new_coord_x, new_coord_y = Hexa_scaling_new_coord(torch.tensor(centers[region-1], requires_grad=True), coord, end_a, end_b, end_c, start_a, start_b, start_c, new_h, current_h)

        new_coord = torch.vstack((new_coord_x, new_coord_y))

        dx_dX = torch.autograd.grad(new_coord_x, coord, grad_outputs=torch.ones_like(new_coord_x), create_graph=True)[0]
        dy_dX = torch.autograd.grad(new_coord_y, coord, grad_outputs=torch.ones_like(new_coord_y), create_graph=True)[0]

        # print("new h = ", new_h)
        # print("coord = ", coord)

        # print("new_coord_x = ", new_coord_x)
        # print("dx_dX = ", dx_dX)
        # print("J = ", dx_dX[0]*dy_dX[1] - dx_dX[1]*dy_dX[0])
        # print()


        u.append(new_coord - torch.unsqueeze(coord,dim=1))
        F.append(torch.tensor([[dx_dX[0],dx_dX[1]],[dy_dX[0],dy_dX[1]]]))
        J.append(dx_dX[0]*dy_dX[1] - dx_dX[1]*dy_dX[0])
        J_export.append((dx_dX[0]*dy_dX[1] - dx_dX[1]*dy_dX[0]).detach())
    return u, F, J, J_export



def Hexa_mapping_non_vect(model, current_h, new_h):

    space_model = model.Space_modes[0]

    P1, P2, centers, impo_points, inner_edges, center_point = Hexa_domain_constants(space_model.mesh)

    xg_modes = []

    for i in range(model.n_modes_truncated):
        u_k,xg_k,detJ_k = model.Space_modes[i]()
        xg_modes.append(xg_k)
    xg = xg_modes[0]

    F = []
    u = []
    J = []

    for i in range(len(xg)):
        coord = torch.tensor(xg[i],requires_grad=True)

        region = Hexa_identify_regions(coord, impo_points, center_point.data)
        end_a, end_b, end_c = line_btw_points(P1[region-1], P2[region-1])   

        n = len(space_model.mesh.borders_nodes[inner_edges[region-1]])
        if n==2:
            idx_a, idx_b = space_model.mesh.borders_nodes[inner_edges[region-1]][0], space_model.mesh.borders_nodes[inner_edges[region-1]][-1]
        else:
            mid = n // 2
            idx_a = space_model.mesh.borders_nodes[inner_edges[region-1]][mid - 1]
            idx_b = space_model.mesh.borders_nodes[inner_edges[region-1]][mid + 1] if mid + 1 < n else space_model.mesh.borders_nodes[inner_edges[region-1]][mid]

        A = space_model.mesh.Nodes[int(idx_a[0])-1][1:3]
        B = space_model.mesh.Nodes[int(idx_b[0])-1][1:3]
        start_a, start_b, start_c = line_btw_points(A, B)   

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

        # new_coord_x, new_coord_y = Hexa_scaling_new_coord(torch.tensor(centers[region-1], requires_grad=True), coord, end_a, end_b, end_c, start_a, start_b, start_c, new_h, current_h)
        # new_coord = torch.vstack((new_coord_x, new_coord_y))
        # dx_dX = torch.autograd.grad(new_coord_x, coord, grad_outputs=torch.ones_like(new_coord_x), create_graph=True)[0]
        # dy_dX = torch.autograd.grad(new_coord_y, coord, grad_outputs=torch.ones_like(new_coord_y), create_graph=True)[0]

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

        F_in = []
        u_in = []
        J_in = []

        for j in range(new_h.shape[0]):
            new_coord_x, new_coord_y = Hexa_scaling_new_coord(torch.tensor(centers[region-1], requires_grad=True), coord, end_a, end_b, end_c, start_a, start_b, start_c, new_h[j], current_h[j])
            new_coord = torch.vstack((new_coord_x, new_coord_y))

            dx_dX = torch.autograd.grad(new_coord_x, coord, grad_outputs=torch.ones_like(new_coord_x), create_graph=True)[0]
            dy_dX = torch.autograd.grad(new_coord_y, coord, grad_outputs=torch.ones_like(new_coord_y), create_graph=True)[0]

            u_in.append(new_coord - torch.unsqueeze(coord,dim=1))
            F_in.append(torch.tensor([[dx_dX[0],dx_dX[1]],[dy_dX[0],dy_dX[1]]]))
            J_in.append(dx_dX[0]*dy_dX[1] - dx_dX[1]*dy_dX[0])

        u.append(u_in)
        J.append(J_in)
        F.append(F_in)

    return F, J