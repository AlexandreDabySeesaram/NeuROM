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

def PotentialEnergyVectorisedBiParametric(model,A, E, u, x, b):
    """Computes the potential energy of the Beam, which will be used as the loss of the HiDeNN"""
    with torch.no_grad(): #No derivatino of heavy side, check if that still works
        f_1 = torch.heaviside(x-5,torch.tensor(1, dtype = torch.float32))
        f_2 = 1-torch.heaviside(x-5,torch.tensor(1, dtype = torch.float32))
    f_1_E_1 = torch.einsum('ik,jk->ij',f_1,E[0])
    f_2_E_2 = torch.einsum('ik,jk->ij',f_2,E[1])
    E_tensor = f_1_E_1[:,:,None]+f_2_E_2[:,None,:]
    # plt.plot(E_tensor[:,10,0].data.view(-1))
    # plt.savefig('Results/E_tensor.pdf')
    # plt.clf()
    Space_modes = [model.Space_modes[l](x) for l in range(model.n_modes)]
    u_i = torch.cat(Space_modes,dim=1)  
    for mode in range(model.n_modes):
        Para_mode_List = [model.Para_modes[mode][l](E[l][:,0].view(-1,1))[:,None] for l in range(model.n_para)]
        if mode == 0:
            lambda_i = [torch.unsqueeze(Para_mode_List[l],dim=0) for l in range(model.n_para)]
        else:
            New_mode = Para_mode_List
            lambda_i = [torch.vstack((lambda_i[l],torch.unsqueeze(New_mode[l],dim=0))) for l in range(model.n_para)]
    du_dx = [torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0] for u_x in Space_modes]
    du_dx = torch.cat(du_dx,dim=1)  
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
