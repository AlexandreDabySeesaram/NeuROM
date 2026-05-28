import torch
import torch.nn as nn

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
        W_uu = k * torch.einsum('xem,xel,e,mp,lp,mt,lt,ms,ls->', u_i_b, u_i_b, torch.abs(detJ_full), L0, L0, L1, L1, L2, L2)
        
        if is_u_ref_nonzero:
            u_ref_t = u_ref * torch.ones(u_i_b.shape[0], device=u_i_b.device)
            W_u_ref = k * torch.einsum('xem,e,mp,mt,ms,x->', u_i_b, torch.abs(detJ_full), L0, L1, L2, u_ref_t)
            W_ref_ref = k * torch.sum(u_ref_t**2) * torch.sum(torch.abs(detJ_full)) * L0.shape[1] * L1.shape[1] * L2.shape[1]
            W = W_uu - 2 * W_u_ref + W_ref_ref
        else:
            W = W_uu
            
        return W / (E[0].shape[0] * E[1].shape[0] * E[2].shape[0])

