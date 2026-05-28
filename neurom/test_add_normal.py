import torch
import torch.nn as nn

def BoundaryStiffnessEnergy_para_normal(model, k, E, u_ref=0):
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
    
    # Try to use the user's normals, otherwise compute them from the mesh nodes
    try:
        n = model.Space_modes[0].mesh.normals.to(u_i_b.device)
        if n.shape[0] != u_i_b.shape[1]:
            print(f"Warning: mesh.normals shape {n.shape} does not match number of boundary elements {u_i_b.shape[1]}. Computing geometrically...")
            raise ValueError()
    except:
        # Fallback to computing normals geometrically from the borders_nodes
        coords = model.Space_modes[0].coordinates_all.to(u_i_b.device)
        border_nodes = torch.tensor(model.Space_modes[0].borders_nodes, dtype=torch.long, device=u_i_b.device) - 1
        v0 = coords[border_nodes[:, 0]]
        v1 = coords[border_nodes[:, 1]]
        v2 = coords[border_nodes[:, 2]]
        n = torch.cross(v1 - v0, v2 - v0, dim=1)
        n = n / torch.norm(n, dim=1, keepdim=True)
        
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
        
        W_uu = k * torch.einsum('em,el,e,mp,lp,mt,lt,ms,ls->', un_i_b, un_i_b, torch.abs(detJ_full), L0, L0, L1, L1, L2, L2)
        
        if is_u_ref_nonzero:
            W_u_ref = k * torch.einsum('em,e,mp,mt,ms,e->', un_i_b, torch.abs(detJ_full), L0, L1, L2, u_ref_n)
            W_ref_ref = k * torch.sum(u_ref_n**2 * torch.abs(detJ_full)) * L0.shape[1] * L1.shape[1] * L2.shape[1]
            W = W_uu - 2 * W_u_ref + W_ref_ref
        else:
            W = W_uu
            
        return W / (E[0].shape[0] * E[1].shape[0] * E[2].shape[0])

