import torch
import torch.nn as nn

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

