import torch
import sys

try:
    from neurom.src.PDE_Library import compute_jacobian, InternalEnergy_2_3D_einsum
    
    N = 10
    dim = 3
    x = torch.randn(N, dim, requires_grad=True)
    # Define a simple u tensor
    # u must be of shape (dim, N)
    w = torch.randn(dim, dim)
    u = torch.matmul(w, x.T) # shape (dim, N)
    
    W_e = InternalEnergy_2_3D_einsum(u, x, lmbda=1.0, mu=1.0, dim=dim)
    print(f"W_e shape: {W_e.shape}")
    print("Success!")
except Exception as e:
    import traceback
    traceback.print_exc()

