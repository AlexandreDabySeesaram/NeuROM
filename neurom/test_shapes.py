import torch

m = 3
b = 3

S_g_m = torch.ones(m)
print("Initial S_g_m:", S_g_m.shape)

lam = torch.ones(m, b)
S_g_m = S_g_m * torch.sum(lam, dim=1)
print("Final S_g_m:", S_g_m.shape)

S_g_m2 = torch.zeros(m)
lam_3d = torch.ones(m, b, 1)
S_g_m2 = S_g_m2 * torch.sum(lam_3d[:, :, 0], dim=1)
print("Final S_g_m2:", S_g_m2.shape)

