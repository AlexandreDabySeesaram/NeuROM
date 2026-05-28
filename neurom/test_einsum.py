import torch

e = 10
m = 5
p = 3
t = 4
s = 2

un = torch.rand(e, m)
P = torch.rand(e)
detJ = torch.rand(e)
L0 = torch.rand(m, p)
L1 = torch.rand(m, t)
L2 = torch.rand(m, s)

# Using einsum
W1 = torch.einsum('em,e,e,mp,mt,ms->', un, P, detJ, L0, L1, L2)

# Using separated sums
I_m = torch.einsum('em,e,e->m', un, P, detJ)
S0 = torch.sum(L0, dim=1)
S1 = torch.sum(L1, dim=1)
S2 = torch.sum(L2, dim=1)
W2 = torch.sum(I_m * S0 * S1 * S2)

print("W1:", W1.item())
print("W2:", W2.item())
print("Diff:", abs(W1.item() - W2.item()))

