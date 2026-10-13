import sys
from pathlib import Path
ROOT = Path("/Users/solal/Desktop/neurom/nl_pgd_beam_buckling/examples/1d_beam_hermite")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "kolmogorov_study"))
import torch
from torch.nn.utils import parametrize
from neurom.quadratures import FourPoints1D
from main import build_problem
from utils import grad_hessian
from moving_load import bspline_load

class Linear(torch.nn.Module):
    def __init__(self, T):
        super().__init__(); self.register_buffer("T", T)
    def forward(self, z):
        return (self.T @ z.flatten()).view_as(z)
    def right_inverse(self, q):
        return torch.linalg.solve(self.T, q.flatten()).view_as(q)

pb = build_problem(100, 100.0, k=1e4, load=bspline_load(2851.35, 0.02, 0.5), quadrature=FourPoints1D(), noise=0.1, seed=0)
m = pb["model"]
pi0 = m().item()
_, _, H = grad_hessian(m)
n = H.shape[0]
T = torch.diag(torch.rand(n) + 0.5) + 0.01 * torch.randn(n, n)
parametrize.register_parametrization(pb["u"], "values_reduced", Linear(T))
print("params:", [(nm, p.shape) for nm, p in m.named_parameters()])
print("Pi before/after:", pi0, m().item())
(z,) = m.parameters()
g = torch.autograd.grad(m(), z)[0]
print("grad shape", g.shape, "norm", g.norm().item())
