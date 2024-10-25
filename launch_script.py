from neurom import main
import torch 
# from torch.backends import opt_einsum
from torch.backends import opt_einsum
opt_einsum.enabled = True
# torch.manual_seed(0)
torch.manual_seed(9975355953847005664)
# print(torch.__version__)
# print(torch.seed())
main()