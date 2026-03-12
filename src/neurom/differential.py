import torch


def grad_factory(dim):
    if dim == 1:
        return grad_1D


def grad_1D(x, u):
    du_dx = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]

    return du_dx
