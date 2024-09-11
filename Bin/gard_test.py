import torch
import numpy as numpy
import torch.nn as nn



class NET(torch.nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.P = nn.ParameterList([nn.Parameter(torch.tensor([1],dtype=torch.float64,requires_grad=True)) for i in range(3)])

    def forward(self, x):

        self.P[0] = nn.Parameter(torch.tensor([0],dtype=torch.float64,requires_grad=True))
        self.P[1] = 2*self.P[2]

        params = self.P
        # params = torch.ones(3)
        # params[0] = nn.Parameter(torch.tensor([0],dtype=torch.float64,requires_grad=True))
        # params[1] = 2*self.P[2]
        # params[2] = self.P[2]

        y = x[0] * params[0] + x[1] * params[1] + x[2] * params[2] 
        return y.sum()


net = NET()
x = torch.ones(3)
print("x = ", x)

optim = torch.optim.Adam(net.parameters(), lr=0.001)


loss = net(x)
loss.backward()

grads = [[p, p.grad] for p in net.parameters()]

print("loss = ", loss)
print("grads = ", grads)
