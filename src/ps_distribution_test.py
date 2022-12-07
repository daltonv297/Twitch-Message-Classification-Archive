import torch
import torch.nn as nn
from power_spherical import PowerSpherical
import math
d = 10
loc = torch.tensor([0.] * (d-1) + [1.], requires_grad=False)
scale = torch.tensor(4., requires_grad=False)
dist = PowerSpherical(loc, scale)

samples = dist.sample((1000,))

loc_new = torch.tensor([1.] + [0.] * (d-1), requires_grad=True)
#loc_new = torch.tensor([0.] * (d-1) + [1.], requires_grad=True)
scale_new = torch.tensor(4., requires_grad=False)
dist_new = PowerSpherical(loc_new, scale_new)
#print(samples[0])
# print(torch.sum(dist.log_prob(samples)))
# dist_new.scale = torch.tensor(4., requires_grad=False)
# print(torch.sum(dist_new.log_prob(samples)))

def forward(x):
    return dist_new.log_prob(x)

def loss(x):
    return -torch.sum(dist_new.log_prob(x))

learning_rate = 0.001
n_iters = 1000

#loss = nn.NLLLoss()
optimizer = torch.optim.SGD([loc_new], lr=learning_rate)

for epoch in range(n_iters):

    l = loss(samples)

    l.backward()

    optimizer.step()

    #with torch.no_grad():

    #     scale_new -= learning_rate * scale_new.grad
        #loc_new -= learning_rate * loc_new.grad

    loc_new = torch.tensor(loc_new / torch.norm(loc_new), requires_grad=True)

    optimizer.zero_grad()
    #scale_new.grad.zero_()
    #loc_new.grad.zero_()

    dist_new = PowerSpherical(loc_new, scale_new)

    if epoch % 10 == 0:
        print('epoch ', epoch, ": scale = ", scale_new.item(), ', loss = ', l.item())

print('loc_new: ', loc_new)