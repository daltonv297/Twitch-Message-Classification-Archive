import torch
import torch.nn as nn
from power_spherical import PowerSpherical
import math
d = 300
loc = torch.tensor([0.] * (d-1) + [1.], requires_grad=False)
scale = torch.tensor(4., requires_grad=False)
dist = PowerSpherical(loc, scale)

samples = dist.sample((50,))

loc_new = torch.tensor([1.] + [0.] * (d-1), requires_grad=True)
#loc_new = torch.tensor([0.] * (d-1) + [1.], requires_grad=False)
scale_new = torch.tensor(1., requires_grad=True)
dist_new = PowerSpherical(loc_new, scale_new)

print('initial loc: ', loc_new)

# def forward(x):
#     return dist_new.log_prob(x)

def loss(x):
    return -torch.sum(dist_new.log_prob(x))

learning_rate = 0.01
n_iters = 500

#optimizer = torch.optim.SGD([scale_new, loc_new], lr=learning_rate)

print('loc diff = ', torch.norm(loc_new - loc).item())
l = loss(samples)

for epoch in range(n_iters):
    l_prev = l
    l = loss(samples)
    l.backward()

    #optimizer.step()

    with torch.no_grad():
        scale_new -= learning_rate * scale_new.grad
        loc_new -= learning_rate * loc_new.grad
        loc_new /= torch.norm(loc_new)


    #optimizer.zero_grad()

    scale_new.grad.zero_()
    loc_new.grad.zero_()

    dist_new = PowerSpherical(loc_new, scale_new)

    if epoch % 10 == 0:
        #print('epoch ', epoch, ": scale = ", scale_new.item(), ', loss = ', l.item())
        print('epoch ', epoch, ": loc diff = ", torch.norm(loc_new - loc).item(), "scale = ", scale_new.item(), 'loss diff= ', abs(l.item()-l_prev.item()), 'loss= ', l.item())
