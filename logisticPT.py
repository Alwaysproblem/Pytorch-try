#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

#%%
# generate some data 2-dimension. shape = (10, 2)
sample_n = 100
meana = np.array([1, 1])
cova = np.array([[0.1, 0],[0, 0.1]])

meanb = np.array([2, 2])
covb = np.array([[0.1, 0],[0, 0.1]])

x_red = np.random.multivariate_normal(mean=meana, cov = cova, size=sample_n)
x_green = np.random.multivariate_normal(mean=meanb, cov = covb, size=sample_n)

y_red = np.array([1] * sample_n)
y_green = np.array([0] * sample_n)

plt.scatter(x_red[:, 0], x_red[:, 1], c = 'red' , marker='.', s = 30)
plt.scatter(x_green[:, 0], x_green[:, 1], c = 'green', marker='.', s = 30)
plt.show()

X = np.concatenate([x_red, x_green])
X = np.concatenate([np.ones((sample_n*2, 1)), X], axis = 1)
y = np.concatenate([y_red, y_green])

y = y[:, None]

assert X.shape == (sample_n*2, 3)
assert y.shape == (sample_n*2, 1)

X = torch.Tensor(X)
y = torch.Tensor(y)


#%%
class LogR(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = Variable(torch.ones(3, 1), requires_grad = True)
        self.sig = nn.Sigmoid()

    def forward(self, X):
        return  self.sig(X @ self.W)


#%%
loss = nn.BCELoss()
m = LogR()

opt = optim.LBFGS([m.W], lr = 0.01)
# opt = optim.Adam([m.W], lr = 0.01)
# opt.zero_grad()

#%%
epochs = np.arange(1000)
cl = []
for i in epochs:
    bce = loss(m(X), y)

    ##as for Pytorch it must be done in backprop
    opt.zero_grad()

    bce.backward()
    
    # for LBFGS
    def closure():
        opt.zero_grad()
        out = m(X)
        rB = loss(out, y)
        rB.backward()
        return rB
    opt.step(closure)

    # opt.step()
    cl.append(bce.item())
    print(f"Epoch:{i} - Weight - {m.W} - loss: {bce.item()}")

#%%
plt.plot(epochs, cl)
plt.show()

#%%
W_hat = m.W.data.numpy().copy()
x = np.arange(-1, 3, step = 0.01)
y_ = - W_hat[0,0] / W_hat[2, 0] - W_hat[1, 0] * x/ W_hat[2, 0]
plt.scatter(x_red[:, 0], x_red[:, 1], c = 'red' , marker='.', s = 30)
plt.scatter(x_green[:, 0], x_green[:, 1], c = 'green', marker='.', s = 30)
plt.plot(x, y_)
plt.show()

#%%
del m

#%%
