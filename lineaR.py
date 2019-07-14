#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

#%%
noise = np.random.randn(10, 1)
# print(noise.shape)
W = np.array([[5, 3]]).T
x = np.array(np.arange(10))[None, :].T
# print(x.shape)
x_ = np.ones((10, 1))
X = np.concatenate((x_, x), axis = 1)

print(X.shape, W.shape)
y = X @ W + noise
print(y)

plt.scatter(x, y, s = 10, marker='o', c = 'red')

#%%
X = torch.Tensor(X)
y = torch.Tensor(y)

#%%
class LR(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = torch.autograd.Variable(torch.ones(2,1), requires_grad = True)
        
    def forward(self, X):
        return X @ self.W

#%%
m = LR()
loss = nn.MSELoss()

#%%
# opt = optim.Adam([m.W], lr=0.1)
opt = optim.SGD([m.W], lr=0.01)
opt.zero_grad()
#%%

epochs = np.arange(500)
cl = []
W_hat_l = []
for i in epochs:
    W_hat_l.append(m.W.data.numpy().copy())
    mse = loss(m(X), y)
    opt.zero_grad()
    mse.backward()
    opt.step()
    cl.append(mse.item())
    print(f"Epoch:{i} - Weight - {m.W} - loss: {mse.item()}")


#%%
W_hat_l = np.concatenate(W_hat_l, axis = 1)
plt.plot(epochs, [W[0, 0]] * len(epochs), 'r--')
plt.plot(epochs, [W[1, 0]] * len(epochs), 'b--')
plt.plot(epochs, W_hat_l[0, :], 'r.')
plt.plot(epochs, W_hat_l[1, :], 'b.')

plt.show()


#%%
del m

#%%
