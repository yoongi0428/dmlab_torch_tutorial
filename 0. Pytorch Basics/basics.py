import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

# REFERENCE : https://github.com/hunkim/PyTorchZeroToAll

# 1. Basic case

# Basic components of Pytorch
x = torch.Tensor([1.0, 2.0, 3.0])
y = torch.Tensor([2.0, 4.0, 6.0])

# Unlike Tensorflow, Pytorch tensor is just a multi-dimensional matrix
# It is 'Variable' which is considered so called "parameter" of a model
# 'requires_grad' option enables 'Variable' to get gradient from the computational graph
w = Variable(torch.Tensor([1.0]), requires_grad=True)


lr = 0.01
for epoch in range(100):
     # Simple regression
     hypothesis = x * w

     # Compute loss. Must be scalar
     loss = torch.mean((y - hypothesis) * (y - hypothesis))

     # Function "backward" compute gradient of loss
     # w.r.t all variables which requires gradients
     loss.backward(retain_graph=True)

     # Manually update weight and clear gradients
     w.data -= lr * w.grad.data
     w.grad.data.zero_()

print('Predict : ', x * w)
print('Weight : ', w)


# 2. Basic Model Practice (Linear Regression)

