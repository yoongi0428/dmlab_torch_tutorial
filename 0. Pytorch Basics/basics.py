import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

# REFERENCE : https://github.com/hunkim/PyTorchZeroToAll

# 1. Basic case
print("=====1. Basic case=====")

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
print()

# 2. Basic Model Practice (Linear Regression)
print("=====2. Basic Model Practice (Linear Regression)=====")

x = torch.Tensor([[1.0], [2.0], [3.0]])
y = torch.Tensor([[2.0], [4.0], [6.0]])

# To build machine learning model using pytorch,
# you have to build python class which inherits "nn.Module".
# In the constructor "__init__", you would set up variables and modules such as linear, activation so on.
# Through the function 'forward', model compute logits using modules defined in the constructor.
class SimpleLinearRegression(nn.Module):
     def __init__(self):
          super(SimpleLinearRegression, self).__init__()
          # nn.Linear contains weight and bias.
          self.linear = nn.Linear(in_features=1, out_features=1)

     def forward(self, x):
          return self.linear(x)

# Instantiate your own model
model = SimpleLinearRegression()

# Define loss function and optimizer
# "model.parameters()" in optimizer passes the learnable parameters to the optimizer.
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
     # Clear gradients before training
     optimizer.zero_grad()

     # Pass x to the model and get predicted logit
     logit = model(x)

     # Compute loss
     loss = criterion(logit, y)

     # Compute gradients of parameters from loss and pass back
     # With gradients from loss.backward(), optimizer updates parameters
     loss.backward()
     optimizer.step()

# Evaluate performance
test_x = Variable(torch.Tensor([[4.0]]))
y_pred = model(test_x)
print("model(4) = %f\n" % y_pred)

# 3. Logistic Regression
print("=====3. Logistic Regression=====")
x = torch.Tensor([[1], [2], [3], [4]])
y = torch.Tensor([[0], [0], [1], [1]])

# Define model
class SimpleLogisticRegression(nn.Module):
     def __init__(self):
          super(SimpleLogisticRegression, self).__init__()
          # Linear layer with activation function
          self.linear = nn.Linear(in_features=1, out_features=1)
          self.sigmoid = nn.Sigmoid()

     def forward(self, x):
          return self.sigmoid(self.linear(x))

# Instantiate model
model = SimpleLogisticRegression()

# For logistic regression, we use binary cross entropy loss.
# logit * log(y) + (1 - logit) * log(1 - y)
# For nn.BCELoss, logit and y must be same shape.
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training procedure
for epoch in range(10000):
     optimizer.zero_grad()

     logit = model(x)

     loss = criterion(logit, y)

     loss.backward()
     optimizer.step()

# Evaluation
test_x = Variable(torch.Tensor([[1]]))
y_pred = model(test_x)
test_x2 = Variable(torch.Tensor([[3]]))
y_pred2 = model(test_x2)
print("model(1) = %f" % y_pred)
print("model(3) = %f" % y_pred2)