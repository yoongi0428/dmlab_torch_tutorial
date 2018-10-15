import torch
import torch.nn as nn
import numpy as np
from utils.mnist_reader import load_mnist

def batch_loader(data, n=1):
    x = data[0]
    y = data[1]
    data_len = len(x)

    for i in range(0, data_len, n):
        yield x[i: min(data_len, i + n), :], y[i: min(data_len, i + n)]

"""
0 : T-shirt/top
1 : Trouser
2 : Pullover
3 : Dress
4 : Coat
5 : Sandal
6 : Shirt
7 : Sneaker
8 : Bag
9 : Ankle boot
"""

# Hyperparameter
data_path = 'data/'
epochs = 10
batch = 100
lr = 0.001
cuda = True

# Load train data
train_x, train_y = load_mnist(data_path, kind='train')
test_x, test_y = load_mnist(data_path, kind='test')
input_dim = 784
output_dim = 10
test_num = len(test_x)

################### MLP ###################
class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()


    def forward(self, x):


        return 1

hidden_dim = 256
model = RNN(input_dim, hidden_dim, output_dim)

criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=lr)


if cuda:
    model.cuda()
    train_x = train_x.cuda()
    train_y = train_y.cuda()
    test_x = test_x.cuda()
    test_y = test_y.cuda()

print('='*10 + ' MLP Training Start ' + '='*10)
for epoch in range(1, epochs + 1):
    loss = 0.0
    for i, (batch_x, batch_y) in enumerate(batch_loader((train_x, train_y), batch)):
        optim. zero_grad()

        out = model(batch_x)
        l = criterion(out, batch_y)

        l.backward()
        optim.step()

        loss += l

    print("[Epoch %3d] Loss : %.4f" % (epoch, loss))

print('[MLP Test Start]')
pred = []
for (batch_x, _) in batch_loader((test_x, test_y), batch):
    out = model(batch_x)
    p = torch.argmax(out, -1)
    pred.append(p)

pred = np.concatenate(pred)
num_correct = len(np.where(pred == test_y)[0])
accuracy = num_correct / test_num

print('Accuracy = %.4f\n\n' % accuracy)


print('end')