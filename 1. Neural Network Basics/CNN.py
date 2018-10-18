import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.mnist_reader import load_mnist, shuffle
import matplotlib.pyplot as plt

def batch_loader(data, n=1):
    x = data[0]
    y = data[1]
    data_len = len(x)

    for i in range(0, data_len, n):
        yield x[i: min(data_len, i + n), :], y[i: min(data_len, i + n)]

"""
[Fahsion MNIST Labels]
0 : T-shirt/top 1 : Trouser 2 : Pullover
3 : Dress       4 : Coat    5 : Sandal
6 : Shirt       7 : Sneaker 8 : Bag     
9 : Ankle boot
"""
label_to_name = {
    0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover',
    3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt',
    7: 'Sneaker', 8: 'Bag', 9: 'Ankel boot'
}

# Hyperparameter
data_path = 'data/'
epochs = 5
batch = 100
lr = 0.001
cuda = True

# Load train data
train_x, train_y = load_mnist(data_path, kind='train')
test_x, test_y = load_mnist(data_path, kind='test')
in_channels = 1
out_channels = 64
kernel_size = 5

output_dim = 10
test_num = len(test_x)

train_x = train_x.reshape(-1, 28, 28)
test_x = test_x.reshape(-1, 28, 28)

################### CNN ###################
class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, output_dim):
        super(CNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.output_proj = nn.Linear(7 * 7 * 32, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)

        out = self.conv1(x)

        out = self.conv2(out)

        out = out.reshape(out.size(0), -1)

        out = self.output_proj(out)

        return out

model = CNN(in_channels, out_channels, kernel_size, output_dim)

criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=lr)


if cuda:
    model.cuda()
    train_x = train_x.cuda()
    train_y = train_y.cuda()
    test_x = test_x.cuda()
    test_y = test_y.cuda()

print('='*10 + ' CNN Training Start ' + '='*10)
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

print('\n\n[CNN Test Start]')
test_x, test_y = shuffle(test_x, test_y)
pred = []
for (batch_x, _) in batch_loader((test_x, test_y), batch):
    out = model(batch_x)
    p = torch.argmax(out, -1)
    pred.append(p)

pred = np.concatenate(pred)
num_correct = len(np.where(pred == test_y)[0])
accuracy = num_correct / test_num

print('Accuracy = %.4f (%d / %d)\n\n' % (accuracy, num_correct, test_num))

samples = test_x[:9, :].reshape(9, 28, 28).cpu().numpy()
pred = list(pred[:9])
ans = list(test_y[:9].cpu().numpy())

fig, ax = plt.subplots(3, 3, figsize=(10, 10))

for i in range(9):
    row = i // 3
    col = i % 3
    cur_ax = ax[row, col]
    cur_ax.imshow(samples[i, :, :])
    cur_ax.axes.get_xaxis().set_visible(False)
    cur_ax.axes.get_yaxis().set_visible(False)

    ans_text = 'Truth: %-11s' % label_to_name[ans[i]]
    pred_text = 'Pred: %11s' % label_to_name[pred[i]]

    cur_ax.text(0.3, -0.1, ans_text, fontsize=8, horizontalalignment='center',
                verticalalignment='center', transform=cur_ax.transAxes,
                color='blue')
    cur_ax.text(0.8, -0.1, pred_text, fontsize=8, horizontalalignment='center',
                verticalalignment='center', transform=cur_ax.transAxes,
                color='red')
plt.show()