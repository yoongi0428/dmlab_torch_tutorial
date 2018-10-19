import torch
import torch.nn as nn
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
Fashion MNIST is a dataset of Zalando's article images.
28 x 28 sized greyscale images with 10 classes
See more details in https://github.com/zalandoresearch/fashion-mnist

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

"""
[Description]
data_path       : directory to save / contains Fashion MNIST data
epochs          : # of epoch to train
batch           : Batch size
lr              : Learning rate
cuda            : Use GPU if True

input_dim       : Input dimensions. (MNIST => 28 * 28 = 784)
hidden_dim      : Hidden Dimension
output_dim      : Dimension of model output (10 classes)
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
hidden_dim = 256
output_dim = 10
test_num = len(test_x)

# MLP Model
################### MLP ###################
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        # 2 layer MLP
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # ReLU
        self.relu = nn.ReLU()

    def forward(self, x):
        # FC - ReLU - FC
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        return out

# Instantiate model
model = MLP(input_dim, hidden_dim, output_dim)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=lr)


if cuda:
    model.cuda()
    train_x = train_x.cuda()
    train_y = train_y.cuda()
    test_x = test_x.cuda()
    test_y = test_y.cuda()

# Training procedure
print('='*10 + ' MLP Training Start ' + '='*10)
for epoch in range(1, epochs + 1):
    loss = 0.0
    for i, (batch_x, batch_y) in enumerate(batch_loader((train_x, train_y), batch)):
        # Clear gradients
        optim. zero_grad()

        # Logit
        out = model(batch_x)

        # Compute batch loss
        l = criterion(out, batch_y)

        # Compute grdients and update weights
        l.backward()
        optim.step()

        loss += l

    print("[Epoch %3d] Loss : %.4f" % (epoch, loss))

# Evaluation
print('\n\n[MLP Test Start]')
test_x, test_y = shuffle(test_x, test_y)
pred = []
for (batch_x, _) in batch_loader((test_x, test_y), batch):
    out = model(batch_x)
    p = torch.argmax(out, -1)
    pred.append(p)

# Accuracy
pred = np.concatenate(pred)
num_correct = len(np.where(pred == test_y)[0])
accuracy = num_correct / test_num

print('Accuracy = %.4f (%d / %d)\n\n' % (accuracy, num_correct, test_num))

# Plot 9 examples with labels
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