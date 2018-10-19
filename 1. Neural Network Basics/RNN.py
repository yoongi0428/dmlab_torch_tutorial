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
input_dim = 28
hidden_dim = 256
output_dim = 10
test_num = len(test_x)

train_x = train_x.reshape(-1, 28, 28)
test_x = test_x.reshape(-1, 28, 28)

################### RNN ###################
class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 1-layer RNN module
        # If batch_first is False, input & output are (seq_len, batch, feature)
        # If batch_first is True, input & output are (batch, seq_len, feature)
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # RNN outputs (output, hidden)
        # (if batch_fist is True)

        # output : Output feature of last layer (batch, seq_len, directions * hidden_dim)
        # hidden : Hidden states for all layers (batch, num_layer * directions, hidden_dim)
        #   - can be separated into hidden.view(num_layers, directions, batch, hidden_dim)

        # For LSTM, hidden is a tuple of (hidden, cell state) with the same shape above

        out, hidden = self.rnn(x)

        last_hidden = out[:, -1, :]

        out = self.output_proj(last_hidden)

        return out

# Instantiate model
model = RNN(input_dim, hidden_dim, output_dim)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=lr)

# Training procedure
if cuda:
    # '.cuda()' method for model and tensor sends the object from CPU to GPU
    model.cuda()
    train_x = train_x.cuda()
    train_y = train_y.cuda()
    test_x = test_x.cuda()
    test_y = test_y.cuda()

print('='*10 + ' RNN Training Start ' + '='*10)
for epoch in range(1, epochs + 1):
    loss = 0.0
    for i, (batch_x, batch_y) in enumerate(batch_loader((train_x, train_y), batch)):
        # clear gradients
        optim. zero_grad()

        # Logit
        out = model(batch_x)

        # Compute batch loss
        l = criterion(out, batch_y)

        # Compute gradients and update weights
        l.backward()
        optim.step()

        loss += l

    print("[Epoch %3d] Loss : %.4f" % (epoch, loss))

# Evaluation
print('\nn[RNN Test Start]')
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