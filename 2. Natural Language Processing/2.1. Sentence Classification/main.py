import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.data_utils import load_data, preprocess, load_vocab
from models.model import RNN, CNN
import numpy as np

# Generator function to yield batch data
def batch_loader(data, n=1):
    x = data[0]
    y = data[1]
    data_len = len(x)

    for i in range(0, data_len, n):
        yield x[i: min(data_len, i + n), :], y[i: min(data_len, i + n)]

# Compute accuracy
def accuracy(pred, ans, threshold=0.5):
    total = len(ans)

    pred = np.concatenate(pred)
    pred[pred > threshold] = 1
    pred[pred <= threshold] = 0

    same = np.array(pred == ans, dtype=int)

    return float(sum(same)) / total

# Hyperparameters

# - General
embedding = 128
output_dim = 1
maxlen = 150
threshold = 0.5

# - RNN
rnn_hidden = 256
num_layers = 1
bi = True

# - CNN
# Kernel sizes and corresponding numbers of filters
# Two lengths must be same

filters = [3, 5, 7]
num_filters = [128, 128, 128]
assert len(filters) == len(num_filters)

# - Training
epochs = 10
batch = 256
lr = 0.001
cuda = True

model = "cnn"   # 'cnn' or 'rnn'

# Load vocabulary and make dictionary
vocabs = load_vocab('data/imdb/imdb.vocab')
w2i = {w: i for i, w in enumerate(vocabs)}
i2w = {i: w for i, w in enumerate(vocabs)}
vocab_size = len(vocabs)

# Load Data
train_x, train_y = load_data('data/', train=True)
train_x, train_y = preprocess(train_x, train_y, w2i, maxlen)

# Build Model & Loss & Optimizer
model = RNN(embedding, rnn_hidden, num_layers, bi, output_dim, vocab_size) \
    if model == 'rnn' else CNN(filters, num_filters, maxlen, vocab_size, embedding, output_dim)

# Loss function & Optimizer
criterion = nn.BCELoss()
optim = torch.optim.Adam(model.parameters(), lr)

if cuda:
    model.cuda()
    train_x = train_x.cuda()
    train_y = train_y.cuda()

# Training procedure
# model.train() makes model be in training mode. (It is not a real training function)
# It is crucial to modules such as batch norm or dropout, which acts different when train or test
model.train()
for epoch in range(1, epochs + 1):
    loss = 0.0

    for i, (batch_x, batch_y) in enumerate(batch_loader((train_x, train_y), batch)):
        # Clear gradient values
        optim.zero_grad()

        # Output
        logit = model(batch_x)

        # Binary Cross Entropy Loss
        # l = F.binary_cross_entropy(logit, batch_y)
        l = criterion(logit, batch_y)

        # Compute gradients and pass
        l.backward()

        # Update Weights
        optim.step()

        # Save loss
        loss += l.item()

    print('[Epoch %3d] Loss = %.4f' % (epoch, loss))

# Test
test_x, test_y = load_data('data/', test=True)
test_x, test_y = preprocess(test_x, test_y, w2i, maxlen)
if cuda:
    test_x = test_x.cuda()
    test_y = test_y.cuda()


model.eval()
pred = []
for i, (batch_x, batch_y) in enumerate(batch_loader((test_x, test_y), batch)):
    # Inference
    out = model(batch_x)

    pred.append(out.data.cpu().numpy())

# Evaluate performance
acc = accuracy(pred, test_y, threshold)
print('\nTEST ACCURACY : %.4f' % acc)