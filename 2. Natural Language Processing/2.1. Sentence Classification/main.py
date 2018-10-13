import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.dataset import Dataset
from models.model import RNN, CNN

def batch_loader(Dataset, n=1):
    data_len = len(Dataset)

    for i in range(0, data_len, n):
        review, rating = Dataset[i: min(data_len, i + n)]
        yield review, rating
        # yield torch.from_numpy(review).type(torch.LongTensor), torch.from_numpy(rating).type(torch.FloatTensor)



# Hyperparameters
### DATA
embedding = 128
output_dim = 1
maxlen = 70
eumjeol = True

### RNN
rnn_hidden = 256
num_layers = 1
bi = False

### CNN

### Training
epochs = 100
batch = 256
lr = 0.01
gpu = True

model = "rnn"   # 'cnn'

# Dataset
training_data = Dataset("data/ratings_train.txt", maxlen, eumjeol)
vocab_size = training_data.vocab_size()

# Model
model = RNN(embedding, rnn_hidden, num_layers, bi, output_dim, vocab_size) if model == 'rnn' else CNN()
criterion = nn.CrossEntropyLoss()

optim = torch.optim.SGD(model.parameters(), lr)

if gpu:
    model.cuda()
    training_data.cuda()

# Train
model.train()
for epoch in range(1, epochs + 1):
    loss = 0.0

    for i, (batch_x, batch_y) in enumerate(batch_loader(training_data, batch)):
        optim.zero_grad()

        logit = model(batch_x)

        # l = criterion(logit, batch_y)
        l = F.binary_cross_entropy(logit, batch_y)

        l.backward()
        optim.step()

        loss += l.item()

    print('[Epoch %3d] Loss = %.4f' % (epoch, loss))

# Test