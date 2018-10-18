import torch
import torch.nn as nn
import numpy as np
from utils.data_utils import load_data, preprocess, load_vocab
from models.model import Seq2seq

def batch_loader(data, n=1):
    x = data[0]
    y = data[1]
    data_len = len(x)

    for i in range(0, data_len, n):
        yield x[i: min(data_len, i + n), :], y[i: min(data_len, i + n)]

# Hyperparameters

# - General
embedding = 32
maxlen = 30

# - Seq2seq
rnn_hidden = 256
num_layers = 1
bi = True
attention = True
attn_type = 'general'   # dot, general, concat
attn_dim = 64  # when concat

# - Training
epochs = 20
batch = 256
lr = 0.005
cuda = True

# Load Data and Build dictionaries
src_train_sent, tar_train_sent = load_data('data/', train=True, small=True)
src_dict, src_cand = load_vocab(src_train_sent)
tar_dict, tar_cand = load_vocab(tar_train_sent)
src_vocab_size = len(src_dict)
tar_vocab_size = len(tar_dict)

src_train, tar_train = preprocess(src_train_sent, tar_train_sent, src_dict, tar_dict, maxlen)

# Build Seq2Seq Model & Loss & Optimizer
model = Seq2seq(embedding, rnn_hidden, num_layers, src_vocab_size, tar_vocab_size, bi, attention, attn_type, attn_dim)

criterion = nn.CrossEntropyLoss(ignore_index=3)
# criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr)

if cuda:
    model.cuda()
    src_train = src_train.cuda()
    tar_train = tar_train.cuda()

# Training
total_batch = np.ceil(len(src_train)/batch)

# Make the model be in training mode (It's not a training function!)
# It is crucial when you use module such as 'Dropout' or 'BatchNorm',
# which behave different at training and testing
model.train()
for epoch in range(1, epochs + 1):
    loss = 0.0
    for i, (batch_src, batch_tar) in enumerate(batch_loader((src_train, tar_train), batch)):
        # 1 ~ (N-1) as decoder input
        # 2 ~ N as expected output
        batch_tar_inp = batch_tar[:, :-1]
        batch_tar = batch_tar[:, 1:]

        # Clear gradient values
        optim.zero_grad()

        # Output (Logits and Attention scores)
        out, score = model(batch_src, batch_tar_inp)

        # Cross Entropy Loss
        l = criterion(out.view(-1, tar_vocab_size), batch_tar.contiguous().view(-1))

        # Compute gradients and pass
        l.backward()

        # Update Weights
        optim.step()

        # Save loss
        loss += l

        if i % 50 == 0:
            print('\tbatch [%3d/%3d], loss : %.4f' % (i, total_batch, l))

    print('[Epoch %3d] Loss : %.4f' % (epoch, loss))

# Test
src_test_sent, tar_test_sent = load_data('data', test=True, small=True)
src_test, tar_test = preprocess(src_test_sent, tar_test_sent, src_dict, tar_dict, maxlen)

if cuda:
    src_test = src_test.cuda()
    tar_test = tar_test.cuda()

translate = []
attentions = []
model.eval()
for i, (batch_src, batch_tar) in enumerate(batch_loader((src_test, tar_test), 5)):
    trans_sent, attn = model.translate(batch_src, maxlen=maxlen)
    # trans_sent, attn = model.translate_batch(batch_src, maxlen=maxlen)

    translate += list(trans_sent.cpu().numpy())
    attentions.append(attn)

    if i >= 10:
        break

tar_re_dict = {tar_dict[w]: w for w in tar_dict}
translate_words = [[tar_re_dict[w] for w in s] for s in translate]

print(translate_words)