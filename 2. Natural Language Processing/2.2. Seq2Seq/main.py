import torch
import torch.nn as nn
import numpy as np
from time import strftime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from utils.data_utils import load_data, preprocess, load_vocab, truncate_after_val
from utils.metrics import calculate_bleu
from models.model import Seq2seq

def batch_loader(data, n=1, shuffle=False):
    x = data[0]
    y = data[1]
    data_len = len(x)

    if shuffle:
        perm = np.random.permutation(data_len)
    else:
        perm = np.arange(0, data_len)
    x = x[perm]
    y = y[perm]
    for i in range(0, data_len, n):
        yield x[i: min(data_len, i + n), :], y[i: min(data_len, i + n)]

# Hyperparameters

# - General
embedding_dim = 300
maxlen = 30
shuffle = True

# - Seq2seq
rnn_hidden = 256
num_layers = 1
bi = True
attention = True
attn_type = 'general'   # dot, general, concat
attn_dim = 128  # when concat

# - Training
epochs = 200
batch = 128
lr = 0.001
cuda = torch.cuda.is_available()

# - Attention visualization
show_attn = False
show_ex_num = 123

# Load Data and Build dictionaries
src_train_sent, tar_train_sent = load_data('data/', train=True, small=True)
src_dict, src_cand = load_vocab(src_train_sent)
tar_dict, tar_cand = load_vocab(tar_train_sent)
src_vocab_size = len(src_dict)
tar_vocab_size = len(tar_dict)

src_train, tar_train = preprocess(src_train_sent, tar_train_sent, src_dict, tar_dict, maxlen)

# Build Seq2Seq Model & Loss & Optimizer
model = Seq2seq(embedding_dim, rnn_hidden, num_layers, src_vocab_size, tar_vocab_size, bi, attention, attn_type, attn_dim)

criterion = nn.NLLLoss(ignore_index=3)
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
    print('EPOCH %s starts at %s' % (epoch, strftime('%Y-%m-%d, %H:%M:%S')))

    loss = 0.0
    for i, (batch_src, batch_tar) in enumerate(batch_loader((src_train, tar_train, shuffle), batch)):
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

    print('[Epoch %3d] Loss : %.4f\n' % (epoch, loss))

# Test
src_test_sent, tar_test_sent = load_data('data', test=True, small=True)
src_test, tar_test = preprocess(src_test_sent, tar_test_sent, src_dict, tar_dict, maxlen)

if cuda:
    src_test = src_test.cuda()
    tar_test = tar_test.cuda()

translate = []
attentions = []

# TODO : Make it clear

model.eval()
for i, (batch_src, batch_tar) in enumerate(batch_loader((src_test, tar_test), 5)):
# for i, (batch_src, batch_tar) in enumerate(batch_loader((src_train, tar_train), 5)):  # For debug
    trans_sent, attn = model.translate(batch_src, maxlen=maxlen)
    # attn : batch, tar_seq, src_seq
    translate += trans_sent
    attentions.append(attn)

EOS = tar_dict['<EOS>']
trans = truncate_after_val(translate, EOS)

tar_test = tar_test[:, 1:]
tar_len = torch.sum(tar_test.ne(3), -1)
ref_words = []
for i, (line, l) in enumerate(zip(tar_test, tar_len)):
    line = line[:l.item()].tolist()
    ref_words.append(line)

if attention:
    attentions = np.concatenate(attentions)

tar_re_dict = {tar_dict[w]: w for w in tar_dict}
src_re_dict = {src_dict[w]: w for w in src_dict}
trans_words = [[tar_re_dict[i] for i in s] for s in trans]

def join_str(s_list, dict, SOS=0, EOS=1):
    s_list = [dict[n] for n in s_list]

    start = 1 if '<SOS>' in s_list else 0
    end = s_list.index('<EOS>') if '<EOS>' in s_list else len(s_list)
    trunc_s_list = s_list[start:end]

    return ' '.join(trunc_s_list), s_list

src = src_test.tolist()
src = [[src_re_dict[i] for i in s[1:]] for s in src]
src = truncate_after_val(src, '<EOS>')

tar = ref_words
tar = [[tar_re_dict[i] for i in s] for s in tar]
tar = truncate_after_val(tar, '<EOS>')

pred = trans_words

print('===========Translation Eval============')
for i in range(10):
    print('Source : ', ' '.join(src[i]))
    print('Reference : ', ' '.join(tar[i]))
    print('Translation : ', ' '.join(pred[i]))
    print()

print('Show attention map for %d-th example...' % show_ex_num)
if attention and show_attn:
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions[show_ex_num], cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + src[show_ex_num] + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + pred[show_ex_num] + ['<EOS>'])

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

tar = [[t] for t in tar]
bleu_4 = calculate_bleu(pred, tar)
print('BLEU-4: %.2f' % bleu_4)


print('===========  FINISHED  ============')