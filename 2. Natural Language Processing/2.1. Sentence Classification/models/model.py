import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, emb_dim, rnn_hidden, num_layers, bi, output_dim, vocab_size):
        super(RNN, self).__init__()
        self.emb_dim = emb_dim
        self.rnn_hidden = rnn_hidden
        self.num_layers = num_layers
        self.bi = bi
        self.output_dim = output_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, self.emb_dim)

        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=rnn_hidden,
            num_layers=num_layers,
            bidirectional=bi,
            batch_first=True
        )
        self.out_proj = nn.Linear(rnn_hidden * 2 if bi else rnn_hidden, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        input = self.embedding(input)

        out, hidden = self.rnn(input)
        last_hidden = out[:, -1, :]

        logit = self.out_proj(last_hidden)

        return self.sigmoid(logit)

class CNN(nn.Module):
    def __init__(self, filters, num_filters, maxlen, vocab_size, emb_dim, output_dim):
        super(CNN, self).__init__()

        self.filters = filters
        self.num_filters = num_filters
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.output_dim = output_dim

        self.embedding = nn.Embedding(vocab_size, self.emb_dim)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num, kernel_size=(size, emb_dim), padding=(size//2, 0))
            for size, num in zip(filters, num_filters)
        ])

        self.maxpool = nn.ModuleList([
            nn.MaxPool2d((1, num), stride=1)
            for num in num_filters
        ])

        self.out_proj = nn.Linear(maxlen * len(filters), output_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        input = self.embedding(input)
        input = input.unsqueeze(1)

        conv_out = []
        for conv, mp in zip(self.convs, self.maxpool):
            out = self.relu(conv(input))
            out = out.permute(0, 3, 2, 1)

            pooled = mp(out).squeeze()

            conv_out.append(pooled)

        # concat
        conv_out = torch.cat(conv_out, 1)

        # proj
        logit = self.out_proj(conv_out)

        return self.sigmoid(logit)