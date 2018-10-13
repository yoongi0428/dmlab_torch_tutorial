import torch
import torch.nn as nn


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
        self.out_proj = nn.Linear(rnn_hidden, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        input = self.embedding(input)

        out, hidden = self.rnn(input)
        last_hidden = out[:, -1, :]

        logit = self.out_proj(last_hidden)

        return self.sigmoid(logit)

class CNN(nn.Module):
    pass