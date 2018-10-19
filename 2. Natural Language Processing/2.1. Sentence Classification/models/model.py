import torch
import torch.nn as nn
import torch.nn.functional as F

# RNN Model
class RNN(nn.Module):
    def __init__(self, emb_dim, rnn_hidden, num_layers, bi, output_dim, vocab_size):
        super(RNN, self).__init__()
        # RNN Parameters
        self.emb_dim = emb_dim
        self.rnn_hidden = rnn_hidden
        self.num_layers = num_layers
        self.bi = bi
        self.output_dim = output_dim
        self.vocab_size = vocab_size

        # Embedding matrix + Lookup operation
        self.embedding = nn.Embedding(vocab_size, self.emb_dim)

        # Define RNN module
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=rnn_hidden,
            num_layers=num_layers,
            bidirectional=bi,
            batch_first=True
        )
        # Linear layer for output
        self.out_proj = nn.Linear(rnn_hidden * 2 if bi else rnn_hidden, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # Embedding lookup
        # (batch, seq_len) => (batch, seq_len, embedding)
        input = self.embedding(input)

        # Run RNN and get last hidden state
        out, hidden = self.rnn(input)
        last_hidden = out[:, -1, :]

        # Mapping into output space
        logit = self.out_proj(last_hidden)

        # Apply sigmoid and return
        return self.sigmoid(logit)

class CNN(nn.Module):
    def __init__(self, filters, num_filters, maxlen, vocab_size, emb_dim, output_dim):
        super(CNN, self).__init__()

        # CNN Parameters
        self.filters = filters
        self.num_filters = num_filters
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.output_dim = output_dim

        # Embedding matrix + Lookup operation
        self.embedding = nn.Embedding(vocab_size, self.emb_dim)

        # - nn.ModuleList : Literally, a list of Modules.
        # However, it is different from python list.
        # "model.parameters()" gathers parameters of layers in nn.ModuleList, while doesn't those in python list.
        # Weights in python list of modules will not be updated during training.
        # - For simplicity, we added padding of (size//2, 0) to keep input shape same
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num, kernel_size=(size, emb_dim), padding=(size//2, 0))
            for size, num in zip(filters, num_filters)
        ])

        # For natural language, we apply max-pooling over time
        # That is, when the output of convolution is (seq_len, features),
        # we max-pool most significant feature per word by applying (1, features) max-pool.
        self.maxpool = nn.ModuleList([
            nn.MaxPool2d((1, num), stride=1)
            for num in num_filters
        ])

        # Output layer
        self.out_proj = nn.Linear(maxlen * len(filters), output_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # Embedding lookup
        input = self.embedding(input)

        # Add extra dimension for convolution
        input = input.unsqueeze(1)

        # Gather convolution outputs
        conv_out = []
        for conv, mp in zip(self.convs, self.maxpool):
            # Convolution + ReLU
            out = self.relu(conv(input))

            # (batch, features, seq, 1) => (batch, 1, seq, features)
            out = out.permute(0, 3, 2, 1)

            # mp => (batch, 1, seq, 1)
            # "squeeze" delete all axis of 1 dimensions
            # After squeeze =? (batch, seq)
            pooled = mp(out).squeeze()

            conv_out.append(pooled)

        # Concat all outputs
        # (batch, seq_len * num of filters)
        conv_out = torch.cat(conv_out, 1)

        # Output layer
        logit = self.out_proj(conv_out)

        return self.sigmoid(logit)