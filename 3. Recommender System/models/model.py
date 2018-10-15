import torch
import torch.nn as nn

class MF(nn.Module):
    def __init__(self, num_users, num_items, hidden_dim, implicit=False):
        super(MF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.implicit = implicit

        self.u = nn.Parameter(torch.randn((num_users, hidden_dim)))
        self.i = nn.Parameter(torch.randn((num_items, hidden_dim)))

        if implicit:
            pass
            # self.sigmoid = nn.Sigmoid()

    def forward(self, *inputs):
        logit = torch.matmul(self.u, self.i.transpose(0, 1))

        if self.implicit:
            pass
            # logit = self.sigmoid(logit)

        return logit


class AE(nn.Module):
    def __init__(self, num_users, num_items, hidden_dims, implicit=False):
        super(AE, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.implicit = implicit

        hidden_dims = [hidden_dims] if type(hidden_dims) is int else hidden_dims
        self.hidden_dims = hidden_dims

        hiddens = [self.num_items] + hidden_dims
        assert len(hiddens) > 1, ValueError

        self.encoder = nn.ModuleList([
            nn.Linear(hiddens[i], hiddens[i + 1])
            for i in range(len(hiddens) - 1)
        ])

        hiddens = hiddens[::-1]

        self.decoder = nn.ModuleList([
            nn.Linear(hiddens[i], hiddens[i + 1])
            for i in range(len(hiddens) - 1)
        ])

        self.act = nn.Sigmoid()

        if implicit:
            self.sigmoid = nn.Tanh()

    def forward(self, input):
        enc = input
        for layer in self.encoder:
            enc = layer(enc)
            enc = self.act(enc)

        dec = enc
        for layer in self.decoder:
            dec = layer(dec)
            dec = self.act(dec)

        if self.implicit:
            dec = self.sigmoid(dec)

        return dec


if __name__ == '__main__':
    mf = MF(10, 100, 5)
    ae = AE(10, 100, [100, 80, 70])


