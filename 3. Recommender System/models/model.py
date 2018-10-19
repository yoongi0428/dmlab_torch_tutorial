import torch
import torch.nn as nn

class MF(nn.Module):
    def __init__(self, num_users, num_items, hidden_dim, implicit=False):
        super(MF, self).__init__()

        # Parameters
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.implicit = implicit

        # User and Item matrix
        self.u = nn.Parameter(torch.randn((num_users, hidden_dim)))
        self.i = nn.Parameter(torch.randn((num_items, hidden_dim)))

        if implicit:
            pass

    def forward(self, *inputs):
        # Reconstruct original matrix
        # (num_user, dim) x (dim x num_item) = (num_user, num_item)
        logit = torch.matmul(self.u, self.i.transpose(0, 1))

        if self.implicit:
            pass

        return logit


class AE(nn.Module):
    def __init__(self, num_users, num_items, hidden_dims, implicit=False):
        super(AE, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.implicit = implicit

        # hidden_dims. make as list if it is integer
        hidden_dims = [hidden_dims] if type(hidden_dims) is int else hidden_dims
        self.hidden_dims = hidden_dims

        hiddens = [self.num_items] + hidden_dims
        assert len(hiddens) > 1, ValueError

        # Encoder
        self.encoder = nn.ModuleList([
            nn.Linear(hiddens[i], hiddens[i + 1])
            for i in range(len(hiddens) - 1)
        ])

        # Reverse the list of dimensions
        hiddens = hiddens[::-1]

        # Decoder
        self.decoder = nn.ModuleList([
            nn.Linear(hiddens[i], hiddens[i + 1])
            for i in range(len(hiddens) - 1)
        ])

        self.act = nn.Sigmoid()

        if implicit:
            self.sigmoid = nn.Sigmoid()

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


