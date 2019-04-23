import torch
import random


class Decoder(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Decoder, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.build()

    def build(self):
        self.V = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.lstmCell = torch.nn.LSTMCell(
            input_size=self.n_input+self.n_hidden,
            hidden_size=self.n_hidden)
        self.h2o = torch.nn.Linear(self.n_hidden * 2, self.n_output)

    def step(self, y, h, c):
        # assume y.shape = (B, F)
        # assume h.shape = tuple((B, H))
        # assume c.shape = (B, H)

        inputs = torch.cat((y, c), 1)
        h, _cell = self.lstmCell(inputs, h)
        inputs = torch.cat((h, c), 1)
        y = self.h2o(inputs)

        return y, (h, _cell)

    def forward(self, c, target, temperature):
        # assume c.shape = (layers*directions, batch, hidden)
        B, T, F = target.shape

        outputs = torch.empty((B, T, F))
        h = (self.V(c).tanh(), self.V(c).tanh())
        y = torch.zeros((B, F))

        for t in range(0, T):
            y, h = self.step(y, h, c)
            outputs[:, t, :] = y
            if random.random() < temperature:
                y = target[:, t, :]  # teacher forcing

        return outputs


class Encoder(torch.nn.Module):
    def __init__(self, n_input=1, n_hidden=10, n_layers=1):
        super(Encoder, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.build()

    def build(self):
        # self.bn = torch.nn.BatchNorm1d(config.FRAMESIZE)
        self.V = torch.nn.Linear(self.n_hidden, self.n_hidden)
        self.lstm = torch.nn.LSTM(
            input_size=self.n_input,
            hidden_size=self.n_hidden,
            batch_first=True)

    def forward(self, x, h):
        # assume x.shape = (batch, time, features),
        # import pdb
        # pdb.set_trace()
        _, (h, _) = self.lstm(x, h)
        c = self.V(h).tanh()

        # assume c = layers*directions, batch, hidden
        return c.squeeze(0)


class AE(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers):
        super(AE, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.build()

    def build(self):
        self.encoder = Encoder(self.n_input, self.n_hidden, self.n_layers)
        self.decoder = Decoder(self.n_input, self.n_hidden, self.n_output)

    def forward(self, x, h, temperature, return_encoding=False):
        # assume x.shape = (B, T, F)
        c = self.encoder(x, h)
        reconstructions = self.decoder(c, x, temperature)
        if return_encoding:
            return reconstructions, c
        return reconstructions

    def init_hidden(self, bsz):
        return (
            torch.nn.Parameter(
                torch.randn(self.n_layers, bsz, self.n_hidden) * 0.01,
                requires_grad=True),
            torch.nn.Parameter(
                torch.randn(self.n_layers, bsz, self.n_hidden) * 0.01,
                requires_grad=True)
        )
