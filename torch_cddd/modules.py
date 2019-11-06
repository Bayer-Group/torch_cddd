import torch
from torch.nn import GRU, Linear, ReLU
from seq2seq.models.baseRNN import BaseRNN


class StackedDenseNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = list()
        self.input_dim = [input_dim] + hidden_dim[:-1]
        self.output_dim = hidden_dim
        for dim_in, dim_out in zip(self.input_dim, self.output_dim):
            self.layers.append(Linear(dim_in, dim_out))
            self.layers.append(ReLU())
        self.layers.append(Linear(hidden_dim[-1], output_dim))
        self.layers = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class StackedRNN(torch.nn.Module):
    def __init__(self, rnn_cell, input_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = torch.nn.ModuleList()
        for i_dim, h_dim in zip([input_dim] + hidden_size[:-1], hidden_size):
            self.layers.append(rnn_cell(i_dim, h_dim))

    def forward(self, input, hx=None):
        hidden = []
        if hx is not None:
            assert hx.shape[-1] == sum(self.hidden_size)
            hx = torch.split(hx, self.hidden_size)
        else:
            hx = [None for _ in self.layers]

        x = input
        for layer, h0 in zip(self.layers, hx):
            x, h = layer(x, h0)
            hidden.append(h)
        hidden = torch.cat(hidden, dim=-1)
        return x, hidden


class BaseStackedRNN(BaseRNN):
    def __init__(self, vocab_size, max_len, hidden_size, input_dropout_p, dropout_p, rnn_cell):
        if not isinstance(hidden_size, list):
            hidden_size = [hidden_size]
        n_layers = len(hidden_size)
        super().__init__(vocab_size, max_len, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell)
        self.rnn_cell = StackedRNN(self.rnn_cell)


