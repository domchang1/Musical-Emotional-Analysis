import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, inp_dim, out_dim, hidden_dim, n_layers=2):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        
        self.rnn = nn.RNN(self.inp_dim, self.hidden_dim, self.n_layers, nonlinearity='relu')
        self.linear = nn.Linear(self.hidden_dim, self.out_dim)
    
    def forward(self, input):
        out = self.rnn(input)
        out = self.linear(out)

