import torch
import torch.nn as nn
import MLP

class RNN(nn.Module):
    def __init__(self, inp_dim, out_dim, hidden_dim, n_layers=2):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.mlp = MLP.MLP(self.inp_dim, 200)
        self.rnn = nn.RNN(200, self.hidden_dim, self.n_layers, nonlinearity='relu', batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, self.out_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input):
        mlp = self.mlp(input)
        h0 = torch.zeros(self.n_layers,self.hidden_dim)
        # input.size(0), 
        out,_ = self.rnn(mlp, h0)
        out = torch.unsqueeze(out, 0)
        print(out.shape)
        out = torch.max(out, dim=1, keepdim=False)[0]
        linear = self.linear(out)
        predictions = self.sigmoid(linear)
        return predictions



