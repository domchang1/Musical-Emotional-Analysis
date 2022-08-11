import torch
import torch.nn as nn
import MLP

class LSTM(nn.Module):
    def __init__(self, inp_dim, out_dim, hidden_dim, n_layers=2):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.mlp = MLP.MLP(self.inp_dim, 32)
        self.lstm = nn.LSTM(32, self.hidden_dim, self.n_layers, nonlinearity='relu', batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2*self.hidden_dim, self.out_dim)
        self.sigmoid = nn.Sigmoid()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, input):
        mlp = self.mlp(input)
        h0 = torch.zeros(2*self.n_layers, self.hidden_dim).to(self.device)
        c0 = torch.zeros(2*self.n_layers, self.hidden_dim).to(self.device)
        out,_ = self.lstm(mlp, (h0,c0))
        out = torch.unsqueeze(out, 0)
        # print(out.shape)
        out = torch.max(out, dim=1, keepdim=False)[0]
        linear = self.linear(out)
        predictions = self.sigmoid(linear)
        return predictions



