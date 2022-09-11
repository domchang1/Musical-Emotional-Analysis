import torch
import torch.nn as nn
import MLP

class LSTM(nn.Module):
    def __init__(self, inp_dim, out_dim, hidden_dim, n_layers=1):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.mlp = MLP.MLP(self.inp_dim, 256)
        self.lstm = nn.LSTM(256, self.hidden_dim, self.n_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2*self.hidden_dim, 64)
        self.cls = nn.Linear(64, self.out_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, input):
        mlp = self.mlp(input)
        h0 = torch.zeros(2*self.n_layers, self.hidden_dim).to(self.device)
        c0 = torch.zeros(2*self.n_layers, self.hidden_dim).to(self.device)
        out,_ = self.lstm(mlp, (h0,c0))
        out = torch.unsqueeze(out, 0)
        # print(out.shape)
        out = torch.max(out, dim=1, keepdim=False)[0]
        linear = nn.functional.relu(self.linear(out))
        logits = self.cls(linear)
        return logits



