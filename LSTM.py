import torch
import torch.nn as nn
import CNN
import MLP

class LSTM(nn.Module):
    def __init__(self, inp_dim, out_dim, hidden_dim, n_layers=1):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.max_pool = nn.MaxPool2d(3)
        self.mlp = MLP.MLP(20,7)
        self.Mel = CNN.CNN(34)
        self.Mfcc = CNN.CNN(13)
        self.lstm = nn.LSTM(20, self.hidden_dim, self.n_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2*self.hidden_dim, 64)
        self.cls = nn.Linear(64, self.out_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, input1, input2):
        mel = self.Mel(input1)
        mfcc = self.Mfcc(input2)
        features = torch.cat((mel, mfcc), 1)
        
        h0 = torch.zeros(2*self.n_layers, self.hidden_dim).to(self.device)
        c0 = torch.zeros(2*self.n_layers, self.hidden_dim).to(self.device)
        hidden,_ = self.lstm(features, (h0,c0))
        max = self.max_pool(hidden)
        out = self.mlp(max)
        out = torch.unsqueeze(out, 0)
        # print(out.shape)
        out = torch.max(out, dim=1, keepdim=False)[0]
        linear = nn.functional.relu(self.linear(out))
        logits = self.cls(linear)
        return logits



