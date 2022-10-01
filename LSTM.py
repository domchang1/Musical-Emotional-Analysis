import torch
import torch.nn as nn
import torch.nn.functional as F
# import MfccCNN
# import MelCNN
import MLP

class LSTM(nn.Module):
    def __init__(self, inp_dim, out_dim, hidden_dim, n_layers=1):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        # self.max_pool = nn.MaxPool1d(3)
        self.inmlp = MLP.MLP(12, 256)
        self.mlp = MLP.MLP(256,out_dim)
        # self.conv1a = nn.Conv2d(1, 5, (1120, 3), (560, 1)) #7
        # self.conv1b = nn.Conv2d(5, 5, (300, 8), (150, 4))
        self.lstm = nn.LSTM(self.inp_dim, self.hidden_dim, self.n_layers, batch_first=True, bidirectional=True)
        # self.linear = nn.Linear(2*self.hidden_dim, 64)
        # self.cls = nn.Linear(64, self.out_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def forward(self, input):
        input = input[None,:]
        # print(input)
        input = self.inmlp(input)
        # print(input.shape)
        # input = F.relu(self.conv1a(input))
        # print(input.shape)
        # features = input.view(len(input[1]), 5*69)
        # print(mel.shape)
        # print(mfcc.shape)
        # features = torch.unsqueeze(features, 0)
        # print(features.shape)
        h0 = torch.zeros(2*self.n_layers, self.hidden_dim).to(self.device)
        c0 = torch.zeros(2*self.n_layers, self.hidden_dim).to(self.device)
        hidden,_ = self.lstm(input, (h0,c0))
        # print(hidden.shape)
        out = torch.max(hidden, dim=0, keepdim=False)[0]
        # print(out.shape)
        out = self.mlp(out)
        # out = torch.unsqueeze(out, 0)
        # print(out.shape)
        # out = torch.max(out, dim=1, keepdim=False)[0]
        # linear = nn.functional.relu(self.linear(out))
        # logits = self.cls(linear)
        return out



