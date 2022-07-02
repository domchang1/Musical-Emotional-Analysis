import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, inp_dim, out_dim, hidden_dim, n_layers=2):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        
        self.rnn = nn.RNN(self.inp_dim, self.hidden_dim, self.n_layers, nonlinearity='relu', batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, self.out_dim)
        self.softmax = nn.Softmax(-1)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, input):
        h0 = torch.zeros(self.n_layers,self.hidden_dim)
        # input.size(0), 
        out,_ = self.rnn(input, h0)
        linear = self.linear(out)
        predictions = self.softmax(linear)
        return predictions


rnn = RNN(10,4,2)
test = torch.randn(3, 10)
output = rnn(test)
print(output)
print(output.shape)

