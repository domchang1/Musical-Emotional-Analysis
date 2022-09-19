import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            # nn.Linear(240, 120),
            # nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        # self.input_fc = nn.Linear(input_dim, 240)
        # self.hidden_fc = nn.Linear(240, 120)
        # self.out_fc = nn.Linear(120, output_dim)

    def forward(self, input):
        # y_1 = F.reul(self.input_fc(input))
        # y_2 = F.reul(self.hidden_fc(y_1))
        # out = F.reul(self.output_fc(y_2))
        return self.layers(input)


