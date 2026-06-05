import torch
import torch.nn as nn

class LP(nn.Module):
    def __init__(self, input_dim=64, output_dim=1):
        super(LP, self).__init__()

        self.relu = nn.ReLU(inplace = True)
        
        self.lp = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.lp(x).view(-1, 1, 1, 1)