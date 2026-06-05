import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=1):
        super(MLP, self).__init__()

        self.relu = nn.ReLU(inplace = True)
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(num_features=hidden_dim),
            self.relu,            
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            self.relu,
            
            nn.Linear(hidden_dim // 2, output_dim),
            self.relu
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.network(x).view(-1, 1, 1, 1)