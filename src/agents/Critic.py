import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, input_size, hidden_dim=64):
        super(Critic, self).__init__()

        self.name = "Critic"

        self.init_params = {
            "input_size": input_size,
            "hidden_dim": hidden_dim,
        }

        self.layer1 = nn.Linear(input_size, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x