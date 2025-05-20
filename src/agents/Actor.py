import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, input_size, action_size, hidden_dim=64):
        super(Actor, self).__init__()

        self.name = "Actor"

        self.init_params = {
            "input_size": input_size,
            "action_size": action_size,
            "hidden_dim": hidden_dim,
        }

        self.layer1 = nn.Linear(input_size, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, action_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.softmax(self.layer2(x), dim=-1)
        return x