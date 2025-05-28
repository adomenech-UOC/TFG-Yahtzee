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


class Critic(nn.Module):
    def __init__(self, input_size, hidden_dim=64):
        super(Critic, self).__init__()

        self.name = "Critic"

        self.init_params = {
            "input_size": input_size,
            "hidden_dim": hidden_dim,
        }

        self.layer1 = nn.Linear(input_size, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 45)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x


class A2C(nn.Module):
    def __init__(self, input_size, action_size, hidden_dim=64):
        super(A2C, self).__init__()

        self.name = "A2C"

        self.init_params = {
            "input_size": input_size,
            "action_size": action_size,
            "hidden_dim": hidden_dim,
        }

        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size)
        )

        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
