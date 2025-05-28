import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim=256):
        super().__init__()

        self.name = "DQN"

        self.init_params = {
            "state_size": state_size,
            "action_size": action_size,
            "hidden_dim": hidden_dim,
        }

        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size)
        )

    def forward(self, x):
        return self.net(x)
    
    
    