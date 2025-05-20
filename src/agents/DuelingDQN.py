import torch.nn as nn


class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim=256):
        super(DuelingDQN, self).__init__()

        self.name = "DuelingDQN"

        self.init_params = {
            "state_size": state_size,
            "action_size": action_size,
            "hidden_dim": hidden_dim,
        }

        # Shared feature extraction
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.value_stream = nn.Linear(hidden_dim, 1)
        self.advantage_stream = nn.Linear(hidden_dim, action_size)

    def forward(self, x):
        
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        advantages_mean = advantages.mean(dim=1, keepdim=True)
        Q = values + (advantages - advantages_mean)

        return Q
