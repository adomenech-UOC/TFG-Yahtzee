import torch.nn as nn
import torch.nn.functional as F


class BasicAgent(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim=256):
        super(BasicAgent, self).__init__()

        self.name = "BasicAgent"

        self.init_params = {
            "state_size": state_size,
            "action_size": action_size,
            "hidden_dim": hidden_dim,
        }

        self.state_size = state_size
        self.action_size = action_size
        self.layer1 = nn.Linear(self.state_size, 128)
        self.layer2 = nn.Linear(128, self.action_size)

    def forward(self, state):
        output = F.relu(self.layer1(state))
        output = self.layer2(output)
        output = F.softmax(output, dim=1)

        return output