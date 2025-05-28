import torch
import random
import numpy as np


def select_action(agent, state, valid_actions_mask, epsilon=None):
    """
    Select a valid action for an specific agent.
    """
    valid_indices = [i for i, valid in enumerate(valid_actions_mask) if valid]

    if epsilon is not None and random.random() < epsilon:
        return random.choice(valid_indices), None
    else:
        with torch.no_grad():
            values = agent(state.unsqueeze(0))
            mask_tensor = torch.tensor(valid_actions_mask,
                                       dtype=torch.bool)
            masked_q = values.clone()
            masked_q[0, ~mask_tensor] = -float('inf')
            action = masked_q.argmax().item()

            return action, values


def load_agent(path, agent_class):
    """
    Loads a trained agent from a checkpoint file.
    """
    checkpoint = torch.load(path)
    state_dict = checkpoint['state_dict']

    params = checkpoint['init_params']

    agent = agent_class(**params)
    agent.load_state_dict(state_dict)

    return agent


def load_agent_n_rewards(path, agent_class):
    agent = load_agent(path, agent_class)
    rewards = load_rewards(path + ".rwd")
    return agent, rewards


def save_rewards(path, rewards):
    arr = np.array(rewards)
    np.savetxt(path, arr, fmt="%d")


def load_rewards(path):
    return np.loadtxt(path, dtype=int)
