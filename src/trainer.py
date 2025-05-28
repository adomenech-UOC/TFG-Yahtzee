from time import gmtime, strftime
import yahtzee
import agents.utils as AgentUtils
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torchrl.data import PrioritizedReplayBuffer, LazyTensorStorage
import os
from torch.optim.lr_scheduler import LambdaLR
import statistics
from tqdm import tqdm
import logger
from torch.distributions import Categorical

ALL_ACTIONS = yahtzee.generate_all_actions()


def get_epsilon(step, num_episodes, epsilon_decay_prop, epsilon_start, epsilon_end):
    est_total_steps = 52*num_episodes
    epsilon_decay = est_total_steps * epsilon_decay_prop
    return max(epsilon_end, epsilon_start - (step / epsilon_decay)*(epsilon_start - epsilon_end))


def train_dqn_agent(
        agent_class,
        n_plays=10000,
        buffer_capacity=5000,
        batch_size=256,
        gamma=0.99,
        lr=1e-4,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_prop=0.7,  # % of total steps
        update_target_every=1000,
        max_grad_norm=0.5,
        buffer_alpha=0.7,
        buffer_beta=0.9,
        hidden_dim=256,
        save_agent=False,
        debug=True,
):
    """
    Train DQN agent.
    """

    # Setup the logger
    logger.debug = debug

    env = yahtzee.YahtzeeGame()

    # Initialize networks
    input_length = len(env.get_encoded_state())

    dqn = agent_class(state_size=input_length, action_size=len(
        ALL_ACTIONS), hidden_dim=hidden_dim)
    target_dqn = agent_class(state_size=input_length, action_size=len(
        ALL_ACTIONS), hidden_dim=hidden_dim)
    target_dqn.load_state_dict(dqn.state_dict())

    optimizer = optim.Adam(dqn.parameters(), lr=lr)

    # lr decay scheduler: linear decay for 80% of training
    def lr_lambda(step): return max(
        (lr/10) / lr, 1 - step / (n_plays*0.8))
    scheduler = LambdaLR(optimizer, lr_lambda)

    # Initialize the prioritized replay buffer
    storage = LazyTensorStorage(buffer_capacity)
    replay_buffer = PrioritizedReplayBuffer(
        # Controls the degree of prioritization (0 is uniform sampling)
        alpha=buffer_alpha,
        # Controls the amount of importance sampling correction
        beta=buffer_beta,
        storage=storage
    )

    results = []
    total_steps = 0
    for play in tqdm(range(n_plays), disable=debug):
        # init environment with state, done status, and valid action mask
        state_dict = env.reset()
        state = env.get_encoded_state()
        state = torch.FloatTensor(state)
        done = False
        valid_actions_mask = env.get_valid_actions_mask()

        while not done:

            # selection action, eps greedy
            epsilon = get_epsilon(total_steps, n_plays,
                                  epsilon_decay_prop, epsilon_start, epsilon_end)
            action_idx, action_probs = AgentUtils.select_action(
                dqn, state, valid_actions_mask, epsilon)

            # Convert action index to game action
            action = env.index_to_action(action_idx)

            logger.log(f"Dice before step: {env.dice}")

            # Execute action
            rolls_left = env.rolls_left
            next_state, reward, done, _ = env.step(action)

            logger.log(f"Action taken: {action}")
            logger.log(f"Rerolls left: {rolls_left}")

            next_state = torch.FloatTensor(next_state)
            next_valid_mask = env.get_valid_actions_mask()

            # Store experience in replay buffer
            experience = TensorDict({
                'state': state.cpu(),
                'action': torch.tensor(action_idx),
                'reward': torch.tensor(reward),
                'next_state': next_state.cpu(),
                'done': torch.tensor(done),
                'next_valid_mask': torch.tensor(next_valid_mask)
            }, batch_size=[])

            replay_buffer.add(experience)

            state = next_state
            valid_actions_mask = next_valid_mask
            total_steps += 1

            # Sample a batch of experiences and perform training
            if len(replay_buffer) >= batch_size:
                batch, info = replay_buffer.sample(
                    batch_size, return_info=True)

                # Extract experiences
                states = batch['state']
                actions = batch['action']
                rewards = batch['reward']
                next_states = batch['next_state']
                dones = batch['done']
                next_valid_masks = batch['next_valid_mask']

                # Extract sampling weights and indices
                weights = info['_weight'].clone().detach().to(
                    dtype=torch.float32)
                indices = info['index']

                # Compute Q-values and gather the Q-values for taken actions
                q_values = dqn(states)  # Shape: [batch_size, action_dim]
                q_selected = q_values.gather(1, actions.unsqueeze(
                    1)).squeeze(1)  # Shape: [batch_size]

                with torch.no_grad():
                    # Compute target Q-values using the target network
                    # Shape: [batch_size, action_dim]
                    q_next = target_dqn(next_states)

                    # Mask invalid actions in next states
                    invalid_next_actions = ~next_valid_masks.bool()
                    q_next[invalid_next_actions] = -float('inf')

                    # Compute max Q-value for next states
                    max_q_next = q_next.max(dim=1)[0]  # Shape: [batch_size]

                    # Compute target: r + gamma * max(Q') * (1 - done)
                    q_target = rewards + gamma * \
                        max_q_next * (1 - dones.float())

                # Compute loss using Huber loss (Smooth L1)
                loss_function = nn.SmoothL1Loss(reduction='none')
                loss = loss_function(q_selected, q_target)

                # Apply importance sampling weights
                loss = (loss * weights).mean()

                # Compute TD-errors for priority update
                with torch.no_grad():
                    td_errors = torch.abs(q_target - q_selected).cpu()

                # Update priorities in the replay buffer
                replay_buffer.update_priority(indices, td_errors)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()

                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    dqn.parameters(), max_norm=max_grad_norm)
                optimizer.step()
            # Move the optimizer one step
            else:
                optimizer.step()

            # Update target network periodically -- hard update
            if total_steps % update_target_every == 0:
                target_dqn.load_state_dict(dqn.state_dict())

        # update lr
        scheduler.step()

        final_score = sum(v for v in env.categories.values() if v is not None)
        logger.log(f"Final score: {final_score}")
        logger.log(f"End state: {env.get_state()}")

        results.append(final_score)

        # Play statistics
        if (play + 1) % 100 == 0 and debug:
            final_score = sum(
                v for v in env.categories.values() if v is not None)
            final_score += env.upper_bonus + env.yahtzee_bonuses
            print(
                f"Play {play+1}: Training score = {final_score:.1f}, Epsilon = {epsilon:.3f}, lr = {scheduler.get_last_lr()[0]:.7f}")

    # Final save at the end of training
    dir_path = save_model(
        dqn,
        n_plays,
        total_steps,
        results
    ) if save_agent else ""

    return dqn, target_dqn, dir_path, results


def train_a2c_agent(
    actor_class,
    critic_class,
    n_plays=10000,
    gamma=0.99,
    lr_actor=1e-4,
    lr_critic=1e-4,
    save_agent=False,
    debug=True,
):
    """
    Train A2C agent.
    """

    # Setup the logger
    logger.debug = debug

    env = yahtzee.YahtzeeGame()

    # Initialize networks
    input_length = len(env.get_encoded_state())

    actor = actor_class(input_size=input_length, action_size=len(
        ALL_ACTIONS))
    critic = critic_class(input_size=input_length)

    optimizer_actor = optim.AdamW(actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.AdamW(critic.parameters(), lr=lr_critic)

    results = []
    total_steps = 0
    for play in tqdm(range(n_plays), disable=debug):
        # init environment with state, done status, and valid action mask
        state_dict = env.reset()
        state = env.get_encoded_state()
        valid_actions_mask = env.get_valid_actions_mask()

        done = False
        while not done:
            state_tensor = torch.FloatTensor(state)
            # Actor selects action
            action_idx, action_probs = AgentUtils.select_action(
                actor, state_tensor, valid_actions_mask)

            dist = Categorical(action_probs)
            sample = dist.sample()

            logger.log(f"Dice before step: {env.dice}")

            # Convert action index to game action
            action = env.index_to_action(action_idx)

            # Execute action
            rolls_left = env.rolls_left
            next_state, reward, done, _ = env.step(action)

            logger.log(f"Action taken: {action}")
            logger.log(f"Rerolls left: {rolls_left}")

            next_state = torch.FloatTensor(next_state)
            next_valid_mask = env.get_valid_actions_mask()

            # Critic estimates value function
            value = critic(state_tensor)
            next_value = critic(next_state)

            # Calculate TD target and Advantage
            td_target = reward + gamma * next_value * (1 - done)
            advantage = td_target - value

            # Critic update with MSE loss
            optimizer_critic.zero_grad()
            critic_loss = F.mse_loss(value, td_target.detach(),
                                     reduction="none")
            # critic_loss.requires_grad = True
            critic_loss.sum().backward()
            optimizer_critic.step()

            # Actor update
            optimizer_actor.zero_grad()
            log_prob = dist.log_prob(sample)
            actor_loss = torch.sum(-log_prob * advantage.detach())
            actor_loss.requires_grad = True
            actor_loss.sum().backward()
            optimizer_actor.step()

            # Update state, episode return, and step count
            state = next_state
            valid_actions_mask = next_valid_mask
            total_steps += 1

        final_score = sum(v for v in env.categories.values() if v is not None)
        logger.log(f"Final score: {final_score}")
        logger.log(f"End state: {env.get_state()}")

        results.append(final_score)

    # Final save at the end of training
    dir_path = save_model(
        actor,
        n_plays,
        total_steps,
        results
    ) if save_agent else ""

    return actor, critic, dir_path, results


def evaluate_model(model, n_plays=500):
    """ Evaluate the model's performance over multiple plays. """

    # Set model to evaluation mode
    model.eval()

    # Env setup
    env = yahtzee.YahtzeeGame()

    # Scores
    total_scores = 0.0
    scores = []

    # Disable gradient computation
    with torch.no_grad():
        for _ in tqdm(range(n_plays)):
            state_dict = env.reset()
            state = env.get_encoded_state()
            state = torch.FloatTensor(state)
            done = False

            while not done:
                valid_actions_mask = env.get_valid_actions_mask()
                action_idx, _ = AgentUtils.select_action(
                    model, state, valid_actions_mask, epsilon=None)

                # Convert action index to game action
                if action_idx < 32:
                    keep_mask = [bool(int(bit)) for bit in f"{action_idx:05b}"]
                    action = ('reroll', keep_mask)
                else:
                    cat_idx = action_idx - 32
                    category = list(env.categories.keys())[cat_idx]
                    action = ('score', category)

                # Execute action
                next_state, reward, done, _ = env.step(action)
                state = torch.FloatTensor(next_state)

            # Calculate final score
            final_score = sum(
                v for v in env.categories.values() if v is not None)
            final_score += env.upper_bonus + env.yahtzee_bonuses
            total_scores += final_score
            scores.append(final_score)

    avg_score = total_scores / n_plays
    median_score = statistics.median(scores)

    return avg_score, median_score, scores


def save_model(
    agent,
    play,
    total_steps,
    rewards
):
    data = {
        'state_dict': agent.state_dict(),
        'init_params': agent.init_params,
        'play': play,
        'total_steps': total_steps
    }

    model_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(model_path, "../models/")
    model_path = os.path.join(model_path, agent.name + "_" +
                              strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + ".rl")

    torch.save(data, model_path)
    print(f"Checkpoint saved to {model_path}")

    reward_path = model_path + ".rwd"

    AgentUtils.save_rewards(reward_path, rewards)

    return model_path
