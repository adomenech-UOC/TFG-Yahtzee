from agents.Actor import Actor
from agents.BasicAgent import BasicAgent
from agents.Critic import Critic
from agents.DQN import DQN
import trainer
import agents.utils as AgentUtils
import os
import time
import logger as logger

# Control random values
import random
import torch
import numpy as np

SEED = 0
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Time table for BasicAgent:
# n_plays = 200, 1 min
# n_plays = 2000, 5 min
# n_plays = 20000, 1h
# n_plays = 200000, 8h

def train_dqn_model():
    params = {
        "n_plays": 200,   
        "batch_size": 256,
        "buffer_capacity": 10000,
        "lr": 0.0001,
        "epsilon_start": 1.0,
        "epsilon_decay_prop": 0.7,
        "buffer_alpha": 0.7,
        "buffer_beta": 0.7,
        "update_target_every": 2000,
        "hidden_dim": 128,
        "save_agent": True,
        "debug": False,
    }

    start = time.time()

    agent, _ = trainer.train_dqn_agent(BasicAgent, **params)
    
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

    n_plays = 100
    avg_score, median_score = trainer.evaluate_model(agent, n_plays)

    print_results(n_plays, avg_score, median_score)

def train_a2c_model():
    params = {
        "n_plays": 100,   
        "lr_actor": 0.0001,
        "lr_critic": 0.0001,
        "save_agent": True,
        "debug": False,
    }

    start = time.time()

    agent, _, _ = trainer.train_a2c_agent(Actor, Critic, **params)

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

def load_model():
    path = os.path.dirname(os.path.realpath(__file__))
    # path = os.path.join(path, "../models/", "DuelingDQN_2025-04-21_17-03-00")
    path = os.path.join(path, "../models/", "Actor_2025-05-22_20-01-28")
    
    model = AgentUtils.load_agent(path, Actor)
    n_plays = 100
    avg_score, median_score = trainer.evaluate_model(model, n_plays)
    
    print_results(n_plays, avg_score, median_score)

def print_results(n_plays, avg_score, median_score):
    print(f"Evaluation results:")
    print(f"Score over {n_plays} games")
    print(f"Avg: {avg_score:.1f}, Median: {median_score:.1f}")    

def test_a2c():
    params = {
    "n_plays": 5000,   
    "lr_actor": 0.001,
    "lr_critic": 0.001,
    "save_agent": True,
    "debug": False,
    }

    start = time.time()

    agent, _, _, results = trainer.train_a2c_agent(Actor, Critic, **params)

    logger.print_time(start)

    logger.print_train_results(results, agent.name)

    n_plays = 100
    avg_score, median_score, scores = trainer.evaluate_model(agent, n_plays)
    print_results(n_plays, avg_score, median_score)


def test_dqn():
    params = {
        "n_plays": 200,   
        "batch_size": 256,
        "buffer_capacity": 10000,
        "lr": 0.0001,
        "epsilon_start": 1.0,
        "epsilon_decay_prop": 0.7,
        "buffer_alpha": 0.7,
        "buffer_beta": 0.7,
        "update_target_every": 2000,
        "hidden_dim": 128,
        "save_agent": True,
        "debug": False,
    }

    start = time.time()

    agent,  _, _, results = trainer.train_dqn_agent(DQN, **params)
    
    logger.print_time(start)

    logger.print_train_results(results, agent.name)

    n_plays = 100
    avg_score, median_score, scores = trainer.evaluate_model(agent, 100)
    print_results(n_plays, avg_score, median_score)

if __name__ == "__main__":
    test_dqn()
