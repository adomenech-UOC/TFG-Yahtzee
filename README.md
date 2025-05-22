# TFG-Yahtzee

This project aims to address the development and subsequent evaluation of an artificial intelligence that is capable of playing the Yahtzee game, the AI ​​will be implemented using Reinforcement Learning (RL) techniques.

Reinforcement learning in games has been fundamental in the development of algorithms that have then been applied to more complex problems, in this study we take Yahtzee as a starting point in order to offer new knowledge about AI models and the adaptability of these models, on the other hand we also want to make a comparison with human strategies that allow us to give a new understanding of the limitations or advantages of machine learning.

Yahtzee is a game that requires sequential decisions with a high degree of uncertainty, since the results of the dice are random in each move, therefore it will be necessary to formalize a model for the game that can give answers based on a Markov Decision Model (MDP).

Once the game is formalized, the best algorithm to play the game will have to be chosen, so some of the algorithms to explore will be; Q-learning, Deep Q-Networks (DQN), or PPO (Proximal Policy Optimization). During this process, the AI ​​and the different responses it gives to the situations it finds itself in will have to be observed and the results will have to be evaluated in comparison to the most optimal move.

In conclusion, the work seeks to establish a basis for the development of an AI capable of choosing the best decision given some previous parameters and as a second instance to test its possible adaptability in other games.

## Contents

This repository contains all the tests made to achieve the goal of testing different algorithms using the Game Yahtzee as an eviroment and extract conclusions about their behaviour.

## Setup the repository

Run the following command ot install all the dependencies:

```
pip install -r requirements
```