# 🎮 Atari Breakout AI using Deep Q-Learning (DQN) and CNNs

This repository contains a Deep Q-Network (DQN) implementation utilizing a Convolutional Neural Network (CNN) to play the classic Atari game **Breakout** (`BreakoutNoFrameskip-v4`) using raw screen pixels as input.

## 📝 Project Overview

* **Task**: Train an RL agent to control a paddle, bounce a ball, and break bricks to maximize the game score.
* **State Representation**: 4 stacked preprocessed frames ($84 \times 84 \times 4$ grayscale images) to capture motion.
* **Algorithm**: Deep Q-Learning (DQN) with Experience Replay and a Target Network.
* **Implementation**: Built with PyTorch, Gymnasium (ALE), and **Stable-Baselines3**.

## 📊 Hyperparameters

| Hyperparameter | Value | Purpose |
|---|---|---|
| `learning_rate` | 1e-4 | Optimisation step size |
| `buffer_size` | 100,000 | Experience replay capacity |
| `batch_size` | 32 | Gradient update sample size |
| `gamma` | 0.99 | Future reward discount factor |
| `target_update_interval` | 1,000 | Target network sync frequency |
| `exploration_fraction` | 0.20 | Decay period for epsilon-greedy |

## 📈 Results

After training for **200,000 agent steps** (~1 hour on CPU, significantly faster on GPU):

* **Random Baseline**: ~1.20 average reward (best score of 8).
* **Trained DQN Agent**: **21.80 average reward** (best score of 31) — representing an **18.2x improvement** over random actions.

## 📁 Deliverables & Structure

* `breakout_dqn_model.zip`: The saved Stable-Baselines3 model.
* `random_agent_gameplay.gif`: Gameplay recording of a random baseline player.
* `trained_agent_gameplay.gif`: Gameplay recording of our trained DQN agent.
* `training_rewards.png`: Visual charts showing training rewards, epsilon decay, and loss.
* `performance_comparison.png`: Comparison analysis charts.

## 🚀 How to Run

1. Open the project in Google Colab.
2. (Optional) Enable GPU: **Runtime -> Change runtime type -> GPU**.
3. Execute the notebook cells sequentially via **Runtime -> Run all**.
