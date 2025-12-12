# Deep Q-Network (DQN) for LunarLander-v3

This project trains a Deep Q-Network (DQN) using PyTorch to solve the "LunarLander-v3" environment from Gymnasium.

<p align="center">
  <img src="docs\media\lunar_lander_trained.gif" alt="Your GIF">
</p>

## Table of Contents

- [Deep Q-Network (DQN) for LunarLander-v3](#deep-q-network-dqn-for-lunarlander-v3)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
  - [Usage](#usage)
    - [Training the Model](#training-the-model)
    - [Running Inference](#running-inference)
  - [Hyperparameters](#hyperparameters)
  - [Training Details](#training-details)
  - [Results](#results)
    - [Trained Model Performance](#trained-model-performance)
  - [LunarLander-v3 Environment](#lunarlander-v3-environment)
    - [Action Space](#action-space)
    - [Observation Space](#observation-space)
    - [Rewards \& Penalties](#rewards--penalties)
    - [Episode Termination](#episode-termination)
  - [References](#references)

## Introduction

The goal of this project is to train an agent using the DQN algorithm to land a spacecraft safely on a designated landing pad in the "LunarLander-v3" environment. The environment is a classic control problem based on Box2D physics, where the agent must optimize its thrusters to achieve a stable landing.

## Installation

### Prerequisites locally

Ensure you have Python installed. You can set up the required dependencies using:

```bash
# Clone the repository
git clone https://github.com/Ignacio0407/Lunar-Lander-v3
cd Lunar-Lander-v3

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128 # Most modern version on November 2025 with cuda 12.8. I could not fit this command in the requirements file, since it doesn't allow for "complex" commands like this one.
```

### Prerequisites on Kaggle

```bash
!git clone https://github.com/Ignacio0407/Lunar-Lander-v3.git
%cd /kaggle/working/Lunar-Lander-v3
!pip install -r requirements-kaggle.txt
# Depending the environment you want to choose.
%cd Discrete_environment
%cd Continuous_environment
```

## Usage

### Training the Model

To train the model, enter any of the folders with python code and then run:

```bash
python main.py
```

### Running Inference

To test the trained model and visualize its performance enter any of the folders with python code and then run:

```bash
python dqn_inference.py
```

In Kaggle, put ! at the beginning for both commands, but notice that the inference will not run graphically, since Kaggle does not allow it.

## LunarLander-v3 Environment

The LunarLander-v3 environment is a reinforcement learning task where an agent controls a lander to safely land on a designated pad. The agent receives rewards based on its landing accuracy and penalties for fuel usage and crashes.

### Action Space

- **Discrete(4)**:
  - `0`: Do nothing
  - `1`: Fire left orientation engine
  - `2`: Fire main engine
  - `3`: Fire right orientation engine

### Observation Space

- **Box(8, float32)**: Contains position, velocity, angle, angular velocity, and landing leg contact status. These are the number of observations.

### Rewards & Penalties

- Reward increases for smooth and centered landing.
- Penalties for excessive tilt, high speed, and unnecessary thruster use.
- +10 points per leg in contact with the ground, +100 for a successful landing, -1000 for crashing (I've hugely increased the penalty for crashing since I prefer for the lander to take more time and land safely rather than crashing).

### Episode Termination

- The lander crashes.
- The lander moves out of bounds.
- The lander stops moving.

For more details, see the official Gymnasium documentation.

## General Hyperparameters

- **Algorithm:** DQN
- **Neural Network:** 8 -> 128 -> 128 -> number_of_actions
- **Episodes:** 600
- **Learning Rate:** 0.0001
- **Discount Factor (Gamma):** 0.99
- **Replay Buffer Size:** 10,000
- **Batch Size:** 128
- **Target Update:** Soft Updates (Polyak Averaging), TAU = 0.005
- **Exploration Strategy:** ε-greedy (ε decays from 1.0 to 0.01 with a 0.995 decay factor per episode)
- **Optimizer:** AdamW
- **Loss Function:** Smooth L1 loss
- **Early Stopping Patience:** 20 (when it reaches 0, the algorithm is stopped)
- **Early Stopping Threshold:** 10 (minimum reward upgrade to not decrease patience. If reward increases in average of the last 100 episodes less than threshold, then patience is reduced by 1)

## Training Details

- **Episodes Trained:** 600, with early stopping implementation to prevent waste of computational resources.
- **Convergence Reward:** 200+ (environment considered solved)
- **Reward shaping:** Can be seen between lines 87-102. I added some rewards and penalties of my own to make the model converge faster.
- **Hardware:** GPU RTX 3060 laptop (6GB VRAM) and Kaggle /Google colab servers
- **Challenges & Improvements:** Applied in-place gradient clipping (±100) to stabilize training.

## Experiments

The modified hyperparameters are the neurons per layer, the depth of the NN, the learning rate and the epsilon (of the greedy policy)

### Discrete actions

#### 32 neurons per layer
None of the models with this NN managed to land successfully even a single time. They were trained with
- General hyperparameters, with a patience of 60
- Modified hyperparameters, not worth mentioning since the model is really bad.

#### 64 neurons per layer
There are four versions of this setup. Two with 2 hidden layers, one with 1 and another one with 4.
- 1 hidden layer: It is a really bad model, never lands properly, showing that more depth is needed.
- 2 Hidden layers: patience of 40. It gave some nice results but many bad ones. I can't show it because it was the first one trained in Kaggle and I lost it. Early stopping triggered in episode 326.
- 2 Hidden layers with hyperparameters modified: Taking into account its small size, it is a good model, succesfully completing the environment in most runs and achieving not too bad results when it does not. NEED TO KNOW THEM
- 4 hidden layer: It is a really bad model, most times it seems like it is going to land correctly but is going too fast and the legs opened and crashed, what happened to the models before I made the reward shaping, showing that not the bigger the NN the better solution it will give. This is mostly due to the fact that big NN tend to stick to what they discover and do not explore as much as little ones, so it likely got stuck in a local maximum.

#### 128 neurons per layer
128_1, 128_2 and 128_best. They were all trained under the same conditions, with general hyperparameters. The only difference was that they received slightly different reward shaping, since the landed condition was respectively for the three models:
- landed = (abs(pos_x) < 0.3 and pos_y < 0.3 and abs(vel_x) < 0.2 and abs(vel_y) < 0.2 and (leg1 == 1 and leg2 == 1))
- landed = (abs(pos_x) < 0.1 and pos_y < 0.1 and abs(vel_x) < 0.1 and abs(vel_y) < 0.1 and (leg1 == 1 or leg2 == 1))
- landed = (abs(pos_x) < 0.2 and pos_y < 0.2 and abs(vel_x) < 0.07 and abs(vel_y) < 0.03 and (leg1 == 1 or leg2 == 1) and abs(angle) < 0.2). Due to the improvement of the reward shaping, the early stopping triggered in episode 376 for 128_best.

128_hyperparameters_change: Stopped training in episode 325.

#### 256 neurons per layer
Early stopping triggered in episode 324. The results are pretty similar to the one that model "best_20_nov" gives, meaning the task is so simple it does not require more neurons.

#### Wind models
All wind models were trained with 

## References

- OpenAI Gymnasium: [LunarLander-v3 Documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- Playing Atari with Deep Reinforcement Learning [Google DeepMind](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
- Deep Q-Network (DQN) Paper Playing Atari Games: [Mnih et al., 2013](https://arxiv.org/abs/1312.5602)
- Pytorch DQN Implementation: [Reinforcement Learning (DQN) Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#reinforcement-learning-dqn-tutorial)

---