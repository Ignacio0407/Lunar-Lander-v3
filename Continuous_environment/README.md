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
    - [Prerequisites Locally](#prerequisites-locally)
    - [Prerequisites Kaggle](#prerequisites-on-kaggle)
  - [Usage](#usage)
    - [Training the Model](#training-the-model)
    - [Running Inference](#running-inference)
  - [Hyperparameters](#general-hyperparameters)
  - [Training Details](#training-details)
  - [LunarLander-v3 Environment](#lunarlander-v3-environment)
    - [Action Space](#action-space)
    - [Observation Space](#observation-space)
    - [Rewards \& Penalties](#rewards--penalties)
    - [Episode Termination](#episode-termination)
  - [Conclusions](#conclusions)
  - [References](#references)

## Introduction

The goal of this project is to train an agent using the DDQN algorithm with PER to make a car learn how to drive, fosucing on not exitting the track by difficult turns. The environment is a classic control problem based on Box2D physics, where the agent must optimize its actions to achieve a stable landing.

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
# If it still gives error with this is because swig was not correctly installed in the requirements, just write
pip install swig
pip install gymnasium[others]
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

- **Discrete(5)**:
  - `0`: Do nothing
  - `1`: Steer right
  - `2`: Steer left
  - `3`: gas
  - `4`: break

### Observation Space

- **Box(0, 255, (96, 96, 3), uint8)**: Contains current image of the environment.

### Rewards & Penalties

- Reward increases for smooth and in-track driving.
The reward is -0.1 every frame and +1000/N for every track tile visited, where N is the total number of tiles visited in the track. For example, if you have finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points.

### Episode Termination
- All the tiles are visited. 
- The car can also go outside the playfield - that is, far off the track, in which case it will receive -100 reward and die.

For more details, see the official Gymnasium documentation.

## General Hyperparameters

- **Algorithm:** DDQN
- **Neural Network:** Input:(4, 84, 84) -> Conv1:(32, 20, 20) -> Conv2:(64, 9, 9) -> Conv3:(64, 7, 7) -> Flatten:(3136, 512) -> Last layer (512, num_actions=5)
- **Episodes:** Depends
- **Learning Rate:** 0.0001
- **Discount Factor (Gamma):** 0.99
- **Replay Buffer Size:** 10,000
- **Batch Size:** 256
- **Target Update:** Soft Updates (Polyak Averaging), TAU = 0.005
- **Exploration Strategy:** ε-greedy (ε decays from 1.0 to 0.01 with a 0.99 decay factor per episode)
- **Optimizer:** AdamW
- **Loss Function:** Smooth L1 loss
- **Early Stopping Patience:** Depends
- **Early Stopping Threshold:** Depends

## Training Details

- **Convergence Reward:** 850+ (environment considered solved)
- **Hardware:** Kaggle gpu p100 16gb and Nvidia a100 40gb
- **Improvements:** Applied in-place gradient normalization (up to 10) to stabilize training and not altering gradients directions, unlike what happens with clipping.

## Experiments

## References

- OpenAI Gymnasium: [LunarLander-v3 Documentation](https://gymnasium.farama.org/environments/box2d/car_racing/)
- Playing Atari with Deep Reinforcement Learning [Google DeepMind](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
- Deep Q-Network (DQN) Paper Playing Atari Games: [Mnih et al., 2013](https://arxiv.org/abs/1312.5602)
- Pytorch DQN Implementation: [Reinforcement Learning (DQN) Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#reinforcement-learning-dqn-tutorial)
- CNNs and Deep Q-Learning:(CS234 2022 standford University Lecture 6)
- Prioritized Experience Replay: [Google DeepMind](https://arxiv.org/pdf/1511.05952)
---