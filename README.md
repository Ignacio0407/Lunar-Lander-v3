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
  - [References](#references)

## Introduction

The goal of this project is to train an agent using the DQN and DDQN algorithm to land a spacecraft safely on a designated landing pad in the "LunarLander-v3" environment and another agent on "CarRacing-v3" gymnasium environment, which focuses on the car not getting off track.

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
pip install gymnasium[other]
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

## References

- Playing Atari with Deep Reinforcement Learning [Google DeepMind](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
- Deep Q-Network (DQN) Paper Playing Atari Games: [Mnih et al., 2013](https://arxiv.org/abs/1312.5602)
- Pytorch DQN Implementation: [Reinforcement Learning (DQN) Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#reinforcement-learning-dqn-tutorial)

---