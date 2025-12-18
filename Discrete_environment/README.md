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

The goal of this project is to train an agent using the DQN and DDQN algorithm to land a spacecraft safely on a designated landing pad in the "LunarLander-v3" environment. The environment is a classic control problem based on Box2D physics, where the agent must optimize its thrusters to achieve a stable landing.

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

- **Algorithm:** DQN and DDQN for wind models
- **Neural Network:** 8 -> 128 -> 128 -> number_of_actions
- **Episodes:** 600
- **Learning Rate:** 0.0001
- **Discount Factor (Gamma):** 0.99
- **Replay Buffer Size:** 10,000
- **Batch Size:** 128
- **Target Update:** Soft Updates (Polyak Averaging), TAU = 0.005
- **Exploration Strategy:** ε-greedy (ε decays from 1.0 to 0.01 with a 0.99 decay factor per episode)
- **Optimizer:** AdamW
- **Loss Function:** Smooth L1 loss
- **Early Stopping Patience:** 20 (when it reaches 0, the algorithm is stopped)
- **Early Stopping Threshold:** 10 (minimum reward upgrade to not decrease patience. If reward increases in average of the last 100 episodes less than threshold, then patience is reduced by 1)

## Training Details

- **Episodes Trained:** 600, with early stopping implementation to prevent waste of computational resources.
- **Convergence Reward:** 200+ (environment considered solved)
- **Reward shaping:** Can be seen between lines 87-102. I added some rewards and penalties of my own to make the model converge faster.
- **Hardware:** GPU RTX 3060 laptop (6GB VRAM), Kaggle gpu p100 16gb and Nvidia a100 40gb
- **Improvements:** Applied in-place gradient normalization (up to 10) to stabilize training and not altering gradients directions, unlike what happens with clipping.

## Experiments

The modified hyperparameters are the neurons per layer, the depth of the NN, the learning rate and the epsilon (of the greedy policy)

#### 32 neurons per layer
None of the models with this NN managed to land successfully even a single time. They were trained with
- General hyperparameters, with a patience of 60
- Modified hyperparameters, not worth mentioning since the model is really bad.

#### 64 neurons per layer
There are four versions of this setup. Two with 2 hidden layers, one with 1 and another one with 4.
- 1 hidden layer: It is a really bad model, never lands properly, showing that more depth is needed.
- 2 Hidden layers: patience of 40. It gave some nice results but many bad ones. I can't show it because it was the first one trained in Kaggle and I lost it. Early stopping triggered in episode 326.
- 2 Hidden layers with hyperparameters modified: Taking into account its small size, it is a good model, succesfully completing the environment in most runs and achieving not too bad results when it does not. Hyperparameters: TAU = 0.006, EPSILON_MIN = 0.01, EPSILON_DECAY = 0.997, EARLY_STOPPING_THRESHOLD = 12, early_stopping_patience = 35.
- 4 hidden layer: It is a really bad model, most times it seems like it is going to land correctly but is going too fast and the legs opened and crashed, what happened to the models before I made the reward shaping, showing that not the bigger the NN the better solution it will give. This is mostly due to the fact that big NN tend to stick to what they discover and do not explore as much as little ones, so it likely got stuck in a local maximum.

#### 128 neurons per layer
128_1, 128_2 and 128_best. They were all trained under the same conditions, with general hyperparameters. The only difference was that they received slightly different reward shaping, since the landed condition was respectively for the three models:
- landed = (abs(pos_x) < 0.3 and pos_y < 0.3 and abs(vel_x) < 0.2 and abs(vel_y) < 0.2 and (leg1 == 1 and leg2 == 1))
- landed = (abs(pos_x) < 0.1 and pos_y < 0.1 and abs(vel_x) < 0.1 and abs(vel_y) < 0.1 and (leg1 == 1 or leg2 == 1))
- landed = (abs(pos_x) < 0.2 and pos_y < 0.2 and abs(vel_x) < 0.07 and abs(vel_y) < 0.03 and (leg1 == 1 or leg2 == 1) and abs(angle) < 0.2). Due to the improvement of the reward shaping, the early stopping triggered in episode 376 for 128_best.

128_hyperparameters_change: Stopped training in episode 325. Hyperparameters: EPSILON_MIN = 0.02, EPSILON_DECAY = 0.996, EARLY_STOPPING_THRESHOLD = 12, early_stopping_patience = 25. It sometimes lands correctly and sometimes keeps thrusting instead of landing.

#### 256 neurons per layer
- 256: Early stopping triggered in episode 324. The results are pretty similar to the one that model "128_best" gives (which is landing pretty accurately most times), meaning the task is so simple it does not require more neurons.
- 256 hyperparameters change: 330 episodes. LR = 2e-4, TAU = 0.004, EPSILON_MIN = 0.02, EARLY_STOPPING_THRESHOLD = 12, early_stopping_patience = 30. It has learnt to get to the center, but keeps thrusting instead of actually landing.

#### Wind models
All wind models were trained with general hyperparameters, changing only the episodes what changed between them was the reward shaping, showing how effective it is in certain cases:
- wind 1: 320 episodes. Performs horribly, never crushes because it wonders off inmediately. 
- wind 2: 345 episodes. Performs much better than the first one, because it tries to go to the center (I adjusted the reward shaping to increase reward the closer it was to the center, x=0), but instead of landing it kept thrusting and most times deviated too much from the centerS.
- wind 3: 600 episodes. This one is clearly the best one so far, since it actually lands accurately sometimes between the flags, with rewards arounf 240, but still it mostly drifts too far away, like model 1.
- wind 4 and wind 5: 600 episodes and added 1 more hidden layer to the NN. It has almost learnt how and where to land, since it goes almost all times to the center, but many times does not land appropiately because it goes too fast and the legs open and crashes. Their reward shaping had minor differences, hence they gave more or less the same results. Difference is that wind 5 had 50000 buffer size in replay memory and batch size of 256.

#### Fine tuned model from 128 best
Not worth mentioning since it is really bad, like wind 1.

#### Models with a lot of episodes and complex network
They were all trained with a batch size of 512 and with the DQN_heavy and other variants that can be seen in the code, I do not put much importance in them because I wanted to see if I could evade overfitting, but it seems unavoidable, if the network is too complex for the task at hand it will refuse to do learn how to do it. That is why they do not have video.

## Conclusions

🚀 General Findings
- The DQN/DDQN approach with PyTorch successfully solves LunarLander‑v3, reaching convergence rewards above 200.
- Reward shaping was critical: without it, larger networks tended to crash or get stuck in local maxima; with tailored shaping, convergence was faster and more stable.
- Early stopping prevented wasted computation, typically halting training between 300–600 episodes depending on architecture and shaping.

🧠 Neural Network Architectures
- Small networks (32 neurons, shallow depth): consistently failed, unable to learn stable landing behavior.
- Moderate networks (64–128 neurons, 2 hidden layers): performed best, balancing capacity and generalization. The “128_best” model with refined reward shaping was the most reliable.
- Large networks (256 neurons, deeper layers): did not improve performance; often overfit or failed to land properly, showing that complexity beyond task requirements harms learning.

⚙️ Hyperparameters & Training
- Learning rate, epsilon decay, and patience strongly influenced outcomes. Slight adjustments (e.g., slower epsilon decay, different TAU) improved stability in smaller networks.
- Replay buffer size and batch size changes had limited impact compared to reward shaping and network size.
- Gradient clipping stabilized training and prevented divergence.

🌬️ Wind Models & Variants
- Introducing wind made the task harder. Reward shaping again determined success: models that rewarded proximity to the center improved but often failed to complete landings.
- Larger networks under wind conditions did not outperform smaller ones, reinforcing that task simplicity favors moderate architectures.

📊 Overfitting & Complexity
- Models with excessive depth or neurons tended to overfit to discovered strategies, refusing to explore alternatives.
- Overfitting manifested as agents thrusting indefinitely or failing to adapt to landing conditions, despite high training rewards.

✅ Overall Conclusions
- Moderate networks (128 neurons, 2 hidden layers) with tailored reward shaping are optimal for LunarLander‑v3.
- Reward shaping is more decisive than network size in achieving stable landings.
- Early stopping and gradient clipping are effective safeguards against wasted computation and instability.
- Overly complex architectures degrade performance in simple control tasks, highlighting the importance of matching model capacity to environment complexity.
- For harder variants (wind, domain randomization), robust reward shaping and exploration strategies are more impactful than scaling up the network.


## References

- OpenAI Gymnasium: [LunarLander-v3 Documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- Playing Atari with Deep Reinforcement Learning [Google DeepMind](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
- Deep Q-Network (DQN) Paper Playing Atari Games: [Mnih et al., 2013](https://arxiv.org/abs/1312.5602)
- Pytorch DQN Implementation: [Reinforcement Learning (DQN) Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#reinforcement-learning-dqn-tutorial)

---