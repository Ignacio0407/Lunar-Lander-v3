# Deep Q-Network (DQN) for LunarLander-v3

This project trains a (Double) Deep Q-Network (DQN) using PyTorch to solve the "LunarLander-v3" environment from Gymnasium.

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
  - [General Hyperparameters](#general-hyperparameters)
  - [Training Details](#training-details)
  - [LunarLander-v3 Environment](#lunarlander-v3-environment)
    - [Action Space](#action-space)
    - [Observation Space](#observation-space)
    - [Rewards \& Penalties](#rewards--penalties)
    - [Episode Termination](#episode-termination)
  - [Experiments](#experiments)
    - [32 Neurons Per Layer](#32-neurons-per-layer)
    - [64 Neurons Per Layer](#64-neurons-per-layer)
    - [128 Neurons Per Layer](#128-neurons-per-layer)
    - [256 Neurons Per Layer](#256-neurons-per-layer)
    - [Wind Models (Domain Randomization)](#wind-models-domain-randomization)
    - [Fine-Tuned Model from 128 Best](#fine-tuned-model-from-128-best)
    - [Models with High Episodes and Complex Networks](#models-with-high-episodes-and-complex-networks)
  - [Comprehensive Model Comparison](#comprehensive-model-comparison)
    - [Standard Environment Performance Analysis](#standard-environment-performance-analysis)
    - [Wind Environment Performance Analysis](#wind-environment-performance-analysis)
    - [Fine-Tuning vs Direct Wind Training](#fine-tuning-vs-direct-wind-training)
    - [Overall Best Models Summary](#overall-best-models-summary)
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

## General Hyperparameters

- **Algorithm:** DQN and DDQN for wind models
- **Neural Network:** 8 → 128 → 128 → number_of_actions = 4
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

### 32 Neurons Per Layer

None of the models with this NN managed to land successfully even a single time. They were trained with:
- General hyperparameters, with a patience of 60
- Modified hyperparameters, not worth mentioning since the model is really bad.

**Verdict**: Insufficient network capacity for the task.

### 64 Neurons Per Layer

There are four versions of this setup. Two with 2 hidden layers, one with 1 and another one with 4.

- **1 hidden layer**: Really bad model, never lands properly, showing that more depth is needed.
- **2 Hidden layers (first version)**: Patience of 40. Gave some nice results but many bad ones. Early stopping triggered in episode 326.
- **2 Hidden layers (hyperparameters modified)**: Taking into account its small size, it is a good model, successfully completing the environment in most runs and achieving reasonable results when it does not. 
  - Hyperparameters: TAU = 0.006, EPSILON_MIN = 0.01, EPSILON_DECAY = 0.997, EARLY_STOPPING_THRESHOLD = 12, early_stopping_patience = 35.
- **4 hidden layers**: Really bad model. Most times it seems like it is going to land correctly but goes too fast, legs open and crashes—what happened to models before reward shaping was implemented, showing that bigger NNs don't always give better solutions. This is mostly due to the fact that big NNs tend to stick to what they discover and do not explore as much as smaller ones, likely getting stuck in a local maximum.

**Verdict**: 2 hidden layers with 64 neurons is viable but requires careful hyperparameter tuning.

### 128 Neurons Per Layer

Three models were trained under the same conditions with general hyperparameters, differing only in reward shaping:

- **128_1**: `landed = (abs(pos_x) < 0.3 and pos_y < 0.3 and abs(vel_x) < 0.2 and abs(vel_y) < 0.2 and (leg1 == 1 and leg2 == 1))`
- **128_2**: `landed = (abs(pos_x) < 0.1 and pos_y < 0.1 and abs(vel_x) < 0.1 and abs(vel_y) < 0.1 and (leg1 == 1 or leg2 == 1))`
- **128_best**: `landed = (abs(pos_x) < 0.2 and pos_y < 0.2 and abs(vel_x) < 0.07 and abs(vel_y) < 0.03 and (leg1 == 1 or leg2 == 1) and abs(angle) < 0.2)`
  - Due to improved reward shaping, early stopping triggered in episode 376.

**128_hyperparameters_change**: Stopped training in episode 325.
- Hyperparameters: EPSILON_MIN = 0.02, EPSILON_DECAY = 0.996, EARLY_STOPPING_THRESHOLD = 12, early_stopping_patience = 25.
- Sometimes lands correctly, sometimes keeps thrusting instead of landing.

**Verdict**: **128_best is the optimal model for standard environment**—excellent landing accuracy with efficient training.

### 256 Neurons Per Layer

- **256**: Early stopping triggered in episode 324. Results are pretty similar to "128_best" (landing accurately most times), meaning the task is so simple it does not require more neurons.
- **256_hyperparameters_change**: 330 episodes.
  - Hyperparameters: LR = 2e-4, TAU = 0.004, EPSILON_MIN = 0.02, EARLY_STOPPING_THRESHOLD = 12, early_stopping_patience = 30.
  - Has learnt to get to the center but keeps thrusting instead of actually landing.

**Verdict**: No significant improvement over 128 neurons—confirms task simplicity doesn't warrant larger networks.

### Wind Models (Domain Randomization)

All wind models were trained with general hyperparameters. What changed between them was the reward shaping, showing how effective it is in certain cases:

- **wind_1** (320 episodes): Performs horribly, never crashes because it wanders off immediately.
- **wind_2** (345 episodes): Performs much better than the first one because it tries to go to the center (reward shaping adjusted to increase reward the closer it was to x=0), but instead of landing it kept thrusting and most times deviated too much from center.
- **wind_3** (600 episodes): Clearly the best one so far, since it actually lands accurately sometimes between the flags with rewards around 240, but still it mostly drifts too far away, like model 1.
- **wind_4** (600 episodes): Added 1 more hidden layer to the NN. Has almost learnt how and where to land, since it goes almost all times to the center, but many times does not land appropriately because it goes too fast, legs open and crashes.
- **wind_5** (600 episodes): Similar to wind_4 with minor reward shaping differences. Had 50,000 buffer size in replay memory and batch size of 256. Results more or less the same as wind_4.

**Verdict**: **wind_3 is the best wind model**, achieving occasional accurate landings with ~240 reward, though consistency remains a challenge.

### Fine-Tuned Model from 128 Best

Starting from the best standard model (128_best) and applying wind conditions:

**Performance**: Not worth mentioning since it is really bad, like wind_1. The fine-tuned model completely failed to adapt to wind conditions, wandering off immediately without landing attempts.

**Verdict**: Fine-tuning from a highly optimized standard model to wind conditions failed catastrophically—worse than training from scratch with wind.

### Models with High Episodes and Complex Networks

All trained with:
- Batch size of 512
- DQN_heavy and other variants (see code)
- Goal: Test if complex networks could avoid overfitting

**Result**: Not worth detailed analysis. Complex networks refused to learn the task when it was too simple for their capacity. They do not have video demonstrations due to poor performance.

**Verdict**: Overfitting is unavoidable when network complexity far exceeds task requirements—networks refuse to learn instead of converging.

## Comprehensive Model Comparison

### Standard Environment Performance Analysis

| Network Size | Best Model | Episodes | Landing Success | Training Efficiency | Notes |
|--------------|------------|----------|----------------|-------------------|-------|
| 32 neurons | None | N/A | Never | N/A | Insufficient capacity |
| 64 neurons, 2HL | 64_hyperparams_mod | ~326 | Mostly | Moderate | Good for its size |
| 128 neurons, 2HL | **128_best** | 376 | Excellent | **Optimal** | **Best overall** |
| 256 neurons, 2HL | 256 | 324 | Excellent | Good | No improvement vs 128 |
| Complex/Deep | None | 600+ | Poor | Very Poor | Refused to learn |

**Winner: 128_best** - Optimal balance of performance, efficiency, and reliability.

**Key Finding**: 128 neurons with 2 hidden layers is the sweet spot. Larger networks (256) offer no benefit, while smaller networks (64) require extensive tuning. Networks that are too complex simply fail to learn.

### Wind Environment Performance Analysis

| Model | Episodes | Landing Success | Reward Range | Behavior Pattern | Effective Reward Shaping |
|-------|----------|----------------|--------------|-----------------|-------------------------|
| wind_1 | 320 | Never | Very Low | Wanders off immediately | No |
| wind_2 | 345 | Rare | Low | Reaches center, keeps thrusting | Partial |
| **wind_3** | 600 | **Sometimes** | **~240 peak** | **Lands between flags occasionally** | **Yes** |
| wind_4 | 600 (+1 layer) | Rare | Moderate | Goes to center, crashes (too fast) | Partial |
| wind_5 | 600 (+1 layer, larger buffer) | Rare | Moderate | Similar to wind_4 | Partial |

**Winner: wind_3** - Only model achieving occasional accurate landings in wind conditions.

**Key Finding**: Wind environment is dramatically harder. Even the best model (wind_3) only occasionally succeeds, while standard models (128_best) succeed consistently. Deeper networks (wind_4, wind_5) didn't help—reward shaping was the critical factor.

### Fine-Tuning vs Direct Wind Training

| Approach | Training Strategy | Episodes | Performance | Landing Success | Verdict |
|----------|------------------|----------|-------------|----------------|---------|
| **Direct Wind Training** | Train from scratch with wind | 320-600 | Variable | wind_3: Sometimes | Possible but difficult |
| **Fine-Tuning from 128_best** | Standard training → apply wind | 376 + wind epochs | Catastrophic | Never | Complete failure |

**Critical Insight**: Fine-tuning from the best standard model (128_best) to wind conditions **completely failed**, performing worse than even the worst direct wind training (wind_1). The highly specialized standard policy could not adapt to wind perturbations.

**Why Fine-Tuning Failed:**
1. **Over-specialization**: 128_best was too optimized for the standard environment, making its policy brittle to environmental changes
2. **Policy Disruption**: Introducing wind after convergence disrupted the learned landing strategy entirely
3. **No Exploration**: The fine-tuned model had low epsilon, preventing it from discovering new strategies for wind conditions

**Recommendation**: For wind environments, **train from scratch** with domain randomization rather than fine-tuning from standard models.

### Overall Best Models Summary

| Category | Model | Episodes | Key Strengths | Use Case |
|----------|-------|----------|---------------|----------|
| **Standard Environment** | **128_best** | 376 | Excellent accuracy, fast convergence, efficient | **Production deployment** |
| **Standard (Alternative)** | 64_hyperparams_mod | ~326 | Good performance with smaller network | Resource-constrained scenarios |
| **Wind Environment** | **wind_3** | 600 | Only model with occasional successful landings | **Wind conditions** |
| **Worst Approach** | Fine-tuned 128_best | 376 + wind | Complete failure in wind | **Avoid this approach** |

## Conclusions

### 🚀 General Findings

- The DQN/DDQN approach with PyTorch successfully solves LunarLander-v3, reaching convergence rewards above 200 in standard conditions.
- **Reward shaping was critical**: without it, larger networks tended to crash or get stuck in local maxima; with tailored shaping, convergence was faster and more stable.
- Early stopping prevented wasted computation, typically halting training between 300–600 episodes depending on architecture and shaping.

### 🧠 Neural Network Architectures

- **Small networks (32 neurons, shallow depth)**: consistently failed, unable to learn stable landing behavior.
- **Moderate networks (64–128 neurons, 2 hidden layers)**: performed best, balancing capacity and generalization. The **128_best model** with refined reward shaping was the most reliable.
- **Large networks (256 neurons, deeper layers)**: did not improve performance; often overfit or failed to land properly, showing that complexity beyond task requirements harms learning.
- **Very complex networks**: refused to learn entirely, demonstrating that excessive capacity for simple tasks leads to training failure.

### ⚙️ Hyperparameters & Training

- Learning rate, epsilon decay, and patience strongly influenced outcomes. Slight adjustments (e.g., slower epsilon decay, different TAU) improved stability in smaller networks.
- Replay buffer size and batch size changes had limited impact compared to reward shaping and network size.
- Gradient normalization stabilized training and prevented divergence without altering gradient directions.

### 🌬️ Wind Models & Domain Randomization

- Introducing wind made the task **dramatically harder**. Standard models achieved consistent success (128_best), while the best wind model (wind_3) only succeeded occasionally.
- **Reward shaping determined success in wind conditions**: models that rewarded proximity to center improved, but completing landings remained challenging.
- Larger networks (wind_4, wind_5) under wind conditions did not outperform moderate ones (wind_3), reinforcing that **reward shaping > network size** for this task.
- **Direct wind training >> Fine-tuning**: Training from scratch with wind produced viable models (wind_3), while fine-tuning from 128_best **catastrophically failed**.

### 🔄 Fine-Tuning Insights

- **Fine-tuning failed catastrophically** when applied to wind conditions from the best standard model (128_best).
- Over-specialized standard models cannot adapt to environmental perturbations—their optimized policies become brittle.
- For domain randomization or harder environments, **always train from scratch** rather than fine-tuning from highly converged standard models.

### 📊 Overfitting & Complexity

- Models with excessive depth or neurons tended to overfit to discovered strategies, refusing to explore alternatives.
- Overfitting manifested as agents thrusting indefinitely or failing to adapt to landing conditions, despite high training rewards.
- **Worst case**: Extremely complex networks simply refused to learn, showing zero improvement over 600+ episodes.

### ✅ Overall Conclusions

1. **Optimal Architecture**: 128 neurons with 2 hidden layers (128_best) is optimal for standard LunarLander-v3—efficient, reliable, and fast converging.

2. **Reward Shaping > Network Size**: Tailored reward shaping is more decisive than network architecture in achieving stable landings, especially under challenging conditions.

3. **Wind Requires Training from Scratch**: Fine-tuning highly optimized models to harder environments fails. Always train from scratch when enabling wind(e.g., wind_3).

4. **Complexity Ceiling**: Overly complex architectures degrade performance in simple control tasks. Match model capacity to environment complexity—bigger is not better.

5. **Wind Environment is Significantly Harder**: Even the best wind model (wind_3) only occasionally succeeds, while standard models succeed consistently. Wind requires substantially more training and better reward shaping.

6. **Early Stopping is Essential**: Prevents wasted computation and overfitting, with optimal stopping points between 300-400 episodes for well-shaped rewards.

7. **64 vs 128 vs 256 Neurons**: 
   - 64: Viable with careful tuning, good for resource constraints
   - 128: **Sweet spot**—best performance-to-efficiency ratio
   - 256: No improvement, unnecessary complexity

8. **Training Strategy Matters More Than Architecture**: The training approach (reward shaping, exploration strategy, training from scratch vs fine-tuning) has greater impact on success than network size or depth.

## References

- OpenAI Gymnasium: [LunarLander-v3 Documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- Playing Atari with Deep Reinforcement Learning [Google DeepMind](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
- Deep Q-Network (DQN) Paper Playing Atari Games: [Mnih et al., 2013](https://arxiv.org/abs/1312.5602)
- Pytorch DQN Implementation: [Reinforcement Learning (DQN) Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#reinforcement-learning-dqn-tutorial)

---