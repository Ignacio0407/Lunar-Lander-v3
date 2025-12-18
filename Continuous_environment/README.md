# Double Deep Q-Network (DQN) for Car Racing Environment

This project trains a Deep Q-Network (DQN) using PyTorch to solve the Car Racing environment from Gymnasium. The agent learns to navigate complex tracks while maintaining optimal speed and avoiding collisions.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
  - [Prerequisites Locally](#prerequisites-locally)
  - [Prerequisites Kaggle](#prerequisites-kaggle)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Running Inference](#running-inference)
- [Hyperparameters](#hyperparameters)
- [Training Details](#training-details)
- [Car Racing Environment](#car-racing-environment)
  - [Action Space](#action-space)
  - [Observation Space](#observation-space)
  - [Rewards & Penalties](#rewards--penalties)
  - [Episode Termination](#episode-termination)
- [Model Performance Analysis](#model-performance-analysis)
  - [Standard Training vs Domain Randomization](#standard-training-vs-domain-randomization)
  - [Training Duration Analysis](#training-duration-analysis)
  - [Batch Size and Update Parameters](#batch-size-and-update-parameters)
  - [Fine-Tuning Strategies](#fine-tuning-strategies)
  - [Performance Comparison](#performance-comparison)
- [Conclusions](#conclusions)
- [References](#references)

## Introduction

The goal of this project is to train an agent using the Double Deep Q-Network (DDQN) algorithm to master the Car Racing environment. The agent learns to navigate challenging tracks with various turns and obstacles while optimizing for speed and track adherence. This project explores different training strategies including domain randomization and fine-tuning to improve generalization and performance.

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

**Note:** In Kaggle, add `!` at the beginning of commands. Graphical inference won't work on Kaggle due to environment limitations.

## Hyperparameters

- **Algorithm:** DDQN with Prioritized Experience Replay (PER)
- **Neural Network Architecture:**
  - Input: (4, 84, 84)
  - Conv1: (32, 20, 20)
  - Conv2: (64, 9, 9)
  - Conv3: (64, 7, 7)
  - Flatten: (3136, 512)
  - Output: (512, num_actions=5)
- **Learning Rate:** 3e-4
- **Discount Factor (Gamma):** 0.99
- **Replay Buffer Size:** 10,000
- **Batch Size:** 256 or 1024 (depending on model)
- **Target Network Update:** Soft Updates (Polyak Averaging), TAU = 0.005 or 0.01
- **Exploration Strategy:** ε-greedy (ε decays from 1.0 to 0.05)
- **Optimizer:** AdamW
- **Loss Function:** Smooth L1 loss
- **Gradient Clipping:** Normalization (max norm=10)

## Training Details

- **Convergence Reward:** 850+ (environment considered solved)
- **Hardware:** NVIDIA A100 40GB, Kaggle GPU (P100 16GB)
- **Frame Processing:** Grayscale, resized to 84x84, frame skipping
- **Improvements:** In-place gradient normalization to stabilize training without altering gradient directions

## Car Racing Environment

The Car Racing environment is a reinforcement learning task where an agent controls a car to navigate a randomly generated racetrack. The agent must learn to drive efficiently while staying on the track and managing speed through turns.

### Action Space

**Discrete(5):**

- 0: Do nothing
- 1: Steer right
- 2: Steer left
- 3: Accelerate
- 4: Brake

### Observation Space

**Box(0, 255, (96, 96, 3), uint8):** RGB image of the current environment state.

### Rewards & Penalties

- Reward increases for smooth driving and staying on track
- Base reward calculation: -0.1 per frame + 1000/N for each track tile visited
- Where N is the total number of tiles in the track
- Example: Finishing in 732 frames yields reward = 1000 - 0.1*732 = 926.8 points
- Significant penalty (-100) for driving off-track

### Episode Termination

- All track tiles are visited (successful completion)
- Car drives far off the track (failure condition)
- Maximum episode length reached (250 steps)

## Model Performance Analysis

### Standard Training vs Domain Randomization

#### Standard Training Models:

**Car Racing 2368** (BATCH_SIZE=256, TAU=0.005, EPSILON_DECAY=0.993):
- Most consistent performer with average reward ~870
- 42/50 episodes scored >850 reward
- 35/50 episodes reached the 250-step limit
- Peak performance: 928.5 reward
- Demonstrates excellent stability despite shortest training duration

**Car Racing 4600** (BATCH_SIZE=256, TAU=0.005, EPSILON_DECAY=0.993):
- Peak performance: 929.5 reward (Episode 36)
- More inconsistent than Car Racing 2368 despite identical hyperparameters
- 8/50 episodes scored <700 reward
- Shows signs of overtraining - longer training with same hyperparameters reduced consistency
- Average reward: ~840 (lower than 2368 model's ~870 despite higher peak)

**Car Racing 15695** (BATCH_SIZE=1024, TAU=0.01, EPSILON_DECAY=0.9995):
- Highest peak reward (931.10) but greater variance
- 12/50 episodes scored <700 reward (significant failures)
- 41/50 episodes reached the 250-step limit
- Extended exploration (slow epsilon decay) enabled discovery of high-reward strategies

#### Domain Randomization Models:

**Car Racing Domain Randomize 2300:**
- Highly unstable performance (18.69 to 879.02)
- Multiple catastrophic failures (e.g., Episode 9: 18.69 reward)
- Demonstrates insufficient training for domain randomization technique

**Car Racing Domain Randomize 7300:**
- Performance highly variable (303.70 to 924.30)
- 15/50 episodes scored <700 reward
- Failed to adapt consistently to randomized environments

**Car domain randomize 16042** (BATCH_SIZE=1024, TAU=0.01, EPSILON_DECAY=0.9995):
- Most extensively trained domain randomization model (16,042 episodes)
- Peak reward: 928.30 (Episode 8)
- Still highly inconsistent despite extensive training
- 18/50 episodes scored <700 reward
- Average reward: ~740 (significantly lower than non-domain-randomized counterparts)
- Demonstrates that even with extensive training, domain randomization creates significant learning challenges

### Training Duration Analysis

**Optimal Training Duration:**
- Car Racing 2368 achieved better consistency than Car Racing 4600 with identical hyperparameters
- Suggests an optimal training duration exists for specific hyperparameter configurations
- Beyond this point, overfitting to specific track patterns may occur

**Extended Training Benefits:**
- Car Racing 15695 achieved highest peak reward despite inconsistency
- Demonstrates that very long training can discover superior strategies when exploration is maintained

**Domain Randomization Requirements:**
- Even after 16,042 episodes, domain randomization model still underperforms compared to standard training
- Suggests domain randomization requires orders of magnitude more training data to overcome environmental variance

### Batch Size and Update Parameters

**Batch Size Effect:**
- Smaller batches (256): Faster convergence to stable policies (Models 2368 & 4600)
- Larger batches (1024): Higher potential peak performance but less consistency (Models 15695 & Domain Randomize 16042)

**Target Network Update (TAU):**
- Smaller TAU (0.005): More stable learning but slower adaptation
- Larger TAU (0.01): Faster knowledge transfer but potentially less stable

**Epsilon Decay Strategy:**
- Faster decay (0.993): Promotes earlier exploitation, leading to consistent but potentially suboptimal policies
- Slower decay (0.9995): Maintains exploration longer, enabling discovery of higher-reward strategies but risking instability

### Fine-Tuning Strategies

**Fine-tuning from 4600 episodes** (4700 additional episodes with domain randomization):
- Average reward: ~810
- Peak performance: 932.10
- 12/50 episodes scored <600 reward
- Successful transfer learning with domain adaptation

**Fine-tuning from 2368 episodes** (2210 additional episodes with domain randomization):
- Average reward: ~730
- Peak performance: 930.40
- 16/50 episodes scored <600 reward
- Less stable than the 4600-based fine-tuning approach

### Performance Comparison

| Model | Training Episodes | Domain Randomization | Batch Size | Avg. Reward | Peak Reward | Consistency (>850) | Episodes <700 |
|-------|-------------------|---------------------|------------|-------------|-------------|-------------------|---------------|
| Car Racing 2368 | 2,368 | No | 256 | 869.5 | 928.5 | 42/50 | 2/50 |
| Car Racing 4600 | 4,600 | No | 256 | 840.2 | 929.5 | 35/50 | 8/50 |
| Car Racing 15695 | 15,695 | No | 1024 | 829.6 | 931.1 | 33/50 | 12/50 |
| Car Racing DR 2300 | 2,300 | Yes | 256 | 628.4 | 879.0 | 18/50 | 22/50 |
| Car Racing DR 7300 | 7,300 | Yes | 1024 | 716.4 | 924.3 | 19/50 | 15/50 |
| Car domain randomize 16042 | 16,042 | Yes | 1024 | 740.3 | 928.3 | 21/50 | 18/50 |
| Fine-tune 4600→4700 | 9,300* | Yes | 256 | 810.7 | 932.1 | 29/50 | 12/50 |
| Fine-tune 2368→2210 | 4,578* | Yes | 256 | 730.3 | 930.4 | 23/50 | 16/50 |

*\*Combined training episodes (base + fine-tuning)*

#### Key Findings:

- Standard training without domain randomization produces more consistent results
- Car Racing 2368 represents the optimal balance of performance and reliability
- Domain randomization shows limited benefits even after extensive training (16,042 episodes)
- Fine-tuning from a well-trained model (4600) with domain randomization yields better results than training from scratch with domain randomization
- There is a clear tradeoff between peak performance and consistency across all models
- Car Racing 4600 demonstrates that additional training with identical hyperparameters can increase peak performance but reduce consistency
- Car domain randomize 16042 shows that even extremely long training periods cannot fully overcome the challenges introduced by domain randomization

## Conclusions

- For production deployment, the **Car Racing 2368** model offers the best reliability-to-performance ratio with 87% of episodes scoring above the solution threshold (850).
- When maximum performance is prioritized over consistency, the **Car Racing 15695** model achieves the highest peak reward (931.10) despite higher failure rate.
- Domain randomization shows limited effectiveness for this environment, requiring substantially more training than standard approaches with diminishing returns. Even after 16,042 episodes, performance remains inconsistent.
- Fine-tuning with domain randomization starting from a pre-trained model is more effective than training from scratch with domain randomization.
- Hyperparameter selection significantly impacts the performance-stability tradeoff:
  - Smaller batch sizes (256) promote stability
  - Faster epsilon decay creates more consistent policies
  - Smaller TAU values (0.005) improve training stability
- Optimal training duration exists for specific hyperparameter sets - Car Racing 2368 achieved better consistency than Car Racing 4600 despite identical hyperparameters.
- Future improvements could include:
  - Curriculum learning approaches that gradually introduce domain variations
  - Ensemble methods to handle environmental variations
  - Testing intermediate batch sizes (512) to balance stability and performance
  - Adaptive exploration strategies that respond to performance plateaus
  - Hybrid approaches combining domain randomization with other generalization techniques

## References

- [OpenAI Gymnasium: Car Racing Documentation](https://gymnasium.farama.org/environments/box2d/car_racing/)
- [Playing Atari with Deep Reinforcement Learning - Google DeepMind](https://arxiv.org/abs/1312.5602)
- [Deep Q-Network (DQN) Paper: Mnih et al., 2013](https://arxiv.org/abs/1312.5602)
- [PyTorch DQN Implementation: Reinforcement Learning (DQN) Tutorial](https://docs.pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)
- [Domain Randomization for Transferring Deep Neural Networks: Tobin et al., 2017](https://arxiv.org/abs/1703.06907)
- [Prioritized Experience Replay: Schaul et al., 2015](https://arxiv.org/abs/1511.05952)