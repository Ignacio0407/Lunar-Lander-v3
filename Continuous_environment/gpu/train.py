import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch_per import TorchPER
from dqn import DQN
from preprocessing import SkipFrame
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
from gymnasium.wrappers import FrameStackObservation as FrameStack

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ENVS = 8
TOTAL_STEPS = 2_000_000
REPLAY_SIZE = 300_000
BATCH_SIZE = 512
GAMMA = 0.99
LR = 1e-4
TARGET_UPDATE = 10_000
TRAIN_EVERY = 4
GRAD_CLIP = 10.0

EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY_STEPS = 1_000_000

print("Using device:", DEVICE)

def make_env():
    env = gym.make("CarRacing-v3", continuous=False)
    env = SkipFrame(env, skip=4)
    env = GrayscaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, (84, 84))
    env = FrameStack(env, 4)
    return env

envs = gym.vector.AsyncVectorEnv([make_env for _ in range(NUM_ENVS)])

obs_shape = envs.single_observation_space.shape
n_actions = envs.single_action_space.n

policy_net = DQN(obs_shape, n_actions).to(DEVICE)
target_net = DQN(obs_shape, n_actions).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

replay = TorchPER(capacity=REPLAY_SIZE, state_shape=obs_shape, device=DEVICE)

def epsilon_by_step(step):
    return max(EPS_END, EPS_START - step * (EPS_START - EPS_END) / EPS_DECAY_STEPS)

def train_step():
    (states, actions, rewards, next_states, dones, indices, weights) = replay.sample(BATCH_SIZE)

    with torch.no_grad():
        next_actions = policy_net(next_states).argmax(1, keepdim=True)
        next_q = target_net(next_states).gather(1, next_actions)
        target_q = rewards + GAMMA * next_q * (1 - dones)

    current_q = policy_net(states).gather(1, actions)
    td_errors = target_q - current_q

    loss = (weights * nn.functional.smooth_l1_loss(current_q, target_q, reduction="none")).mean()

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), GRAD_CLIP)
    optimizer.step()

    replay.update_priorities(indices, td_errors)


states, _ = envs.reset()
states = torch.tensor(states, device=DEVICE, dtype=torch.float32)

for global_step in range(1, TOTAL_STEPS + 1):

    eps = epsilon_by_step(global_step)
    if np.random.rand() < eps:
        actions = torch.randint(0, n_actions, (NUM_ENVS, 1), device=DEVICE)
    else:
        with torch.no_grad():
            actions = policy_net(states).argmax(1, keepdim=True)

    next_states, rewards, terms, truncs, _ = envs.step(actions.cpu().numpy())
    dones = terms | truncs

    next_states_t = torch.tensor(next_states, device=DEVICE, dtype=torch.float32)
    rewards_t = torch.tensor(rewards, device=DEVICE).unsqueeze(1)
    dones_t = torch.tensor(dones, device=DEVICE).unsqueeze(1).float()

    for i in range(NUM_ENVS):
        replay.push(states[i], actions[i], rewards_t[i], next_states_t[i], dones_t[i])

    states = next_states_t

    if replay.size >= BATCH_SIZE and global_step % TRAIN_EVERY == 0:
        train_step()

    if global_step % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        print(f"🔁 Target updated at step {global_step}")

    if global_step % 50_000 == 0:
        torch.save(policy_net.state_dict(), f"checkpoints/car_racing_step_{global_step}.pth")
        print(f"💾 Checkpoint at {global_step}")

envs.close()
torch.save(policy_net.state_dict(), "car_racing_final.pth")
print("✅ Training finished")