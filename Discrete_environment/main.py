import random
import gymnasium as gym
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dqn import DQN

Transition = namedtuple("Transition", ["state", "action", "next_state", "reward", "done"])

# Experience replay buffer
class ReplayMemory:
    def __init__(self, capacity:float):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return (random.sample(self.memory, batch_size) if batch_size < len(self.memory) else self.memory)

    def __len__(self):
        return len(self.memory)


NUM_EPISODES = 600
BATCH_SIZE = 128 # Number of transitions sampled from the replay buffer
GAMMA = 0.99 # Discount factor of q or policy network
LR = 1e-4
TAU = 0.005 # Update rate of the target network

epsilon = 1.0  # Starting value of epsilon for epsilon greedy policy. 1 is full exploration (all actions taken randomly)
EPSILON_MIN = 0.01  # Minimum value
EPSILON_DECAY = 0.995  # Decay factor per episode, higher means a slower decay

EARLY_STOPPING_THRESHOLD = 10
early_stopping_patience = 20
best_reward = -200.
stop_training = False

reward_list = []
reward_average_100_episodes = 0.
reward_counter = 0
episode_durations = []

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# Initialize the environment
#env = gym.make("LunarLander-v3", render_mode="human")
env = gym.make("LunarLander-v3")

n_observations = env.observation_space.shape[0]
n_actions:int = env.action_space.n

policy_net = DQN(n_observations, n_actions).to(DEVICE)
target_net = DQN(n_observations, n_actions).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())

replay_memory = ReplayMemory(10000)


def select_action(state):
    if np.random.rand() < epsilon:
        return torch.tensor([[env.action_space.sample()]], dtype=torch.long, device=DEVICE)
    else:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)  # Exploit (best action)


optimizer = optim.AdamW(policy_net.parameters(), lr=LR)
criterion = nn.SmoothL1Loss()

for episode in range(NUM_EPISODES):
    if stop_training:
        break
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    total_reward:float = 0

    for t in count():
        action = select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action.item())
        
        # --- Reward adjustment ---
        # If it has already landed (very close to the floor and with low speed), propulsion is penalized, so that it stays put.
        pos_x, pos_y, vel_x, vel_y, angle, ang_vel, leg1, leg2 = next_state.tolist()
        landed = (abs(pos_x) < 0.2 and pos_y < 0.2 and abs(vel_x) < 0.07 and abs(vel_y) < 0.03 and (leg1 == 1 or leg2 == 1) and abs(angle) < 0.2)
        if landed and (action.item() == 1 or action.item() == 3):
            reward -= 1  # Penalty for thrusting unnecesarly
        if landed and action.item() == 0:
            reward += 1  # bonus for not moving once it has landed
        # Detecting crash: episode finished without correct landing
        if terminated and not landed:
            reward -= 1000  # Strong penalization for crashing
        # Proportional penalization to the horizontal distance to the center
        reward -= abs(pos_x) * 0.05 
        # Bonus for being close to the center
        if abs(pos_x) < 0.1 and abs(pos_y) < 0.1:
            reward += 1.0
    
        done = terminated or truncated
        reward = torch.tensor([reward], device=DEVICE)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        replay_memory.push(state, action, next_state, reward, done)

        state = next_state
        total_reward += reward.item()

        if len(replay_memory) >= BATCH_SIZE:
            transitions = replay_memory.sample(BATCH_SIZE)
            states, actions, next_states, rewards, dones = Transition(*zip(*transitions))

            states_batch = torch.cat(states)
            next_states_batch = torch.cat(next_states)
            actions_batch = torch.cat(actions)
            rewards = torch.tensor(rewards, device=DEVICE)
            dones = torch.tensor(dones, device=DEVICE)

            # Compute expected q-values
            q_target = (GAMMA * target_net(next_states_batch).detach().max(1)[0] * ~dones + rewards) # objective value calculated with target net values. ~dones is mask that fills terminal states with 0s.
            q_policy = policy_net(states_batch).gather(1, actions_batch) # prediction of Q(s,a) of policy net for the real states and actions taken by the lander.

            # Calculate the Huber loss
            loss = criterion(q_policy, q_target.unsqueeze(1))

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()

            # In-place gradient clipping to stabilize training
            torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)

            optimizer.step()

        # Soft Update of target network's weights - θ′ ← τ θ + (1 −τ )θ′
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(TAU * policy_param.data + (1 - TAU) * target_param.data)

        if done:
            episode_durations.append(t + 1)
            reward_list.append(total_reward)
            print("Episode", episode)
            reward_counter += 1
            if episode > 300 and len(reward_list) >= 100:
                reward_average_100_episodes = np.mean(reward_list[-100:])
                if reward_average_100_episodes > best_reward + EARLY_STOPPING_THRESHOLD:
                    best_reward = reward_average_100_episodes
                    early_stopping_patience = 20  # reset patience because a new better path might arise
                else:
                    early_stopping_patience -= 1
                    print("Patience", early_stopping_patience)
                    if early_stopping_patience == 0:
                        print("Early stopping triggered")
                        stop_training = True
            break

    # Decay epsilon
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

# Save the trained model
torch.save(policy_net.state_dict(), "models/dqn_lunar_lander_discrete_environment.pth")
print("Model saved successfully!")

print("Complete")
env.close()