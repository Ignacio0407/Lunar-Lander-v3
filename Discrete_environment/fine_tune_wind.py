from collections import deque, namedtuple
import random
import gymnasium as gym
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dqn_dynamic import DQN_dynamic
import os

Transition = namedtuple("Transition", ["state", "action", "next_state", "reward", "done"])

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, min(batch_size, len(self.memory)))
    
    def __len__(self):
        return len(self.memory)

NUM_EPISODES = 3000
BATCH_SIZE = 256
GAMMA = 0.99
LR = 1e-4
TAU = 0.005

# ⚡ Less exploration to take advantage of previous model knowledge
epsilon = 0.3
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995 

EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_THRESHOLD = 10 
INITIAL_PATIENCE = 150
early_stopping_patience = INITIAL_PATIENCE
best_reward = -200.0
stop_training = False
reward_list = []
main_thrust_counter = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Device for fine_tuning: {DEVICE}")

# 🌬️
env = gym.make("LunarLander-v3", enable_wind=True)

n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n

base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "models", "128_best.pth")
checkpoint = torch.load(model_path)
policy_net = DQN_dynamic(n_observations, n_actions, state_dict=checkpoint).to(DEVICE)
policy_net.load_state_dict(checkpoint)
policy_net.train()
target_net = DQN_dynamic(n_observations, n_actions, state_dict=checkpoint).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

replay_memory = ReplayMemory(150000)

def select_action(state):
    if np.random.rand() < epsilon:
        return torch.tensor([[env.action_space.sample()]], dtype=torch.long, device=DEVICE)
    else:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
criterion = nn.SmoothL1Loss()

for episode in range(NUM_EPISODES):
    if stop_training:
        break
        
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    total_reward = 0.0
    main_thrust_counter = 0
    
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        
        pos_x, pos_y, vel_x, vel_y, angle, ang_vel, leg1, leg2 = observation
        
        near_landed = (abs(pos_x) < 0.25 and pos_y < 0.2 and abs(vel_x) < 0.01 and abs(vel_y) < 0.05 and abs(angle) < 0.3)
        
        landed = (abs(pos_x) < 0.3 and pos_y < 0.01 and (leg1 == 1 and leg2 == 1))
        
        if near_landed and (action.item() == 1 or action.item() == 3):
            reward -= 0.3
        if near_landed and action.item() == 2:
            main_thrust_counter += 1
        else:
            main_thrust_counter = 0
            
        if main_thrust_counter > 5:
            reward -= 3
        
        if landed:
            if action.item() == 0:
                reward += 10
            else:
                reward -= 5
        
        reward -= abs(pos_x) * 0.03
        if abs(pos_x) < 0.35:
            reward += 0.2
        if pos_y < 0.6 and abs(vel_x) < 0.15 and abs(vel_y) < 0.15:
            reward += 0.15
        if pos_y < 0.35 and abs(ang_vel) > 0.4:
            reward -= 0.5
        
        if done and not landed:
            reward -= 35
        
        reward_tensor = torch.tensor([reward], device=DEVICE)
        
        next_state = None
        if not done:
            next_state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        
        replay_memory.push(state, action, next_state, reward_tensor, done)
        
        state = next_state
        total_reward += reward
        
        if len(replay_memory) >= BATCH_SIZE:
            transitions = replay_memory.sample(BATCH_SIZE)
            batch = Transition(*zip(*transitions))

            state_batch = torch.cat(batch.state).to(DEVICE)
            action_batch = torch.cat(batch.action).to(DEVICE)
            reward_batch = torch.cat(batch.reward).to(DEVICE)
            done_batch = torch.tensor(batch.done, device=DEVICE, dtype=torch.float32)

            # Create mask for non-final states
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=DEVICE, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(DEVICE)
            
            # Double DQN (DDQN) - CORRECT IMPLEMENTATION
            next_state_values_full = torch.zeros(BATCH_SIZE, device=DEVICE) # Configure values for terminal states

            if non_final_next_states.size(0) > 0:
                with torch.no_grad():
                    # Select actions using policy net. Outputs Q values for each action.
                    next_actions = policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                    # Evaluate using target net
                    all_q_values = target_net(non_final_next_states) # target_net predicts [130, 135, 140, 125] rewards for the actions.
                    # Only values for actions chosen by policy
                    next_state_values_full[non_final_mask] = all_q_values.gather(1, next_actions).squeeze(1) # Get values up to current state
            
            q_policy = policy_net(state_batch).gather(1, action_batch)
            # Compute expected Q values. done_batch es 1 for terminals, 0 for non-terminals.
            q_target = reward_batch.squeeze() + (GAMMA * next_state_values_full * (1 - done_batch))

            # Compute loss
            loss = criterion(q_policy, q_target.unsqueeze(1))
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()

            # In-place gradient clipping to stabilize training
            torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
            optimizer.step()
        
        # --- SOFT UPDATE TARGET NETWORK ---
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(TAU * policy_param.data + (1 - TAU) * target_param.data)
        
        if done:
            reward_list.append(total_reward)
            print("Episode", episode)
            if EARLY_STOPPING_ENABLED and episode > 200 and len(reward_list) >= 100:
                current_avg = np.mean(reward_list[-100:])
                if current_avg > best_reward + EARLY_STOPPING_THRESHOLD:
                    best_reward = current_avg
                    early_stopping_patience = INITIAL_PATIENCE
                else:
                    early_stopping_patience -= 1
                    print(f"⏳ Patience: {early_stopping_patience}/{INITIAL_PATIENCE}")
                    if early_stopping_patience <= 0:
                        stop_training = True
            break
    
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

torch.save(policy_net.state_dict(), "models/fine_tuned_wind.pth")
print("🎉 Fine-tuning completed and model saved!")
env.close()