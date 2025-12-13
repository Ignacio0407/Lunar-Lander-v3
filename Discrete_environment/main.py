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

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, min(batch_size, len(self.memory)))
    
    def __len__(self):
        return len(self.memory)

NUM_EPISODES = 10000
BATCH_SIZE = 256 # Number of transitions sampled from the replay buffer
GAMMA = 0.99 # Discount factor of q or policy network
LR = 3e-4
TAU = 0.005 # Update rate of the target network

epsilon = 1.0  # Starting value of epsilon for epsilon greedy policy. 1 is full exploration (all actions taken randomly)
EPSILON_MIN = 0.05  # Minimum value
EPSILON_DECAY = 0.993  # Decay factor per episode, higher means a slower decay

EARLY_STOPPING_ENABLED = False
EARLY_STOPPING_THRESHOLD = 20
EARLY_STOPPING_STARTING_EPISODE = 1000
INITIAL_PATIENCE = 300
early_stopping_patience = INITIAL_PATIENCE
best_reward = -200.
stop_training = False
reward_list = []
reward_average_100_episodes = 0.
reward_counter:int = 0
main_thrust_counter:int = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Initialize environment WITH WIND
env = gym.make("LunarLander-v3", enable_wind=False)

n_observations = env.observation_space.shape[0]  # 8 observaciones en LunarLander-v3
n_actions = env.action_space.n  # 4 acciones discretas

policy_net = DQN(n_observations, n_actions).to(DEVICE)
target_net = DQN(n_observations, n_actions).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # Target network in evaluation mode

replay_memory = ReplayMemory(100000)

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
    main_thrust_counter = 0  # Reset contador por episodio
    
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
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
                    # Select actions using policy net (1 action per state). Outputs Q values for each action.
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
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10)
            optimizer.step()
        
        # --- SOFT UPDATE TARGET NETWORK ---
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(TAU * policy_param.data + (1 - TAU) * target_param.data)
        
        if done:
            reward_list.append(total_reward)
            print("Episode", episode, "Reward", total_reward)
            reward_counter += 1
            if EARLY_STOPPING_ENABLED and episode > EARLY_STOPPING_STARTING_EPISODE and len(reward_list) >= 100:
                reward_average_100_episodes = np.mean(reward_list[-100:])
                if reward_average_100_episodes > best_reward + EARLY_STOPPING_THRESHOLD:
                    best_reward = reward_average_100_episodes
                    early_stopping_patience = INITIAL_PATIENCE  # reset patience because a new better path might arise
                else:
                    early_stopping_patience -= 1
                    print("Patience", early_stopping_patience)
                    if early_stopping_patience == 0:
                        print("Early stopping triggered")
                        stop_training = True
            break
    
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

torch.save(policy_net.state_dict(), "models/ddqn_lunar_lander_windy.pth")
print("Training completed and model saved successfully!")
env.close()