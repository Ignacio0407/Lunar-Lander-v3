import random
import gymnasium as gym
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from prioritized_replay_memory import PrioritizedReplayMemory, Transition
from dqn import DQN

NUM_EPISODES = 10000
BATCH_SIZE = 256 # Number of transitions sampled from the replay buffer
GAMMA = 0.99 # Discount factor of q or policy network
LR = 3e-4
TAU = 0.005 # Update rate of the target network

epsilon = 1.0  # Starting value of epsilon for epsilon greedy policy. 1 is full exploration (all actions taken randomly)
EPSILON_MIN = 0.05  # Minimum value
EPSILON_DECAY = 0.993  # Decay factor per episode, higher means a slower decay

EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_THRESHOLD = 20
EARLY_STOPPING_STARTING_EPISODE = 4000
INITIAL_PATIENCE = 200
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
env = gym.make("LunarLander-v3", enable_wind=True)

n_observations = env.observation_space.shape[0]  # 8 observations in LunarLander-v3
n_actions = env.action_space.n  # 4 discrete actions

policy_net = DQN(n_observations, n_actions).to(DEVICE)
target_net = DQN(n_observations, n_actions).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # Target network in evaluation mode

replay_memory = PrioritizedReplayMemory(capacity=100000, alpha=0.6, beta_start=0.4, beta_frames=100000)

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
    main_thrust_counter = 0  # Reset contador per episode
    
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        pos_x, pos_y, vel_x, vel_y, angle, ang_vel, leg1, leg2 = observation 
        near_landed = (abs(pos_x) < 0.25 and pos_y < 0.2 and abs(vel_x) < 0.01 and abs(vel_y) < 0.05 and abs(angle) < 0.3)
        landed = (abs(pos_x) < 0.25 and pos_y < 0.02 and (leg1 == 1 and leg2 == 1))
        if near_landed and (action.item() == 1 or action.item() == 3):
            reward -= 0.5  # Penalty for thrusting unnecessarily
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
        
        # Proportional penalization to the horizontal distance to the center
        reward -= abs(pos_x) * 0.05 
        # Bonus for being close to the center
        if abs(pos_x) < 0.3:
            reward += 0.3
        # Stability in descend
        if pos_y < 0.5 and abs(vel_x) < 0.1 and abs(vel_y) < 0.1:
            reward += 0.2
        # Penaliza movimientos bruscos cerca del suelo
        if pos_y < 0.3 and abs(ang_vel) > 0.3:
            reward -= 0.75
        
        # --- SPECIAL HANDLING FOR TERMINAL STATES ---
        if terminated and not landed:
            reward -= 50
        
        # --- STORE TRANSITION ---
        reward_tensor = torch.tensor([reward], device=DEVICE)
        
        next_state = None
        if not done:
            next_state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        
        replay_memory.push(state, action, next_state, reward_tensor, done)
        
        state = next_state
        total_reward += reward
        
        if len(replay_memory) >= BATCH_SIZE:
            transitions, indices, weights = replay_memory.sample(BATCH_SIZE)
            batch = Transition(*zip(*transitions))

            state_batch = torch.cat(batch.state).to(DEVICE)              # shape [B, obs_dim]
            action_batch = torch.cat(batch.action).to(DEVICE)            # shape [B, 1]
            reward_batch = torch.cat(batch.reward).to(DEVICE)            # shape [B, 1] o [B]
            done_batch = torch.tensor(batch.done, device=DEVICE, dtype=torch.float32)  # shape [B]

            # mask and next states
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=DEVICE, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(DEVICE) if any(non_final_mask.cpu().numpy()) else torch.empty((0, state_batch.size(1)), device=DEVICE)

            next_state_values_full = torch.zeros(BATCH_SIZE, device=DEVICE, dtype=torch.float32)

            if non_final_next_states.size(0) > 0:
                with torch.no_grad():
                    next_actions = policy_net(non_final_next_states).max(1)[1].unsqueeze(1)     # [N_non_final, 1]
                    all_q_values = target_net(non_final_next_states)                           # [N_non_final, n_actions]
                    selected_q = all_q_values.gather(1, next_actions).squeeze(1)               # [N_non_final]
                    next_state_values_full[non_final_mask] = selected_q

            q_policy = policy_net(state_batch).gather(1, action_batch).squeeze(1)  # shape [B]

            q_target = reward_batch.squeeze().to(dtype=torch.float32) + (GAMMA * next_state_values_full * (1 - done_batch))

            td_errors = (q_target.detach() - q_policy.detach()).flatten()  # shape [B]

            # weights: already returned as 1D torch tensor on DEVICE and float32
            # ensure same dtype as loss
            weights = weights.to(dtype=torch.float32)

            # Huber loss per sample (reduction='none' -> shape [B])
            loss_per_sample = torch.nn.functional.smooth_l1_loss(q_policy, q_target, reduction='none')

            # apply importance-sampling weights (element-wise)
            loss = (weights * loss_per_sample).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10)
            optimizer.step()

            replay_memory.update_priorities(indices, td_errors.abs().cpu().numpy())
        
        # --- SOFT UPDATE TARGET NETWORK ---
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(TAU * policy_param.data + (1 - TAU) * target_param.data)
        
        if done:
            reward_list.append(total_reward)
            print("Episode", episode)
            reward_counter += 1
            if EARLY_STOPPING_ENABLED and episode > EARLY_STOPPING_STARTING_EPISODE and len(reward_list) >= 100:
                reward_average_100_episodes = np.mean(reward_list[-100:])
                if reward_average_100_episodes > best_reward + EARLY_STOPPING_THRESHOLD:
                    best_reward = reward_average_100_episodes
                    early_stopping_patience = INITIAL_PATIENCE  # reset patience because a new better path might arise
                else:
                    early_stopping_patience -= 1
                    print(f"⏳ Patience: {early_stopping_patience}/{INITIAL_PATIENCE}")
                    if early_stopping_patience == 0:
                        print("Early stopping triggered")
                        stop_training = True
            break
    
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    if episode % 100 == 0 and episode > 0:
        torch.save(policy_net.state_dict(), f"/kaggle/working/checkpoint_ep{episode}.pth")
        print(f"💾 Checkpoint saved at episode {episode}")

torch.save(policy_net.state_dict(), "models/ddqn_lunar_lander_windy.pth")
print("Training completed and model saved successfully!")
env.close()