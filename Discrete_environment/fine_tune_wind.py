import gymnasium as gym
from replay_memory import ReplayMemory, Transition
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dqn import DQN2
import os

NUM_EPISODES = 8000
BATCH_SIZE = 256
GAMMA = 0.99
LR = 5e-6
TAU = 0.002

epsilon = 0.4
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9998

EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_THRESHOLD = 2
EARLY_STOPPING_STARTING_EPISODE = 6000
INITIAL_PATIENCE = 600
early_stopping_patience = INITIAL_PATIENCE
best_reward = -np.inf
stop_training = False
reward_list = []

USE_REWARD_SHAPING = True
REWARD_SHAPING_INTENSITY = 0.3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Fine-tuning device: {DEVICE}")

env = gym.make("LunarLander-v3", enable_wind=True)

n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n

base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "models", "wind_4.pth")

print(f"📦 Loading pre-trained model: {model_path}")

checkpoint = torch.load(model_path, map_location=DEVICE)
policy_net = DQN2(n_observations, n_actions).to(DEVICE)
policy_net.load_state_dict(checkpoint)
policy_net.train()

target_net = DQN2(n_observations, n_actions).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

print("✅ Pre-trained model loaded!")

replay_memory = ReplayMemory(150000)

def select_action(state):
    if np.random.rand() < epsilon:
        return torch.tensor([[env.action_space.sample()]], dtype=torch.long, device=DEVICE)
    else:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)

def shape_reward(original_reward, state, action, done):
    if not USE_REWARD_SHAPING:
        return original_reward
    
    shaped = original_reward
    pos_x, pos_y, vel_x, vel_y, angle, ang_vel, leg1, leg2 = state
    
    if done:
        if original_reward > 100:
            shaped += 20 * REWARD_SHAPING_INTENSITY
        elif original_reward < -50:
            shaped -= 50 * REWARD_SHAPING_INTENSITY
    
    near_ground = pos_y < 0.3 and (leg1 == 1 or leg2 == 1)
    if near_ground:
        velocity_magnitude = np.sqrt(vel_x**2 + vel_y**2)
        if velocity_magnitude > 0.5:
            shaped -= 2 * REWARD_SHAPING_INTENSITY
        
        if abs(pos_x) < 0.1:
            shaped += 1 * REWARD_SHAPING_INTENSITY
    
    return shaped

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True, weight_decay=2e-5)
criterion = nn.SmoothL1Loss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=200, verbose=True)

print("="*70)
print("🌬️  FINE-TUNING WITH WIND")
print(f"📊 Episodes: {NUM_EPISODES} | Batch: {BATCH_SIZE} | LR: {LR}")
print(f"🎯 Reward shaping: {USE_REWARD_SHAPING} (intensity={REWARD_SHAPING_INTENSITY})")
print("="*70)

for episode in range(NUM_EPISODES):
    if stop_training:
        break
    
    state, info = env.reset()
    state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    total_reward = 0.0
    
    for t in count():
        action = select_action(state_tensor)
        next_state, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        
        shaped_reward = shape_reward(reward, state, action.item(), done)
        
        reward_tensor = torch.tensor([shaped_reward], device=DEVICE)
        next_state_tensor = None
        if not done:
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        
        replay_memory.push(state_tensor, action, next_state_tensor, reward_tensor, done)
        
        state = next_state
        state_tensor = next_state_tensor
        total_reward += reward
        
        if len(replay_memory) >= BATCH_SIZE:
            transitions = replay_memory.sample(BATCH_SIZE)
            batch = Transition(*zip(*transitions))

            state_batch = torch.cat(batch.state).to(DEVICE)
            action_batch = torch.cat(batch.action).to(DEVICE)
            reward_batch = torch.cat(batch.reward).to(DEVICE)
            done_batch = torch.tensor(batch.done, device=DEVICE, dtype=torch.float32)

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=DEVICE, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(DEVICE)
            
            next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)

            if non_final_next_states.size(0) > 0:
                with torch.no_grad():
                    next_actions = policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                    next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_actions).squeeze(1)
            
            q_policy = policy_net(state_batch).gather(1, action_batch)
            q_target = reward_batch.squeeze() + (GAMMA * next_state_values * (1 - done_batch))

            loss = criterion(q_policy, q_target.unsqueeze(1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5)
            optimizer.step()
            
            for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                target_param.data.copy_(TAU * policy_param.data + (1 - TAU) * target_param.data)
        
        if done:
            reward_list.append(total_reward)
            
            if len(reward_list) >= 100:
                avg_reward = np.mean(reward_list[-100:])
                scheduler.step(avg_reward)
            
            if episode % 50 == 0:
                avg_100 = np.mean(reward_list[-100:]) if len(reward_list) >= 100 else np.mean(reward_list)
                std_100 = np.std(reward_list[-100:]) if len(reward_list) >= 100 else 0
                print(f"Ep {episode:5d} | R: {total_reward:7.2f} | Avg: {avg_100:6.2f}±{std_100:5.2f} | ε: {epsilon:.4f}")
            
            if episode % 500 == 0 and episode > 0:
                torch.save(policy_net.state_dict(), f"checks/finetune_ep{episode}.pth")
                print(f"💾 Checkpoint saved")
            
            if EARLY_STOPPING_ENABLED and episode > EARLY_STOPPING_STARTING_EPISODE and len(reward_list) >= 100:
                current_avg = np.mean(reward_list[-100:])
                if current_avg > best_reward + EARLY_STOPPING_THRESHOLD:
                    best_reward = current_avg
                    early_stopping_patience = INITIAL_PATIENCE
                    torch.save(policy_net.state_dict(), "models/fine_tuned_wind_BEST.pth")
                    print(f"💎 NEW BEST! Avg: {best_reward:.2f}")
                else:
                    early_stopping_patience -= 1
                    if early_stopping_patience <= 0:
                        print(f"⏹️  Early stopping at episode {episode}")
                        stop_training = True
            break
    
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

torch.save(policy_net.state_dict(), "models/fine_tuned_wind_final.pth")

final_avg = np.mean(reward_list[-100:]) if len(reward_list) >= 100 else np.mean(reward_list)
final_std = np.std(reward_list[-100:]) if len(reward_list) >= 100 else 0

print("\n" + "="*70)
print("🎉 FINE-TUNING COMPLETED!")
print("="*70)
print(f"📈 Best avg: {best_reward:.2f}")
print(f"📊 Final 100: {final_avg:.2f} ± {final_std:.2f}")
print("="*70)
env.close()