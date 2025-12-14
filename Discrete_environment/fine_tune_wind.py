import gymnasium as gym
from replay_memory import ReplayMemory, Transition
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dqn import DQN_heavy
import os

NUM_EPISODES = 10000
BATCH_SIZE = 128
GAMMA = 0.99
LR = 1e-5
TAU = 0.001

epsilon = 0.5
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9997

EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_THRESHOLD = 4
EARLY_STOPPING_STARTING_EPISODE = 6000
INITIAL_PATIENCE = 400
early_stopping_patience = INITIAL_PATIENCE
best_reward = -np.inf
stop_training = False
reward_list = []

WARMUP_EPISODES = 30

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Device for fine-tuning: {DEVICE}")

env = gym.make("LunarLander-v3", enable_wind=True)

n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n

base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "models", "128_best.pth")

print(f"📦 Loading pre-trained model from: {model_path}")

try:
    checkpoint = torch.load(model_path, map_location=DEVICE)

    from dqn import DQN as DQN_old  # Tu red original
    policy_net = DQN_old(n_observations, n_actions).to(DEVICE)
    policy_net.load_state_dict(checkpoint)
    
    # Target network
    target_net = DQN_old(n_observations, n_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    print("✅ Pre-trained model loaded successfully!")
    
except Exception as e:
    print(f"⚠️  Could not load pre-trained model: {e}")
    print("🔧 Training from scratch instead...")
    policy_net = DQN_heavy(n_observations, n_actions).to(DEVICE)
    target_net = DQN_heavy(n_observations, n_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

policy_net.train()

replay_memory = ReplayMemory(100000)

def select_action(state, epsilon_override=None):
    eps = epsilon_override if epsilon_override is not None else epsilon
    if np.random.rand() < eps:
        return torch.tensor([[env.action_space.sample()]], dtype=torch.long, device=DEVICE)
    else:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)

# Optimizer with weight decay for regularization
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True, weight_decay=1e-5)
criterion = nn.SmoothL1Loss()

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=150, verbose=True)

print("🌬️  Starting fine-tuning with WIND enabled...")
print(f"📊 Episodes: {NUM_EPISODES}, Batch size: {BATCH_SIZE}, LR: {LR}")
print(f"🎯 Initial epsilon: {epsilon} (higher than training for wind exploration)")

warmup_done = False

for episode in range(NUM_EPISODES):
    if stop_training:
        break
    
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    total_reward = 0.0
    
    if episode < WARMUP_EPISODES:
        epsilon_current = 0.8
    else:
        epsilon_current = epsilon
        if not warmup_done:
            print(f"✅ Warm-up completed! Buffer size: {len(replay_memory)}")
            warmup_done = True
    
    for t in count():
        action = select_action(state, epsilon_override=epsilon_current)
        observation, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        
        reward_tensor = torch.tensor([reward], device=DEVICE)
        
        next_state = None
        if not done:
            next_state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        
        replay_memory.push(state, action, next_state, reward_tensor, done)
        
        state = next_state
        total_reward += reward
        
        if episode >= WARMUP_EPISODES and len(replay_memory) >= BATCH_SIZE:
            transitions = replay_memory.sample(BATCH_SIZE)
            batch = Transition(*zip(*transitions))

            state_batch = torch.cat(batch.state).to(DEVICE)
            action_batch = torch.cat(batch.action).to(DEVICE)
            reward_batch = torch.cat(batch.reward).to(DEVICE)
            done_batch = torch.tensor(batch.done, device=DEVICE, dtype=torch.float32)

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=DEVICE, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(DEVICE)
            
            # Double DQN
            next_state_values_full = torch.zeros(BATCH_SIZE, device=DEVICE)

            if non_final_next_states.size(0) > 0:
                with torch.no_grad():
                    next_actions = policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                    all_q_values = target_net(non_final_next_states)
                    next_state_values_full[non_final_mask] = all_q_values.gather(1, next_actions).squeeze(1)
            
            q_policy = policy_net(state_batch).gather(1, action_batch)
            q_target = reward_batch.squeeze() + (GAMMA * next_state_values_full * (1 - done_batch))

            loss = criterion(q_policy, q_target.unsqueeze(1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Soft update target network
            for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                target_param.data.copy_(TAU * policy_param.data + (1 - TAU) * target_param.data)
        
        if done:
            reward_list.append(total_reward)
            
            if len(reward_list) >= 100:
                avg_reward = np.mean(reward_list[-100:])
                scheduler.step(avg_reward)
            
            if episode % 50 == 0:
                avg_100 = np.mean(reward_list[-100:]) if len(reward_list) >= 100 else np.mean(reward_list)
                print(f"Episode {episode:5d} | Reward: {total_reward:7.2f} | Avg(100): {avg_100:7.2f} | ε: {epsilon:.4f}")
            
            if EARLY_STOPPING_ENABLED and episode > EARLY_STOPPING_STARTING_EPISODE and len(reward_list) >= 100:
                current_avg = np.mean(reward_list[-100:])
                if current_avg > best_reward + EARLY_STOPPING_THRESHOLD:
                    best_reward = current_avg
                    early_stopping_patience = INITIAL_PATIENCE
                    torch.save(policy_net.state_dict(), "models/fine_tuned_wind_best.pth")
                    print(f"💾 New best fine-tuned model! Avg: {best_reward:.2f}")
                else:
                    early_stopping_patience -= 1
                    if early_stopping_patience <= 0:
                        print(f"⏹️  Early stopping at episode {episode}")
                        stop_training = True
            break
    
    if episode >= WARMUP_EPISODES:
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

torch.save(policy_net.state_dict(), "models/fine_tuned_wind_final.pth")
print("🎉 Fine-tuning completed!")
print(f"📈 Best average reward: {best_reward:.2f}")
print(f"📉 Final average reward (last 100): {np.mean(reward_list[-100:]):.2f}")
env.close()