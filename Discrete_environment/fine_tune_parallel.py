import gymnasium as gym
from replay_memory import PrioritizedReplayMemory, Transition
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dqn import DQN_heavy
import os
import time

NUM_PARALLEL_ENVS = 20
NUM_EPISODES = 15000
BATCH_SIZE = 512
GAMMA = 0.99
LR = 1e-5
TAU = 0.002

epsilon = 0.6
EPSILON_MIN = 0.02
EPSILON_DECAY = 0.9998

EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_THRESHOLD = 3
EARLY_STOPPING_STARTING_EPISODE = 10000
INITIAL_PATIENCE = 800
early_stopping_patience = INITIAL_PATIENCE
best_reward = -np.inf
stop_training = False
reward_list = []

WARMUP_EPISODES = 150

USE_PRIORITIZED_REPLAY = True
PRIORITIZED_REPLAY_ALPHA = 0.7
PRIORITIZED_REPLAY_BETA_START = 0.6
PRIORITIZED_REPLAY_BETA_END = 1.0

USE_REWARD_SHAPING = False
REWARD_SHAPING_WEIGHT = 0.03

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Fine-tuning on: {DEVICE}")
print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

def shape_reward_light(reward, done):
    if not USE_REWARD_SHAPING:
        return reward
    shaped = reward
    if done and reward < -50:
        shaped += reward * REWARD_SHAPING_WEIGHT
    elif done and reward > 100:
        shaped += reward * REWARD_SHAPING_WEIGHT
    return shaped

class ParallelEnvManager:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.envs = [gym.make("LunarLander-v3", enable_wind=True) for _ in range(num_envs)]
        self.states = [env.reset()[0] for env in self.envs]
        self.episode_rewards = [0.0] * num_envs
        self.episode_lengths = [0] * num_envs
    
    def step(self, actions):
        experiences = []
        completed_episodes = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            shaped_reward = shape_reward_light(reward, done)
            
            experiences.append({'state': self.states[i], 'action': action, 'next_state': next_state if not done else None,
                'reward': shaped_reward, 'done': done, 'env_id': i})
            
            self.episode_rewards[i] += reward
            self.episode_lengths[i] += 1
            
            if done:
                completed_episodes.append({
                    'env_id': i,
                    'reward': self.episode_rewards[i],
                    'length': self.episode_lengths[i]
                })
                self.states[i], _ = env.reset()
                self.episode_rewards[i] = 0.0
                self.episode_lengths[i] = 0
            else:
                self.states[i] = next_state
        
        return experiences, completed_episodes
    
    def get_states(self):
        return torch.tensor(np.array(self.states), dtype=torch.float32, device=DEVICE)
    
    def close(self):
        for env in self.envs:
            env.close()

env_manager = ParallelEnvManager(NUM_PARALLEL_ENVS)

n_observations = 8
n_actions = 4

base_dir = os.path.dirname(__file__)
no_wind_model_path = os.path.join(base_dir, "models", "best_no_wind.pth")

print(f"📦 Loading pre-trained model: {no_wind_model_path}")

try:
    checkpoint = torch.load(no_wind_model_path, map_location=DEVICE)
    policy_net = DQN_heavy(n_observations, n_actions).to(DEVICE)
    policy_net.load_state_dict(checkpoint)
    policy_net.train()
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print(f"⚠️  Model not found, training from scratch")
    policy_net = DQN_heavy(n_observations, n_actions).to(DEVICE)

target_net = DQN_heavy(n_observations, n_actions).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

if torch.cuda.is_available():
    policy_net = torch.compile(policy_net)
    target_net = torch.compile(target_net)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

replay_memory = PrioritizedReplayMemory(150000, alpha=PRIORITIZED_REPLAY_ALPHA)

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True, weight_decay=2e-5)
criterion = nn.SmoothL1Loss(reduction='none')
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=350, verbose=True)

def select_actions_batch(states_batch, epsilon_val):
    if np.random.rand() < epsilon_val:
        return [np.random.randint(0, n_actions) for _ in range(len(states_batch))]
    else:
        with torch.no_grad():
            q_values = policy_net(states_batch)
            actions = q_values.max(1).indices.cpu().numpy()
        return actions.tolist()

def optimize_model_batch():
    if len(replay_memory) < BATCH_SIZE:
        return None
    
    beta = PRIORITIZED_REPLAY_BETA_START + \
           (PRIORITIZED_REPLAY_BETA_END - PRIORITIZED_REPLAY_BETA_START) * \
           min(1.0, (total_episodes - WARMUP_EPISODES) / (NUM_EPISODES - WARMUP_EPISODES))
    
    transitions, indices, weights = replay_memory.sample(BATCH_SIZE, beta)
    weights = torch.FloatTensor(weights).to(DEVICE).unsqueeze(1)
    
    batch = Transition(*zip(*transitions))
    
    state_batch = torch.cat([torch.tensor(s, dtype=torch.float32).unsqueeze(0) for s in batch.state]).to(DEVICE)
    action_batch = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).to(DEVICE)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(DEVICE)
    done_batch = torch.tensor(batch.done, dtype=torch.float32).to(DEVICE)
    
    non_final_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool, device=DEVICE)
    non_final_next_states = torch.cat([
        torch.tensor(s, dtype=torch.float32).unsqueeze(0) 
        for s in batch.next_state if s is not None
    ]).to(DEVICE)
    
    next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
    
    if non_final_next_states.size(0) > 0:
        with torch.no_grad():
            next_actions = policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
            next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_actions).squeeze(1)
    
    q_policy = policy_net(state_batch).gather(1, action_batch)
    q_target = reward_batch + (GAMMA * next_state_values * (1 - done_batch))
    
    td_errors = criterion(q_policy, q_target.unsqueeze(1))
    td_errors_clipped = torch.clamp(td_errors, -20, 20)
    loss = (td_errors_clipped * weights).mean()
    
    priorities = td_errors.detach().cpu().numpy().flatten()
    replay_memory.update_priorities(indices, np.abs(priorities))
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5)
    optimizer.step()
    
    return loss.item()

print("="*70)
print("🌬️  PARALLEL FINE-TUNING WITH WIND")
print(f"🌐 Parallel environments: {NUM_PARALLEL_ENVS}")
print(f"📊 Episodes: {NUM_EPISODES} | Batch: {BATCH_SIZE} | LR: {LR}")
print(f"🎯 Target time: 5-6 hours")
print("="*70)

total_episodes = 0
steps_per_optimization = 4
step_counter = 0
success_count = 0
crash_count = 0

start_time = time.time()

while total_episodes < NUM_EPISODES and not stop_training:
    states_batch = env_manager.get_states()
    
    # Warm-up con exploración alta
    if total_episodes < WARMUP_EPISODES:
        epsilon_current = 0.8
    else:
        epsilon_current = epsilon
    
    actions = select_actions_batch(states_batch, epsilon_current)
    experiences, completed = env_manager.step(actions)
    
    for exp in experiences:
        replay_memory.push(exp['state'], exp['action'], exp['next_state'], exp['reward'], exp['done'])
    
    for ep in completed:
        total_episodes += 1
        reward_list.append(ep['reward'])
        
        if ep['reward'] > 0:
            success_count += 1
        if ep['reward'] < -100:
            crash_count += 1
        
        if total_episodes % 50 == 0:
            avg_100 = np.mean(reward_list[-100:]) if len(reward_list) >= 100 else np.mean(reward_list)
            std_100 = np.std(reward_list[-100:]) if len(reward_list) >= 100 else np.std(reward_list)
            success_rate = (success_count / total_episodes) * 100
            elapsed = time.time() - start_time
            eps_per_hour = total_episodes / (elapsed / 3600)
            eta_hours = (NUM_EPISODES - total_episodes) / eps_per_hour if eps_per_hour > 0 else 0
            
            print(f"Ep {total_episodes:6d} | R: {ep['reward']:7.2f} | "
                  f"Avg: {avg_100:6.2f}±{std_100:5.2f} | "
                  f"ε: {epsilon:.5f} | Success: {success_rate:4.1f}% | "
                  f"{eps_per_hour:.0f} ep/h | ETA: {eta_hours:.1f}h")
        
        if total_episodes % 1000 == 0 and total_episodes > 0:
            torch.save(policy_net.state_dict(), f"checkpoints/finetune_ep{total_episodes}.pth")
            print(f"💾 Checkpoint saved")
        
        if EARLY_STOPPING_ENABLED and total_episodes > EARLY_STOPPING_STARTING_EPISODE and len(reward_list) >= 100:
            current_avg = np.mean(reward_list[-100:])
            if current_avg > best_reward + EARLY_STOPPING_THRESHOLD:
                best_reward = current_avg
                early_stopping_patience = INITIAL_PATIENCE
                torch.save(policy_net.state_dict(), "models/best_finetuned_parallel.pth")
                print(f"💎 New best fine-tuned! Avg: {best_reward:.2f}")
            else:
                early_stopping_patience -= 1
                if early_stopping_patience <= 0:
                    print(f"\n⏹️  Early stopping at episode {total_episodes}")
                    stop_training = True
                    break
    
    step_counter += 1
    if step_counter >= steps_per_optimization and total_episodes >= WARMUP_EPISODES:
        step_counter = 0
        optimize_model_batch()
        
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(TAU * policy_param.data + (1 - TAU) * target_param.data)
    
    if total_episodes >= WARMUP_EPISODES:
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    
    if len(reward_list) >= 100 and total_episodes % 100 == 0:
        scheduler.step(np.mean(reward_list[-100:]))

env_manager.close()
torch.save(policy_net.state_dict(), "models/fine_tuned_parallel_final.pth")

final_avg = np.mean(reward_list[-100:])
final_std = np.std(reward_list[-100:])
final_success = (sum(1 for r in reward_list[-100:] if r > 0) / 100) * 100
total_time = time.time() - start_time

print("\n" + "="*70)
print("🎉 FINE-TUNING COMPLETED!")
print("="*70)
print(f"⏱️  Total time: {total_time/3600:.2f} hours")
print(f"📈 Best average: {best_reward:.2f}")
print(f"📊 Final 100: {final_avg:.2f} ± {final_std:.2f}")
print(f"✨ Success: {final_success:.1f}%")
print("="*70)