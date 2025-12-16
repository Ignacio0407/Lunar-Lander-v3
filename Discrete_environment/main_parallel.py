import gymnasium as gym
from replay_memory import ReplayMemory, PrioritizedReplayMemory, Transition
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dqn import DQN_heavy
import time

NUM_PARALLEL_ENVS = 24
NUM_EPISODES = 20000
BATCH_SIZE = 512
GAMMA = 0.99
LR = 3e-5
TAU = 0.005

epsilon = 1.0
EPSILON_MIN = 0.005
EPSILON_DECAY = 0.9993

EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_THRESHOLD = 1
EARLY_STOPPING_STARTING_EPISODE = 15000
INITIAL_PATIENCE = 800
early_stopping_patience = INITIAL_PATIENCE
best_reward = -np.inf
stop_training = False
reward_list = []

WARMUP_EPISODES = 100
warmup_done = False

USE_PRIORITIZED_REPLAY = True
PRIORITIZED_REPLAY_ALPHA = 0.7
PRIORITIZED_REPLAY_BETA_START = 0.5
PRIORITIZED_REPLAY_BETA_END = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

class ParallelEnvManager:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.envs = [gym.make("LunarLander-v3", enable_wind=False) for _ in range(num_envs)]
        self.reset_all()
    
    def reset_all(self):
        self.states = []
        self.episode_rewards = [0.0] * self.num_envs
        self.episode_lengths = [0] * self.num_envs
        self.dones = [False] * self.num_envs
        
        for env in self.envs:
            state, _ = env.reset()
            self.states.append(state)
    
    def step(self, actions):
        experiences = []
        completed_episodes = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            if self.dones[i]:
                continue
                
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            experiences.append({'state': self.states[i], 'action': action, 'next_state': next_state if not done else None,
                'reward': reward, 'done': done, 'env_id': i})
            
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
    
    def get_states_tensor(self):
        return torch.tensor(np.array(self.states), dtype=torch.float32, device=DEVICE)
    
    def close(self):
        for env in self.envs:
            env.close()

env_manager = ParallelEnvManager(NUM_PARALLEL_ENVS)

n_observations = 8
n_actions = 4

policy_net = DQN_heavy(n_observations, n_actions).to(DEVICE)
target_net = DQN_heavy(n_observations, n_actions).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

if USE_PRIORITIZED_REPLAY:
    replay_memory = PrioritizedReplayMemory(200000, alpha=PRIORITIZED_REPLAY_ALPHA)
else:
    replay_memory = ReplayMemory(200000)

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True, weight_decay=1e-5)
criterion = nn.SmoothL1Loss(reduction='none')

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=200)

def select_actions_batch(states_batch, epsilon_val):
    batch_size = states_batch.shape[0]
    
    if np.random.rand() < epsilon_val:
        return [np.random.randint(0, n_actions) for _ in range(batch_size)]
    else:
        with torch.no_grad():
            q_values = policy_net(states_batch)
            actions = q_values.max(1).indices.cpu().numpy()
        return actions.tolist()

def optimize_model_batch():
    if len(replay_memory) < BATCH_SIZE:
        return None
    
    if USE_PRIORITIZED_REPLAY:
        beta = PRIORITIZED_REPLAY_BETA_START + (PRIORITIZED_REPLAY_BETA_END - PRIORITIZED_REPLAY_BETA_START) * \
               min(1.0, (total_episodes - WARMUP_EPISODES) / (NUM_EPISODES - WARMUP_EPISODES))
        transitions, indices, weights = replay_memory.sample(BATCH_SIZE, beta)
        weights = torch.FloatTensor(weights).to(DEVICE).unsqueeze(1)
    else:
        transitions = replay_memory.sample(BATCH_SIZE)
        weights = torch.ones(BATCH_SIZE, 1).to(DEVICE)
    
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
    loss = (td_errors * weights).mean()
    
    if USE_PRIORITIZED_REPLAY:
        priorities = td_errors.detach().cpu().numpy().flatten()
        replay_memory.update_priorities(indices, np.abs(priorities))
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()

print("="*70)
print(f"🔥 PARALLEL TRAINING ON {DEVICE}")
print(f"🌐 Parallel environments: {NUM_PARALLEL_ENVS}")
print(f"📊 Episodes: {NUM_EPISODES} | Batch size: {BATCH_SIZE} | LR: {LR}")
print(f"💾 Replay buffer: {replay_memory.memory.maxlen}")
print(f"🔍 Prioritized Replay: {USE_PRIORITIZED_REPLAY}")
print("="*70)

total_episodes = 0
steps_per_optimization = 4
step_counter = 0
start_time = time.time()

while total_episodes < NUM_EPISODES and not stop_training:
    states_batch = env_manager.get_states_tensor()
    
    epsilon_current = epsilon if total_episodes >= WARMUP_EPISODES else 1.0
    actions = select_actions_batch(states_batch, epsilon_current)
    
    experiences, completed = env_manager.step(actions)
    
    for exp in experiences:
        replay_memory.push(exp['state'], exp['action'], exp['next_state'], exp['reward'], exp['done'])
    
    for ep in completed:
        total_episodes += 1
        reward_list.append(ep['reward'])
        
        if total_episodes % 50 == 0:
            avg_100 = np.mean(reward_list[-100:]) if len(reward_list) >= 100 else np.mean(reward_list)
            elapsed = time.time() - start_time
            eps_per_hour = total_episodes / (elapsed / 3600) if elapsed > 0 else 0
            
            print(f"Episode {total_episodes:5d} | Reward: {ep['reward']:7.2f} | Avg(100): {avg_100:7.2f} | "
                  f"ε: {epsilon:.4f} | Buffer: {len(replay_memory)} | {eps_per_hour:.0f} ep/h")
        
        if EARLY_STOPPING_ENABLED and total_episodes > EARLY_STOPPING_STARTING_EPISODE and len(reward_list) >= 100:
            current_avg = np.mean(reward_list[-100:])
            if current_avg > best_reward + EARLY_STOPPING_THRESHOLD:
                best_reward = current_avg
                early_stopping_patience = INITIAL_PATIENCE
                torch.save(policy_net.state_dict(), "models/best_parallel.pth")
                print(f"💎 New best! Avg: {best_reward:.2f}")
            else:
                early_stopping_patience -= 1
                if early_stopping_patience <= 0:
                    print(f"⏹️  Early stopping triggered at episode {total_episodes}")
                    stop_training = True
                    break
        
        if total_episodes % 100 == 0 and total_episodes > 0:
            torch.save(policy_net.state_dict(), f"checks/parallel_checkpoint_ep{total_episodes}.pth")
            print(f"💾 Checkpoint saved at episode {total_episodes}")
    
    step_counter += 1
    if step_counter >= steps_per_optimization and total_episodes >= WARMUP_EPISODES:
        step_counter = 0
        loss = optimize_model_batch()
        
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(TAU * policy_param.data + (1 - TAU) * target_param.data)
    
    if total_episodes >= WARMUP_EPISODES:
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    
    if len(reward_list) >= 100 and total_episodes % 100 == 0:
        scheduler.step(np.mean(reward_list[-100:]))

env_manager.close()
torch.save(policy_net.state_dict(), "models/ddqn_parallel_final.pth")

total_time = time.time() - start_time
final_avg = np.mean(reward_list[-100:]) if len(reward_list) >= 100 else np.mean(reward_list)
final_std = np.std(reward_list[-100:]) if len(reward_list) >= 100 else np.std(reward_list)

print("\n" + "="*70)
print("🎉 PARALLEL TRAINING COMPLETED!")
print("="*70)
print(f"⏱️  Total time: {total_time/3600:.2f} hours ({total_episodes/(total_time/3600):.0f} ep/h)")
print(f"📈 Best average reward: {best_reward:.2f}")
print(f"📉 Final average reward (last 100): {final_avg:.2f} ± {final_std:.2f}")
print("="*70)