import gymnasium as gym
from replay_memory import PrioritizedReplayMemory, Transition
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dqn import DQN_heavy
import time
from collections import deque

"""
CURRICULUM LEARNING PARALELO: Entrenar progresivamente con múltiples entornos
- Fase 1 (0-4000): Sin viento
- Fase 2 (4000-8000): Viento débil (wind_power=5)
- Fase 3 (8000-12000): Viento normal (wind_power=15, default)
"""

NUM_PARALLEL_ENVS = 24
NUM_EPISODES = 12000
BATCH_SIZE = 512
GAMMA = 0.99
LR = 3e-5
TAU = 0.005

epsilon = 1.0
EPSILON_MIN = 0.005
EPSILON_DECAY = 0.9993

WARMUP_EPISODES = 100
USE_PRIORITIZED_REPLAY = True

CURRICULUM_STAGES = [
    {"name": "No Wind", "start": 0, "end": 4000, "wind": False, "wind_power": 0},
    {"name": "Weak Wind", "start": 4000, "end": 8000, "wind": True, "wind_power": 5.0},
    {"name": "Normal Wind", "start": 8000, "end": 12000, "wind": True, "wind_power": 15.0},
]

current_stage = 0
reward_list = []
stage_rewards = {stage["name"]: [] for stage in CURRICULUM_STAGES}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

class ParallelEnvManager:
    """
    Gestor de múltiples entornos paralelos para recolección eficiente de experiencias.
    """
    def __init__(self, num_envs, stage_config):
        self.num_envs = num_envs
        self.stage_config = stage_config
        self.envs = [
            gym.make("LunarLander-v3", 
                    enable_wind=stage_config["wind"], 
                    wind_power=stage_config["wind_power"]) 
            for _ in range(num_envs)
        ]
        self.reset_all()
    
    def reset_all(self):
        """Reinicia todos los entornos y estados"""
        self.states = []
        self.episode_rewards = [0.0] * self.num_envs
        self.episode_lengths = [0] * self.num_envs
        self.dones = [False] * self.num_envs
        
        for env in self.envs:
            state, _ = env.reset()
            self.states.append(state)
    
    def step(self, actions):
        """
        Ejecuta un paso en todos los entornos con las acciones dadas.
        Retorna experiencias y episodios completados.
        """
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
                completed_episodes.append({'env_id': i, 'reward': self.episode_rewards[i], 'length': self.episode_lengths[i],
                    'stage': current_stage})

                self.states[i], _ = env.reset()
                self.episode_rewards[i] = 0.0
                self.episode_lengths[i] = 0
                self.dones[i] = False
            else:
                self.states[i] = next_state
        
        return experiences, completed_episodes
    
    def get_states_tensor(self):
        """Retorna todos los estados actuales como tensor batch"""
        return torch.tensor(np.array(self.states), dtype=torch.float32, device=DEVICE)
    
    def update_stage(self, new_stage_config):
        """Actualiza todos los entornos a una nueva etapa del curriculum"""
        for env in self.envs:
            env.close()
        
        self.stage_config = new_stage_config
        self.envs = [
            gym.make("LunarLander-v3", 
                    enable_wind=new_stage_config["wind"], 
                    wind_power=new_stage_config["wind_power"]) 
            for _ in range(self.num_envs)
        ]
        self.reset_all()
    
    def close(self):
        for env in self.envs:
            env.close()

env_manager = ParallelEnvManager(NUM_PARALLEL_ENVS, CURRICULUM_STAGES[0])

n_observations = 8
n_actions = 4

policy_net = DQN_heavy(n_observations, n_actions).to(DEVICE)
target_net = DQN_heavy(n_observations, n_actions).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Comentar las líneas que causaban el error de compilación
# if torch.cuda.is_available():
#     policy_net = torch.compile(policy_net)
#     target_net = torch.compile(target_net)
#     torch.backends.cudnn.benchmark = True
#     torch.set_float32_matmul_precision('high')

replay_memory = PrioritizedReplayMemory(200000)

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True, weight_decay=1e-4)
criterion = nn.SmoothL1Loss(reduction='none')

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=200)

def select_actions_batch(states_batch, epsilon_val):
    """
    Selecciona acciones para un batch de estados (vectorizado).
    """
    batch_size = states_batch.shape[0]
    
    if np.random.rand() < epsilon_val:
        # Exploration
        return [np.random.randint(0, n_actions) for _ in range(batch_size)]
    else:
        # Explotation
        with torch.no_grad():
            q_values = policy_net(states_batch)
            actions = q_values.max(1).indices.cpu().numpy()
        return actions.tolist()

def optimize_model_batch():
    """Optimización con batch grande para mejor rendimiento en GPU"""
    if len(replay_memory) < BATCH_SIZE:
        return None
    
    beta = 0.5 + 0.5 * min(1.0, total_episodes / NUM_EPISODES)
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
    
    # Double DQN
    next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
    
    if non_final_next_states.size(0) > 0:
        with torch.no_grad():
            next_actions = policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
            next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_actions).squeeze(1)
    
    q_policy = policy_net(state_batch).gather(1, action_batch)
    q_target = reward_batch + (GAMMA * next_state_values * (1 - done_batch))
    
    td_errors = criterion(q_policy, q_target.unsqueeze(1))
    loss = (td_errors * weights).mean()
    
    priorities = td_errors.detach().cpu().numpy().flatten()
    replay_memory.update_priorities(indices, np.abs(priorities))
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5)
    optimizer.step()
    
    return loss.item()

print("="*70)
print(f"🔥 PARALLEL CURRICULUM LEARNING ON {DEVICE}")
print(f"🌐 Parallel environments: {NUM_PARALLEL_ENVS}")
print(f"📊 Episodes: {NUM_EPISODES} | Batch: {BATCH_SIZE} | LR: {LR}")
print(f"💾 Replay buffer: {replay_memory.memory.maxlen}")
print("="*70)
for stage in CURRICULUM_STAGES:
    wind_str = f"Wind={stage['wind_power']}" if stage['wind'] else "No Wind"
    print(f"  Stage {CURRICULUM_STAGES.index(stage)+1}: Episodes {stage['start']}-{stage['end']} | {wind_str}")
print("="*70)

total_episodes = 0
steps_per_optimization = 4  # Optimize each N steps
step_counter = 0
warmup_done = False
stage_episode_counts = {stage["name"]: 0 for stage in CURRICULUM_STAGES}

start_time = time.time()

while total_episodes < NUM_EPISODES:
    for i, stage in enumerate(CURRICULUM_STAGES):
        if stage["start"] <= total_episodes < stage["end"] and i != current_stage:
            current_stage = i
            stage_name = stage["name"]
            print(f"\n🎯 STAGE {i+1}: {stage_name} (wind_power={stage['wind_power']})")
            print(f"   Total Episodes: {total_episodes} | Buffer size: {len(replay_memory)}")
            print(f"   Epsilon: {epsilon:.5f} | Episodes in stage: {stage_episode_counts[stage_name]}\n")
            
            env_manager.update_stage(stage)

            epsilon = min(0.3, epsilon * 1.5)
            
            torch.save(policy_net.state_dict(), f"models/parallel_curriculum_stage{i}.pth")
    
    states_batch = env_manager.get_states_tensor()
    
    epsilon_current = epsilon if total_episodes >= WARMUP_EPISODES else 1.0
    actions = select_actions_batch(states_batch, epsilon_current)
    
    experiences, completed = env_manager.step(actions)

    for exp in experiences:
        replay_memory.push(exp['state'], exp['action'], exp['next_state'], exp['reward'], exp['done'])

    for ep in completed:
        total_episodes += 1
        stage_name = CURRICULUM_STAGES[current_stage]["name"]
        stage_episode_counts[stage_name] += 1
        
        reward_list.append(ep['reward'])
        stage_rewards[stage_name].append(ep['reward'])
        
        if total_episodes % 50 == 0:
            current_stage_avg = np.mean(stage_rewards[stage_name][-50:]) if len(stage_rewards[stage_name]) >= 50 else np.mean(stage_rewards[stage_name])
            overall_avg = np.mean(reward_list[-100:]) if len(reward_list) >= 100 else np.mean(reward_list)
            elapsed = time.time() - start_time
            eps_per_hour = total_episodes / (elapsed / 3600) if elapsed > 0 else 0
            
            print(f"Ep {total_episodes:5d} [{stage_name}] | R: {ep['reward']:7.2f} | "
                  f"Stage Avg(50): {current_stage_avg:7.2f} | "
                  f"Overall Avg(100): {overall_avg:7.2f} | "
                  f"ε: {epsilon:.5f} | {eps_per_hour:.0f} ep/h")

        if total_episodes % 1000 == 0 and total_episodes > 0:
            torch.save(policy_net.state_dict(), f"curr_checks/parallel_curriculum_ep{total_episodes}.pth")
            print(f"💾 Checkpoint saved at episode {total_episodes}")
    
    step_counter += 1
    if step_counter >= steps_per_optimization and total_episodes >= WARMUP_EPISODES:
        step_counter = 0
        loss = optimize_model_batch()
        
        # Target network soft update
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(TAU * policy_param.data + (1 - TAU) * target_param.data)
 
    if total_episodes >= WARMUP_EPISODES:
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    if len(reward_list) >= 100 and total_episodes % 100 == 0:
        scheduler.step(np.mean(reward_list[-100:]))

env_manager.close()
torch.save(policy_net.state_dict(), "models/parallel_curriculum_final.pth")

total_time = time.time() - start_time
final_avg = np.mean(reward_list[-100:]) if len(reward_list) >= 100 else np.mean(reward_list)
final_std = np.std(reward_list[-100:]) if len(reward_list) >= 100 else np.std(reward_list)

print("\n" + "="*70)
print("🎉 PARALLEL CURRICULUM TRAINING COMPLETED!")
print("="*70)
print(f"⏱️  Total time: {total_time/3600:.2f} hours ({total_episodes/(total_time/3600):.0f} ep/h)")
print(f"📊 Final 100 episodes: {final_avg:.2f} ± {final_std:.2f}")
print("\nSTAGE STATISTICS:")
print("-"*70)
for stage_name, rewards in stage_rewards.items():
    if rewards:
        avg = np.mean(rewards)
        std = np.std(rewards)
        count = len(rewards)
        print(f"{stage_name:15s} | Avg: {avg:7.2f} ± {std:5.2f} | Episodes: {count:4d} | % Total: {count/total_episodes*100:5.1f}%")
print("="*70)