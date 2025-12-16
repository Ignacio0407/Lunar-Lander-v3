import gymnasium as gym
from replay_memory import PrioritizedReplayMemory, Transition
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dqn import DQN_heavy
import torch.multiprocessing as mp
from collections import deque
import time

NUM_PARALLEL_ENVS = 24
NUM_EPISODES = 30000
BATCH_SIZE = 512
GAMMA = 0.99
LR = 5e-5
TAU = 0.003

epsilon = 1.0
EPSILON_MIN = 0.02
EPSILON_DECAY = 0.99975

EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_THRESHOLD = 3
EARLY_STOPPING_STARTING_EPISODE = 20000
INITIAL_PATIENCE = 1000
early_stopping_patience = INITIAL_PATIENCE
best_reward = -np.inf
stop_training = False
reward_list = []

WARMUP_EPISODES = 200
warmup_done = False

USE_PRIORITIZED_REPLAY = True
PRIORITIZED_REPLAY_ALPHA = 0.7
PRIORITIZED_REPLAY_BETA_START = 0.5
PRIORITIZED_REPLAY_BETA_END = 1.0

USE_REWARD_SHAPING = False
REWARD_SHAPING_WEIGHT = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {DEVICE}")
print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

def shape_reward_light(reward, done, state=None):
    """
    Reward shaping MUY LEVE para no condicionar el entrenamiento.
    Solo un pequeño empujón en la dirección correcta.
    """
    if not USE_REWARD_SHAPING:
        return reward
    
    shaped = reward

    if done and reward < -50:
        shaped += reward * REWARD_SHAPING_WEIGHT
    
    elif done and reward > 100:
        shaped += reward * REWARD_SHAPING_WEIGHT
    
    return shaped

def worker_process(worker_id, env_queue, experience_queue, stop_event, epsilon_value):
    """
    Proceso worker que ejecuta un entorno y recolecta experiencias.
    """
    env = gym.make("LunarLander-v3", enable_wind=True)
    
    while not stop_event.is_set():
        try:
            if not env_queue.empty():
                model_state = env_queue.get(timeout=0.1)
                if model_state is None:
                    break

            state, _ = env.reset()
            episode_reward = 0.0
            episode_data = []
            
            done = False
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

                eps = epsilon_value.value
                if np.random.rand() < eps:
                    action = env.action_space.sample()
                else:
                    action = env.action_space.sample() if np.random.rand() < 0.1 else np.random.randint(0, 4)
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Reward shaping leve
                shaped_reward = shape_reward_light(reward, done, state)
                
                episode_data.append((state, action, next_state if not done else None, shaped_reward, done))
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # Enviar experiencias al proceso principal
            experience_queue.put((worker_id, episode_reward, episode_data))
            
        except Exception as e:
            print(f"Worker {worker_id} error: {e}")
            break
    
    env.close()

class ParallelEnvManager:
    """
    Gestor simplificado para recolección paralela de experiencias.
    Para A100, usamos vectorización directa en GPU que es más eficiente.
    """
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.envs = [gym.make("LunarLander-v3", enable_wind=True) for _ in range(num_envs)]
        self.states = [env.reset()[0] for env in self.envs]
        self.episode_rewards = [0.0] * num_envs
        self.episode_lengths = [0] * num_envs
    
    def step(self, actions):
        """
        Ejecuta un paso en todos los entornos con las acciones dadas.
        actions: lista de acciones (una por entorno)
        """
        experiences = []
        completed_episodes = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            shaped_reward = shape_reward_light(reward, done, self.states[i])
            
            experiences.append({
                'state': self.states[i],
                'action': action,
                'next_state': next_state if not done else None,
                'reward': shaped_reward,
                'done': done,
                'env_id': i
            })
            
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
        """Retorna todos los estados actuales como batch tensor"""
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

if torch.cuda.is_available():
    policy_net = torch.compile(policy_net)
    target_net = torch.compile(target_net)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

replay_memory = PrioritizedReplayMemory(200000, alpha=PRIORITIZED_REPLAY_ALPHA)

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True, weight_decay=2e-5)
criterion = nn.SmoothL1Loss(reduction='none')

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=400, verbose=True)

def select_actions_batch(states_batch, epsilon_val):
    """
    Selecciona acciones para un batch de estados (vectorizado).
    """
    if np.random.rand() < epsilon_val:
        # Exploration
        return [np.random.randint(0, n_actions) for _ in range(len(states_batch))]
    else:
        # Explotation
        with torch.no_grad():
            q_values = policy_net(states_batch)
            actions = q_values.max(1).indices.cpu().numpy()
        return actions.tolist()

def optimize_model_batch():
    """Optimización con batch grande para A100"""
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
    
    # Double DQN
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
print("🔥 PARALLEL TRAINING ON A100")
print(f"🌐 Parallel environments: {NUM_PARALLEL_ENVS}")
print(f"📊 Episodes: {NUM_EPISODES} | Batch: {BATCH_SIZE} | LR: {LR}")
print(f"💾 Replay buffer: {replay_memory.memory.maxlen}")
print(f"🎯 Reward shaping: {USE_REWARD_SHAPING} (weight={REWARD_SHAPING_WEIGHT})")
print("="*70)

total_episodes = 0
steps_per_optimization = 4  # Optimize each N steps
step_counter = 0
success_count = 0
crash_count = 0

start_time = time.time()

while total_episodes < NUM_EPISODES and not stop_training:
    states_batch = env_manager.get_states()
    
    epsilon_current = epsilon if total_episodes >= WARMUP_EPISODES else 1.0
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
            
            print(f"Ep {total_episodes:6d} | R: {ep['reward']:7.2f} | "
                  f"Avg: {avg_100:6.2f}±{std_100:5.2f} | "
                  f"ε: {epsilon:.5f} | Success: {success_rate:4.1f}% | "
                  f"{eps_per_hour:.0f} ep/h")
        
        if total_episodes % 1000 == 0 and total_episodes > 0:
            torch.save(policy_net.state_dict(), f"checkpoints/checkpoint_ep{total_episodes}.pth")
            print(f"💾 Checkpoint saved at episode {total_episodes}")
        
        # Early stopping
        if EARLY_STOPPING_ENABLED and total_episodes > EARLY_STOPPING_STARTING_EPISODE and len(reward_list) >= 100:
            current_avg = np.mean(reward_list[-100:])
            if current_avg > best_reward + EARLY_STOPPING_THRESHOLD:
                best_reward = current_avg
                early_stopping_patience = INITIAL_PATIENCE
                torch.save(policy_net.state_dict(), "models/best_parallel_wind.pth")
                print(f"💎 New best! Avg: {best_reward:.2f}")
            else:
                early_stopping_patience -= 1
                if early_stopping_patience <= 0:
                    print(f"\n⏹️  Early stopping at episode {total_episodes}")
                    stop_training = True
                    break
    
    # Optimize net each N steps
    step_counter += 1
    if step_counter >= steps_per_optimization and total_episodes >= WARMUP_EPISODES:
        step_counter = 0
        loss = optimize_model_batch()
        
        # Soft update target network
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(TAU * policy_param.data + (1 - TAU) * target_param.data)
    
    # Decay epsilon
    if total_episodes >= WARMUP_EPISODES:
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    
    # Scheduler update
    if len(reward_list) >= 100 and total_episodes % 100 == 0:
        scheduler.step(np.mean(reward_list[-100:]))

env_manager.close()
torch.save(policy_net.state_dict(), "models/ddqn_parallel_wind_final.pth")

final_avg = np.mean(reward_list[-100:])
final_std = np.std(reward_list[-100:])
final_success = (sum(1 for r in reward_list[-100:] if r > 0) / 100) * 100
total_time = time.time() - start_time

print("\n" + "="*70)
print("🎉 PARALLEL TRAINING COMPLETED!")
print("="*70)
print(f"⏱️  Total time: {total_time/3600:.2f} hours ({total_episodes/(total_time/3600):.0f} ep/h)")
print(f"📈 Best average: {best_reward:.2f}")
print(f"📊 Final 100 eps: {final_avg:.2f} ± {final_std:.2f}")
print(f"✨ Success rate: {final_success:.1f}%")
print(f"📉 Overall success: {(success_count/total_episodes)*100:.1f}%")
print("="*70)