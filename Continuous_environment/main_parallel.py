from dqn import DQN
from prioritized_replay_memory import PrioritizedReplayMemory, Transition, DEVICE
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from preprocessing import SkipFrame
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
from gymnasium.wrappers import FrameStackObservation as FrameStack
import time
import os
from collections import deque

def select_available_gpu():
    for gpu_id in range(4):  # GPUs 0-4
        try:
            torch.cuda.set_device(gpu_id)
            test_tensor = torch.zeros(1000, 1000, device=f'cuda:{gpu_id}')
            del test_tensor
            torch.cuda.empty_cache()
            return gpu_id
        except RuntimeError:
            continue
    raise RuntimeError("No GPU available with sufficient memory")

DEVICE = torch.device(f"cuda:{select_available_gpu()}" if torch.cuda.is_available() else "cpu")

NUM_PARALLEL_ENVS = 32
NUM_EPISODES = 20000
BATCH_SIZE = 1024
GAMMA = 0.99
LR = 3e-4
TAU = 0.01

epsilon = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9995

EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_THRESHOLD = 1.5
EARLY_STOPPING_STARTING_EPISODE = 5000
INITIAL_PATIENCE = 300
early_stopping_patience = INITIAL_PATIENCE
best_reward = -250.
stop_training = False
reward_list = []

print(f"🚀 Using device: {DEVICE}")
print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

class ParallelEnvManager:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.envs = []
        self.states = []
        self.episode_rewards = [0.0] * num_envs
        self.episode_lengths = [0] * num_envs
        
        print(f"🌐 Initializing {num_envs} parallel environments...")
        
        for i in range(num_envs):
            env = gym.make("CarRacing-v3", continuous=False, domain_randomize=False)
            env = SkipFrame(env, skip=4)
            env = GrayscaleObservation(env, keep_dim=False)
            env = ResizeObservation(env, shape=(84, 84))
            env = FrameStack(env, 4)
            self.envs.append(env)
            state, _ = env.reset()
            self.states.append(state)
            
            if i == 0:
                print(f"✅ Environment shape: {state.shape}")
        
        print(f"✅ All {num_envs} environments initialized!")
    
    def step(self, actions):
        """Ejecuta un paso en todos los entornos"""
        next_states = []
        rewards = []
        dones = []
        infos = []
        completed_episodes = []
        
        for i, action in enumerate(actions):
            next_state, reward, terminated, truncated, info = self.envs[i].step(action)
            done = terminated or truncated
            
            self.episode_rewards[i] += reward
            self.episode_lengths[i] += 1
            
            if done:
                completed_episodes.append({
                    'env_id': i,
                    'reward': self.episode_rewards[i],
                    'length': self.episode_lengths[i]
                })
                next_state, _ = self.envs[i].reset()
                self.episode_rewards[i] = 0.0
                self.episode_lengths[i] = 0
            
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            self.states[i] = next_state
        
        return next_states, rewards, dones, infos, completed_episodes
    
    def get_states_batch(self):
        """Retorna batch de estados como tensor"""
        # states tiene shape (num_envs, 4, 84, 84)
        states_array = np.array(self.states)
        return torch.tensor(states_array, dtype=torch.float32, device=DEVICE)
    
    def reset(self):
        for i in range(self.num_envs):
            state, _ = self.envs[i].reset()
            self.states[i] = state
        return self.states
    
    def close(self):
        for env in self.envs:
            env.close()

# ============== INICIALIZACIÓN ==============
env_manager = ParallelEnvManager(NUM_PARALLEL_ENVS)

n_actions = env_manager.envs[0].action_space.n
n_observations = env_manager.envs[0].observation_space.shape[0]
print(f"🎮 Number of actions: {n_actions}")

policy_net = DQN(n_observations, n_actions).to(DEVICE)
target_net = DQN(n_observations, n_actions).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Optimizaciones A100
if DEVICE.type == 'cuda':
    print("⚡ Enabling A100 optimizations...")
    policy_net = torch.compile(policy_net)
    target_net = torch.compile(target_net)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    print("✅ Optimizations enabled!")

replay_memory = PrioritizedReplayMemory(500000, alpha=0.6, beta_start=0.4, beta_frames=200000)

def select_actions_batch(states_batch, epsilon_val):
    """Selecciona acciones para un batch de estados"""
    if np.random.rand() < epsilon_val:
        return np.random.randint(0, n_actions, NUM_PARALLEL_ENVS)
    else:
        with torch.no_grad():
            q_values = policy_net(states_batch)
            actions = q_values.max(1).indices.cpu().numpy()
        return actions

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=100)

def optimize_model():
    """Optimización con batch grande"""
    if len(replay_memory) < BATCH_SIZE:
        return None
    
    transitions, indices, weights = replay_memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    
    state_batch = torch.cat(batch.state).to(DEVICE)
    action_batch = torch.cat(batch.action).to(DEVICE)
    reward_batch = torch.cat(batch.reward).to(DEVICE)
    done_batch = torch.tensor(batch.done, dtype=torch.float32, device=DEVICE)
    
    non_final_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool, device=DEVICE)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(DEVICE)
    
    # Double DQN - SIN autocast en el cálculo de next_state_values
    next_state_values = torch.zeros(BATCH_SIZE, dtype=torch.float32, device=DEVICE)
    
    if non_final_next_states.size(0) > 0:
        with torch.no_grad():
            # Calcular sin mixed precision para evitar dtype mismatch
            next_actions = policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
            next_q_values = target_net(non_final_next_states).gather(1, next_actions).squeeze(1)
            # Asegurar que sea float32
            next_state_values[non_final_mask] = next_q_values.float()
    
    # Calcular loss - aquí SÍ podemos usar autocast
    q_policy = policy_net(state_batch).gather(1, action_batch).squeeze()
    q_target = reward_batch + (GAMMA * next_state_values * (1 - done_batch))
    
    td_errors = (q_target - q_policy).detach()
    
    # Loss con weights de prioritized replay
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    loss = (weights_tensor * nn.functional.smooth_l1_loss(q_policy, q_target, reduction='none')).mean()
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=5.0)
    optimizer.step()
    
    replay_memory.update_priorities(indices, td_errors.abs().cpu().numpy())
    
    return loss.item()

# ============== ENTRENAMIENTO PARALELO ==============
print("\n" + "="*70)
print("🏁 STARTING PARALLEL TRAINING - CAR RACING")
print("="*70)
print(f"🌐 Parallel environments: {NUM_PARALLEL_ENVS}")
print(f"📊 Target episodes: {NUM_EPISODES}")
print(f"🎯 Batch size: {BATCH_SIZE}")
print("="*70 + "\n")

start_time = time.time()
total_steps = 0
episode_count = 0
recent_rewards = deque(maxlen=100)
steps_per_optimization = 4

os.makedirs("checks", exist_ok=True)
os.makedirs("models", exist_ok=True)

while episode_count < NUM_EPISODES and not stop_training:
    # Obtener estados y seleccionar acciones
    states_batch = env_manager.get_states_batch()
    actions = select_actions_batch(states_batch, epsilon)
    
    # Ejecutar paso en todos los entornos
    next_states, rewards, dones, _, completed = env_manager.step(actions)
    
    # Agregar experiencias al replay buffer
    for i in range(NUM_PARALLEL_ENVS):
        state = torch.tensor(env_manager.states[i], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        action = torch.tensor([actions[i]], dtype=torch.long, device=DEVICE)
        reward = torch.tensor([rewards[i]], dtype=torch.float32, device=DEVICE)
        
        next_state = None
        if not dones[i]:
            next_state = torch.tensor(next_states[i], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        
        replay_memory.push(state, action, next_state, reward, dones[i])
    
    # Procesar episodios completados
    for ep in completed:
        episode_count += 1
        recent_rewards.append(ep['reward'])
        reward_list.append(ep['reward'])
        
        # Logging
        if episode_count % 50 == 0:
            avg_reward = np.mean(recent_rewards)
            elapsed = time.time() - start_time
            eps_per_hour = episode_count / (elapsed / 3600) if elapsed > 0 else 0
            steps_per_sec = total_steps / elapsed if elapsed > 0 else 0
            
            print(f"Ep {episode_count:5d} | "
                  f"R: {ep['reward']:6.2f} | "
                  f"Avg(100): {avg_reward:6.2f} | "
                  f"ε: {epsilon:.4f} | "
                  f"Buffer: {len(replay_memory):6d} | "
                  f"{eps_per_hour:.0f} ep/h | "
                  f"{steps_per_sec:.1f} steps/s")
        
        # Checkpoints
        if episode_count % 1000 == 0 and episode_count > 0:
            checkpoint_path = f"checks/car_racing_ep{episode_count}.pth"
            torch.save(policy_net.state_dict(), checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Early stopping
        if EARLY_STOPPING_ENABLED and episode_count > EARLY_STOPPING_STARTING_EPISODE and len(recent_rewards) >= 100:
            current_avg = np.mean(recent_rewards)
            scheduler.step(current_avg)
            
            if current_avg > best_reward + EARLY_STOPPING_THRESHOLD:
                best_reward = current_avg
                early_stopping_patience = INITIAL_PATIENCE
                torch.save(policy_net.state_dict(), "models/best_car_racing.pth")
                print(f"💎 New best model! Avg: {best_reward:.2f}")
            else:
                early_stopping_patience -= 1
                if early_stopping_patience <= 0:
                    print(f"\n⏹️  Early stopping at episode {episode_count}")
                    stop_training = True
                    break
    
    # Optimizar red
    total_steps += NUM_PARALLEL_ENVS
    if total_steps % (steps_per_optimization * NUM_PARALLEL_ENVS) == 0 and len(replay_memory) >= BATCH_SIZE:
        loss = optimize_model()
        
        # Soft update target network
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(TAU * policy_param.data + (1 - TAU) * target_param.data)
    
    # Epsilon decay (empezar después de 1000 episodios)
    if episode_count >= 1000:
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

# ============== GUARDAR Y FINALIZAR ==============
env_manager.close()

final_path = "models/car_racing_final.pth"
torch.save(policy_net.state_dict(), final_path)

total_time = time.time() - start_time
final_avg = np.mean(recent_rewards) if len(recent_rewards) > 0 else 0

print("\n" + "="*70)
print("🎉 TRAINING COMPLETED!")
print("="*70)
print(f"⏱️  Total time: {total_time/3600:.2f} hours")
print(f"📊 Total episodes: {episode_count}")
print(f"📈 Best average reward: {best_reward:.2f}")
print(f"📉 Final average (100 ep): {final_avg:.2f}")
print(f"💾 Final model saved: {final_path}")
print("="*70)