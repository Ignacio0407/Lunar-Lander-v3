import gymnasium as gym
from prioritized_replay_memory import PrioritizedReplayMemory, Transition, DEVICE
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dqn import DQN
from preprocessing import SkipFrame
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
from gymnasium.wrappers import FrameStackObservation as FrameStack
import time
from collections import deque
import os

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
EARLY_STOPPING_STARTING_EPISODE = 15000
INITIAL_PATIENCE = 300
early_stopping_patience = INITIAL_PATIENCE
best_reward = -250.0
stop_training = False
reward_list = []

print(f"🚀 Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

class ParallelEnvManager:
    """Gestor de múltiples entornos paralelos para recolección eficiente de experiencias"""
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.envs = []
        self.states = []
 
        for _ in range(num_envs):
            env = gym.make("CarRacing-v3", continuous=False, domain_randomize=True)
            env = SkipFrame(env, skip=4)
            env = GrayscaleObservation(env, keep_dim=False)
            env = ResizeObservation(env, shape=(84, 84))
            env = FrameStack(env, 4)
            self.envs.append(env)
            state, _ = env.reset()
            self.states.append(state)
    
    def step(self, actions):
        """Ejecutar acciones en todos los entornos en paralelo"""
        next_states = []
        rewards = []
        dones = []
        infos = []
        
        for i, action in enumerate(actions):
            next_state, reward, terminated, truncated, info = self.envs[i].step(action)
            done = terminated or truncated
            
            if done:
                next_state, _ = self.envs[i].reset()
            
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            self.states[i] = next_state if not done else next_state
        
        return next_states, rewards, dones, infos
    
    def reset(self):
        """Reiniciar todos los entornos"""
        for i in range(self.num_envs):
            state, _ = self.envs[i].reset()
            self.states[i] = state
        return self.states
    
    def close(self):
        """Cerrar todos los entornos"""
        for env in self.envs:
            env.close()

env_manager = ParallelEnvManager(NUM_PARALLEL_ENVS)

n_observations = env_manager.envs[0].observation_space.shape[0]
n_actions = env_manager.envs[0].action_space.n

policy_net = DQN(n_observations, n_actions).to(DEVICE)
target_net = DQN(n_observations, n_actions).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

'''
if DEVICE.type == 'cuda':
    policy_net = torch.compile(policy_net)
    target_net = torch.compile(target_net)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
'''

replay_memory = PrioritizedReplayMemory(500000, alpha=0.6, beta_start=0.4, beta_frames=200000)

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=100)

def select_actions(states, epsilon_override=None):
    """Seleccionar acciones para múltiples estados en batch (vectorizado)"""
    eps = epsilon_override if epsilon_override is not None else epsilon
    batch_size = len(states)

    state_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=DEVICE)
    
    if np.random.rand() < eps:
        # Exploration
        return np.random.randint(0, n_actions, batch_size)
    else:
        # Explotation
        with torch.no_grad(), torch.cuda.amp.autocast():
            q_values = policy_net(state_tensor)
            actions = q_values.max(1).indices.cpu().numpy()
        return actions

def optimize_model():
    """Optimización en batch con AMP para máxima velocidad en GPU"""
    if len(replay_memory) < BATCH_SIZE:
        return 0.0
    
    transitions, indices, weights = replay_memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    
    state_batch = torch.tensor(np.array([s.cpu().numpy() if isinstance(s, torch.Tensor) else s for s in batch.state]), 
                              dtype=torch.float32, device=DEVICE)
    action_batch = torch.tensor(batch.action, dtype=torch.long, device=DEVICE).unsqueeze(1)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=DEVICE)
    done_batch = torch.tensor(batch.done, dtype=torch.float32, device=DEVICE)
    
    non_final_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool, device=DEVICE)
    non_final_next_states = []
    for s in batch.next_state:
        if s is not None:
            if isinstance(s, torch.Tensor):
                non_final_next_states.append(s.cpu().numpy())
            else:
                non_final_next_states.append(s)
    
    if len(non_final_next_states) > 0:
        non_final_next_states = torch.tensor(np.array(non_final_next_states), dtype=torch.float32, device=DEVICE)
    
    # Double DQN with AMP
    with torch.cuda.amp.autocast():
        next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
        
        if len(non_final_next_states) > 0 and non_final_next_states.size(0) > 0:
            with torch.no_grad():
                next_actions = policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_actions).squeeze(1)
        
        # Q-values
        q_policy = policy_net(state_batch).gather(1, action_batch).squeeze()
        q_target = reward_batch + (GAMMA * next_state_values * (1 - done_batch))
        
        td_errors = q_target.detach() - q_policy.detach()
        loss = (torch.tensor(weights, device=DEVICE) * 
                torch.nn.functional.smooth_l1_loss(q_policy, q_target, reduction='none')).mean()
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=5.0)
    optimizer.step()
    
    replay_memory.update_priorities(indices, td_errors.abs().cpu().numpy())
    
    return loss.item()

print("="*70)
print("🏎️  PARALLEL CAR RACING TRAINING WITH DOMAIN RANDOMIZATION")
print(f"🌐 Parallel Environments: {NUM_PARALLEL_ENVS}")
print(f"📊 Batch Size: {BATCH_SIZE} | LR: {LR} | Gamma: {GAMMA}")
print(f"💾 Replay Buffer: {replay_memory.size}")
print("="*70)

start_time = time.time()
total_steps = 0
episode_count = 0
recent_rewards = deque(maxlen=100)

try:
    while episode_count < NUM_EPISODES and not stop_training:
        actions = select_actions(env_manager.states)

        next_states, rewards, dones, _ = env_manager.step(actions)
        
        for i in range(NUM_PARALLEL_ENVS):
            state = torch.tensor(env_manager.states[i], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            next_state = torch.tensor(next_states[i], dtype=torch.float32, device=DEVICE).unsqueeze(0) if not dones[i] else None
            action = torch.tensor([actions[i]], dtype=torch.long, device=DEVICE)
            reward = torch.tensor([rewards[i]], device=DEVICE)
            
            replay_memory.push(state, action, next_state, reward, dones[i])
            
            if dones[i]:
                episode_count += 1
                recent_rewards.append(rewards[i])
                reward_list.append(rewards[i])

                if episode_count % 50 == 0:
                    current_avg = np.mean(recent_rewards)
                    elapsed = time.time() - start_time
                    steps_per_sec = total_steps / elapsed if elapsed > 0 else 0
                    time_per_episode = elapsed / episode_count if episode_count > 0 else 0
                    
                    print(f"Ep {episode_count:5d} | " +
                          f"R: {rewards[i]:6.2f} | " +
                          f"Avg(100): {current_avg:6.2f} | " +
                          f"ε: {epsilon:.3f} | " +
                          f"Buffer: {len(replay_memory):6d} | " +
                          f"{steps_per_sec:.1f} steps/s | " +
                          f"{time_per_episode:.2f}s/ep")
                
                # Early stopping
                if EARLY_STOPPING_ENABLED and episode_count > EARLY_STOPPING_STARTING_EPISODE and len(recent_rewards) >= 100:
                    current_avg = np.mean(recent_rewards)
                    scheduler.step(current_avg)
                    
                    if current_avg > best_reward + EARLY_STOPPING_THRESHOLD:
                        best_reward = current_avg
                        early_stopping_patience = INITIAL_PATIENCE
                        torch.save(policy_net.state_dict(), "models/best_car_racing_domain_randomize.pth")
                        print(f"💎 New best model saved! Avg: {best_reward:.2f}")
                    else:
                        early_stopping_patience -= 1
                        if early_stopping_patience <= 0:
                            print(f"\n⏹️  Early stopping triggered at episode {episode_count}")
                            stop_training = True
                            break
        
        total_steps += NUM_PARALLEL_ENVS
        if total_steps % (4 * NUM_PARALLEL_ENVS) == 0 and len(replay_memory) >= BATCH_SIZE:
            loss = optimize_model()
            
            # Target network soft update
            for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                target_param.data.copy_(TAU * policy_param.data + (1 - TAU) * target_param.data)

        if episode_count >= 1000:
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        if episode_count % 1000 == 0 and episode_count > 0:
            checkpoint_path = f"checkpoints/car_racing_dr_ep{episode_count}.pth"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(policy_net.state_dict(), checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    final_path = "models/car_racing_domain_randomize_final.pth"
    os.makedirs("models", exist_ok=True)
    torch.save(policy_net.state_dict(), final_path)
    print(f"✅ Final model saved: {final_path}")

except KeyboardInterrupt:
    print("\n🛑 Training interrupted by user")
    final_path = "models/car_racing_domain_randomize_interrupted.pth"
    os.makedirs("models", exist_ok=True)
    torch.save(policy_net.state_dict(), final_path)
    print(f"💾 Model saved after interruption: {final_path}")

except Exception as e:
    print(f"❌ Error during training: {e}")
    import traceback
    traceback.print_exc()
    final_path = "models/car_racing_domain_randomize_error.pth"
    os.makedirs("models", exist_ok=True)
    torch.save(policy_net.state_dict(), final_path)
    print(f"💾 Model saved after error: {final_path}")

finally:
    env_manager.close()
    total_time = time.time() - start_time
    
    if reward_list:
        final_avg = np.mean(reward_list[-100:]) if len(reward_list) >= 100 else np.mean(reward_list)
        final_std = np.std(reward_list[-100:]) if len(reward_list) >= 100 else np.std(reward_list)
        
        print("\n" + "="*70)
        print("🏁 TRAINING COMPLETED!")
        print("="*70)
        print(f"⏱️  Total time: {total_time/3600:.2f} hours")
        print(f"📊 Episodes completed: {episode_count}/{NUM_EPISODES}")
        print(f"📈 Best average reward: {best_reward:.2f}")
        print(f"📉 Final average reward (last 100): {final_avg:.2f} ± {final_std:.2f}")
        print(f"⚡ Average speed: {(total_steps/total_time):.1f} steps/second")
        print("="*70)

print("🎉 Training finished successfully!")