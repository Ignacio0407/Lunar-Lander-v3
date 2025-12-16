from prioritized_replay_memory import PrioritizedReplayMemory, Transition, DEVICE
import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
from dqn import DQN
from preprocessing import SkipFrame
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
from gymnasium.wrappers import FrameStackObservation as FrameStack
import time
import os
from collections import deque

NUM_PARALLEL_ENVS = 24
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
reward_average_100_episodes = 0.

print(f"Using device: {DEVICE}")

class ParallelEnvManager:
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
        for i in range(self.num_envs):
            state, _ = self.envs[i].reset()
            self.states[i] = state
        return self.states
    
    def close(self):
        for env in self.envs:
            env.close()

env_manager = ParallelEnvManager(NUM_PARALLEL_ENVS)

n_observations = env_manager.envs[0].observation_space.shape
n_actions = env_manager.envs[0].action_space.n

policy_net = DQN(n_observations, n_actions).to(DEVICE)
target_net = DQN(n_observations, n_actions).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

if DEVICE.type == 'cuda':
    policy_net = torch.compile(policy_net)
    target_net = torch.compile(target_net)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

replay_memory = PrioritizedReplayMemory(500000, alpha=0.6, beta_start=0.4, beta_frames=200000)

def select_actions(states, epsilon_override=None):
    eps = epsilon_override if epsilon_override is not None else epsilon
    batch_size = len(states)
    
    state_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=DEVICE)
    state_tensor = state_tensor.permute(0, 3, 1, 2) if state_tensor.dim() == 4 else state_tensor
    
    if np.random.rand() < eps:
        return np.random.randint(0, n_actions, batch_size)
    else:
        with torch.no_grad(), torch.amp.autocast('cuda'):
            q_values = policy_net(state_tensor)
            actions = q_values.max(1).indices.cpu().numpy()
        return actions

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=100)

start_time = time.time()
total_steps = 0
episode_count = 0
recent_rewards = deque(maxlen=100)

while episode_count < NUM_EPISODES and not stop_training:
    actions = select_actions(env_manager.states)
    
    next_states, rewards, dones, _ = env_manager.step(actions)
    
    for i in range(NUM_PARALLEL_ENVS):
        state = torch.tensor(env_manager.states[i], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        next_state = torch.tensor(next_states[i], dtype=torch.float32, device=DEVICE).unsqueeze(0) if not dones[i] else None
        action = torch.tensor([actions[i]], dtype=torch.long, device=DEVICE)
        reward = torch.tensor([rewards[i]], device=DEVICE)
        
        if state.dim() == 5:
            state = state.squeeze(1)
        if next_state is not None and next_state.dim() == 5:
            next_state = next_state.squeeze(1)
            
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
            
            if EARLY_STOPPING_ENABLED and episode_count > EARLY_STOPPING_STARTING_EPISODE and len(recent_rewards) >= 100:
                current_avg = np.mean(recent_rewards)
                scheduler.step(current_avg)
                
                if current_avg > best_reward + EARLY_STOPPING_THRESHOLD:
                    best_reward = current_avg
                    early_stopping_patience = INITIAL_PATIENCE
                    torch.save(policy_net.state_dict(), "models/best_car_racing.pth")
                    print(f"💎 New best model saved! Avg: {best_reward:.2f}")
                else:
                    early_stopping_patience -= 1
                    if early_stopping_patience <= 0:
                        print(f"\n⏹️  Early stopping triggered at episode {episode_count}")
                        stop_training = True
                        break
    
    total_steps += NUM_PARALLEL_ENVS
    if total_steps % (4 * NUM_PARALLEL_ENVS) == 0 and len(replay_memory) >= BATCH_SIZE:
        if len(replay_memory) >= BATCH_SIZE:
            transitions, indices, weights = replay_memory.sample(BATCH_SIZE)
            batch = Transition(*zip(*transitions))

            state_batch = torch.cat(batch.state).to(DEVICE)
            if state_batch.dim() == 5:
                state_batch = state_batch.squeeze(1)
                
            action_batch = torch.tensor(batch.action, dtype=torch.long, device=DEVICE).unsqueeze(1)
            reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=DEVICE)
            done_batch = torch.tensor(batch.done, dtype=torch.float32, device=DEVICE)

            non_final_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool, device=DEVICE)
            non_final_next_states = [s for s in batch.next_state if s is not None]
            
            if len(non_final_next_states) > 0:
                non_final_next_states = torch.cat(non_final_next_states).to(DEVICE)
                if non_final_next_states.dim() == 5:
                    non_final_next_states = non_final_next_states.squeeze(1)
            
            next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)

            if len(non_final_next_states) > 0 and len(non_final_next_states) > 0:
                with torch.no_grad():
                    next_actions = policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                    next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_actions).squeeze(1) 
            
            with torch.amp.autocast('cuda'):
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
        
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(TAU * policy_param.data + (1 - TAU) * target_param.data)
    
    if episode_count >= 1000:
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    
    if episode_count % 1000 == 0 and episode_count > 0:
        checkpoint_path = f"checks/car_racing_ep{episode_count}.pth"
        os.makedirs("checks", exist_ok=True)
        torch.save(policy_net.state_dict(), checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")

env_manager.close()
final_path = "models/car_racing_final.pth"
os.makedirs("models", exist_ok=True)
torch.save(policy_net.state_dict(), final_path)
print(f"✅ Final model saved: {final_path}")

total_time = time.time() - start_time
print(f"⏱️  Total training time: {total_time/3600:.2f} hours")
print("Training completed and model saved successfully!")