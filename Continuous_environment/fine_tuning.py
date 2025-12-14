from prioritized_replay_memory import PrioritizedReplayMemory, Transition, DEVICE
import gymnasium as gym
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dqn import DQN
from preprocessing import SkipFrame
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
from gymnasium.wrappers import FrameStackObservation as FrameStack
import os

NUM_EPISODES = 10000
BATCH_SIZE = 256
GAMMA = 0.99
LR = 3e-5  # ✅ Lower learning rate for fine-tuning
TAU = 0.005

epsilon = 0.5  # ✅ Less exploration (it already knows how to drive)
EPSILON_MIN = 0.1
EPSILON_DECAY = 0.998

EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_THRESHOLD = 5
EARLY_STOPPING_STARTING_EPISODE = 7000  # ✅ Evaluate before.
INITIAL_PATIENCE = 400
early_stopping_patience = INITIAL_PATIENCE
best_reward = 300.0  # ✅ Even though car_racing.pth gave 860 at worse I do not want to alter the training with this value
stop_training = False
reward_list = []

print(f"🔥 Device: {DEVICE}")

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "car_racing_4600.pth")

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found: {MODEL_PATH}")
    exit(1)

env = gym.make("CarRacing-v3", continuous=False, domain_randomize=True)
env = SkipFrame(env, skip=4)
env = GrayscaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=(84, 84))
env = FrameStack(env, 4)

n_observations = 4
n_actions = env.action_space.n

policy_net = DQN(n_observations, n_actions).to(DEVICE)
target_net = DQN(n_observations, n_actions).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
policy_net.load_state_dict(checkpoint)
target_net.load_state_dict(checkpoint)

target_net.eval()

# ===== REPLAY MEMORY =====
# ⚠️ IMPORTANT: Start with empty memory or charge previous one?
# Option 1: Empty memory (quicker at the beginning)
replay_memory = PrioritizedReplayMemory(100000, alpha=0.6, beta_start=0.4, beta_frames=50000)

# Option 2: Pre-fill with random episodes (better)
print("📦 Pre-llenando replay memory con episodios del modelo actual...")
for warmup_ep in range(50):  # 50 warmup episodes
    obs, _ = env.reset()
    for _ in count():
        state = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            if np.random.rand() < 0.3:  # 30% random
                action = env.action_space.sample()
            else:
                action = policy_net(state).argmax(1).item()
        
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        next_state = None if done else torch.tensor(next_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        replay_memory.push(state, torch.tensor([[action]], device=DEVICE), next_state, torch.tensor([reward], device=DEVICE), done)
        
        obs = next_obs
        if done:
            break
    
    if warmup_ep % 10 == 0:
        print(f"  Warmup: {warmup_ep}/50 episodios, Memory: {len(replay_memory)}")

print(f"✅ Replay memory inicializada: {len(replay_memory)} transiciones")

def select_action(state):
    if np.random.rand() < epsilon:
        return torch.tensor([[env.action_space.sample()]], dtype=torch.long, device=DEVICE)
    else:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

print("\n🚀 Starting fine_tuning...\n")

for episode in range(NUM_EPISODES):
    if stop_training:
        break
    
    obs, info = env.reset()
    state = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    total_reward = 0.0
    
    for t in count():
        action = select_action(state)
        next_obs, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        
        next_state = None if done else torch.tensor(next_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        reward_tensor = torch.tensor([reward], device=DEVICE)
        
        replay_memory.push(state, action, next_state, reward_tensor, done)
        
        state = next_state
        total_reward += reward
        
        if len(replay_memory) >= BATCH_SIZE:
            transitions, indices, weights = replay_memory.sample(BATCH_SIZE)
            batch = Transition(*zip(*transitions))

            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            done_batch = torch.tensor(batch.done, device=DEVICE, dtype=torch.float32)

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=DEVICE, dtype=torch.bool)
            
            if non_final_mask.any():
                non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
                next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
                
                with torch.no_grad():
                    next_actions = policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                    all_q_values = target_net(non_final_next_states)
                    next_state_values[non_final_mask] = all_q_values.gather(1, next_actions).squeeze(1)
            else:
                next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
            
            q_policy = policy_net(state_batch).gather(1, action_batch).squeeze()
            q_target = reward_batch.squeeze() + (GAMMA * next_state_values * (1 - done_batch))

            td_errors = q_target.detach() - q_policy.detach()
            loss = (weights * torch.nn.functional.smooth_l1_loss(q_policy, q_target, reduction='none')).mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10)
            optimizer.step()

            replay_memory.update_priorities(indices, td_errors.abs().cpu().numpy())
        
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(TAU * policy_param.data + (1 - TAU) * target_param.data)
        
        if done:
            reward_list.append(total_reward)
            
            if episode % 10 == 0:
                avg_10 = np.mean(reward_list[-10:]) if len(reward_list) >= 10 else total_reward
                print(f"Episode {episode:4d} | Reward: {total_reward:7.2f} | "
                      f"Avg(10): {avg_10:7.2f} | Epsilon: {epsilon:.3f}")
            
            if EARLY_STOPPING_ENABLED and episode > EARLY_STOPPING_STARTING_EPISODE and len(reward_list) >= 100:
                current_avg = np.mean(reward_list[-100:])
                
                if current_avg > best_reward + EARLY_STOPPING_THRESHOLD:
                    best_reward = current_avg
                    early_stopping_patience = INITIAL_PATIENCE
                else:
                    early_stopping_patience -= 1
                    if early_stopping_patience % 50 == 0:
                        print(f"⏳ Patience: {early_stopping_patience}/{INITIAL_PATIENCE}")
                    if early_stopping_patience <= 0:
                        print(f"🛑 Early stopping at episode {episode}")
                        stop_training = True
            break
    
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    
    if episode % 100 == 0 and episode > 0:
        torch.save(policy_net.state_dict(), f"checkpoints/retrain_checkpoint_ep{episode}.pth")
        print(f"💾 Checkpoint saved at episode {episode}")

torch.save(policy_net.state_dict(), "models/car_racing_fine_tune.pth")
print(f"\n🎉 Fine tuning completed!")
print(f"🏆 Best avg reward: {best_reward:.2f}")
env.close()