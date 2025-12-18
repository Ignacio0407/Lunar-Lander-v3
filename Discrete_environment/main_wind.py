import gymnasium as gym
from replay_memory import ReplayMemory, PrioritizedReplayMemory, Transition
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dqn import DQN, DQN_heavy

NUM_EPISODES = 12000
BATCH_SIZE = 256
GAMMA = 0.99
LR = 3e-5
TAU = 0.005

epsilon = 1.0
EPSILON_MIN = 0.005
EPSILON_DECAY = 0.9993

EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_THRESHOLD = 1
EARLY_STOPPING_STARTING_EPISODE = 10000
INITIAL_PATIENCE = 800
early_stopping_patience = INITIAL_PATIENCE
best_reward = -np.inf
stop_training = False
reward_list = []

USE_PRIORITIZED_REPLAY = True
PRIORITIZED_REPLAY_ALPHA = 0.7
PRIORITIZED_REPLAY_BETA_START = 0.5
PRIORITIZED_REPLAY_BETA_END = 1.0

SIMPLE_NETWORK = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {DEVICE}")

env = gym.make("LunarLander-v3", enable_wind=True)

n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n

if SIMPLE_NETWORK == False:
    policy_net = DQN_heavy(n_observations, n_actions).to(DEVICE)
    target_net = DQN_heavy(n_observations, n_actions).to(DEVICE)
else:
    policy_net = DQN(n_observations, n_actions).to(DEVICE)
    target_net = DQN(n_observations, n_actions).to(DEVICE)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

if USE_PRIORITIZED_REPLAY:
    replay_memory = PrioritizedReplayMemory(100000)
else:
    replay_memory = ReplayMemory(100000)

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True, weight_decay=1e-5)
criterion = nn.SmoothL1Loss(reduction='none')

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=200)


def select_action(state):
    if np.random.rand() < epsilon:
        return torch.tensor([[env.action_space.sample()]], dtype=torch.long, device=DEVICE)
    else:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)


def optimize_model():
    if len(replay_memory) < BATCH_SIZE:
        return
    
    if USE_PRIORITIZED_REPLAY:
        beta = PRIORITIZED_REPLAY_BETA_START + (PRIORITIZED_REPLAY_BETA_END - PRIORITIZED_REPLAY_BETA_START) * \
               min(1.0, episode / NUM_EPISODES)
        transitions, indices, weights = replay_memory.sample(BATCH_SIZE, beta)
        weights = torch.FloatTensor(weights).to(DEVICE).unsqueeze(1)
    else:
        transitions = replay_memory.sample(BATCH_SIZE)
        weights = torch.ones(BATCH_SIZE, 1).to(DEVICE)
    
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
    
    td_errors = criterion(q_policy, q_target.unsqueeze(1))
    loss = (td_errors * weights).mean()
    
    if USE_PRIORITIZED_REPLAY:
        priorities = td_errors.detach().cpu().numpy().flatten()
        replay_memory.update_priorities(indices, priorities)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()

print(f"📊 Episodes: {NUM_EPISODES}, Batch size: {BATCH_SIZE}, LR: {LR}")
print(f"🔍 Prioritized Replay: {USE_PRIORITIZED_REPLAY}")

for episode in range(NUM_EPISODES):
    if stop_training:
        break
    
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    total_reward = 0.0
    
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        
        reward_tensor = torch.tensor([reward], device=DEVICE)
        
        next_state = None
        if not done:
            next_state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        
        replay_memory.push(state, action, next_state, reward_tensor, done)
        
        state = next_state
        total_reward += reward
        
        optimize_model()
        
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(TAU * policy_param.data + (1 - TAU) * target_param.data)
        
        if done:
            reward_list.append(total_reward)
            
            if len(reward_list) >= 100:
                avg_reward = np.mean(reward_list[-100:])
                scheduler.step(avg_reward)
            
            if episode % 50 == 0:
                avg_100 = np.mean(reward_list[-100:]) if len(reward_list) >= 100 else np.mean(reward_list)
                print(f"Episode {episode:5d} | Reward: {total_reward:7.2f} | Avg(100): {avg_100:7.2f} | ε: {epsilon:.4f} | Buffer: {len(replay_memory)}")

            if EARLY_STOPPING_ENABLED and episode > EARLY_STOPPING_STARTING_EPISODE and len(reward_list) >= 100:
                current_avg = np.mean(reward_list[-100:])
                if current_avg > best_reward + EARLY_STOPPING_THRESHOLD:
                    best_reward = current_avg
                    early_stopping_patience = INITIAL_PATIENCE
                else:
                    early_stopping_patience -= 1
                    if early_stopping_patience <= 0:
                        print(f"⏹️  Early stopping triggered at episode {episode}")
                        stop_training = True
            break
    
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    
    if episode % 100 == 0 and episode > 0:
        torch.save(policy_net.state_dict(), f"checks/checkpoint_ep{episode}.pth")
        print(f"💾 Checkpoint saved at episode {episode}")

torch.save(policy_net.state_dict(), "models/ddqn_lunar_lander_final.pth")
print("🎉 Training completed and model saved successfully!")
print(f"📈 Best average reward: {best_reward:.2f}")
print(f"📉 Final average reward (last 100): {np.mean(reward_list[-100:]):.2f}")
env.close()