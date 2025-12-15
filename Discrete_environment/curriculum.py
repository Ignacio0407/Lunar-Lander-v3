import gymnasium as gym
from replay_memory import PrioritizedReplayMemory, Transition
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dqn import DQN_heavy

"""
CURRICULUM LEARNING: Entrenar progresivamente
- Fase 1 (0-4000): Sin viento
- Fase 2 (4000-8000): Viento débil (wind_power=5)
- Fase 3 (8000-12000): Viento normal (wind_power=15, default)
"""

NUM_EPISODES = 12000
BATCH_SIZE = 256
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

env = gym.make("LunarLander-v3", enable_wind=CURRICULUM_STAGES[0]["wind"], wind_power=CURRICULUM_STAGES[0]["wind_power"])

n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n

policy_net = DQN_heavy(n_observations, n_actions).to(DEVICE)
target_net = DQN_heavy(n_observations, n_actions).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

replay_memory = PrioritizedReplayMemory(100000)

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True, weight_decay=1e-4)
criterion = nn.SmoothL1Loss(reduction='none')

def select_action(state, epsilon_override=None):
    eps = epsilon_override if epsilon_override is not None else epsilon
    if np.random.rand() < eps:
        return torch.tensor([[env.action_space.sample()]], dtype=torch.long, device=DEVICE)
    else:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)

def optimize_model():
    if len(replay_memory) < BATCH_SIZE:
        return
    
    beta = 0.5 + 0.5 * min(1.0, episode / NUM_EPISODES)
    transitions, indices, weights = replay_memory.sample(BATCH_SIZE, beta)
    weights = torch.FloatTensor(weights).to(DEVICE).unsqueeze(1)
    
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
    
    priorities = td_errors.detach().cpu().numpy().flatten()
    replay_memory.update_priorities(indices, np.abs(priorities))
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5)
    optimizer.step()
    
    return loss.item()

print("="*70)
print("🎓 CURRICULUM LEARNING")
for stage in CURRICULUM_STAGES:
    wind_str = f"Wind={stage['wind_power']}" if stage['wind'] else "No Wind"
    print(f"  Stage {CURRICULUM_STAGES.index(stage)+1}: Episodes {stage['start']}-{stage['end']} | {wind_str}")
print("="*70)

warmup_done = False

for episode in range(NUM_EPISODES):
    for i, stage in enumerate(CURRICULUM_STAGES):
        if stage["start"] <= episode < stage["end"] and i != current_stage:
            current_stage = i
            env.close()
            env = gym.make(
                "LunarLander-v3",
                enable_wind=stage["wind"],
                wind_power=stage["wind_power"]
            )
            print(f"\n🎯 STAGE {i+1}: {stage['name']} (wind_power={stage['wind_power']})")
            print(f"   Epsilon: {epsilon:.5f} | Buffer: {len(replay_memory)}\n")
            
            epsilon = min(0.3, epsilon * 1.5)
            
            torch.save(policy_net.state_dict(), f"models/curriculum_stage{i}.pth")
            break
    
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    total_reward = 0.0
    
    if episode < WARMUP_EPISODES:
        epsilon_current = 1.0
    else:
        epsilon_current = epsilon
        if not warmup_done:
            print(f"✅ Warm-up completed! Starting curriculum...\n")
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
        
        if episode >= WARMUP_EPISODES:
            optimize_model()
            
            for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                target_param.data.copy_(TAU * policy_param.data + (1 - TAU) * target_param.data)
        
        if done:
            reward_list.append(total_reward)
            stage_name = CURRICULUM_STAGES[current_stage]["name"]
            stage_rewards[stage_name].append(total_reward)

            if episode % 50 == 0:
                stage_avg = np.mean(stage_rewards[stage_name][-50:]) if len(stage_rewards[stage_name]) >= 50 else np.mean(stage_rewards[stage_name])
                overall_avg = np.mean(reward_list[-100:]) if len(reward_list) >= 100 else np.mean(reward_list)
                
                print(f"Ep {episode:5d} [{stage_name}] | R: {total_reward:7.2f} | "
                      f"Stage Avg(50): {stage_avg:7.2f} | "
                      f"Overall Avg(100): {overall_avg:7.2f} | "
                      f"ε: {epsilon:.5f}")

            if episode % 100 == 0 and episode > 0:
                torch.save(policy_net.state_dict(), f"curr_checks/curriculum_ep{episode}.pth")
            
            break
    
    if episode >= WARMUP_EPISODES:
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

torch.save(policy_net.state_dict(), "models/curriculum_final.pth")

print("\n" + "="*70)
print("🎉 CURRICULUM TRAINING COMPLETED!")
print("="*70)
for stage_name, rewards in stage_rewards.items():
    if rewards:
        avg = np.mean(rewards)
        std = np.std(rewards)
        print(f"{stage_name:15s} | Avg: {avg:7.2f} ± {std:5.2f} | Episodes: {len(rewards)}")

final_avg = np.mean(reward_list[-100:])
final_std = np.std(reward_list[-100:])
print(f"\n{'FINAL (last 100)':15s} | Avg: {final_avg:7.2f} ± {final_std:5.2f}")
print("="*70)

env.close()