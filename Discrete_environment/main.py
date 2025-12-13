import gymnasium as gym
from itertools import count
import torch
import torch.optim as optim
import numpy as np
from prioritized_replay_memory import PrioritizedReplayMemory, Transition
from dqn import DQN

NUM_EPISODES = 10000
BATCH_SIZE = 256  # Number of transitions sampled from the replay buffer
GAMMA = 0.99  # Discount factor of q or policy network
LR = 3e-4
TAU = 0.005  # Update rate of the target network

epsilon = 1.0  # Starting value of epsilon for epsilon greedy policy. 1 is full exploration (all actions taken randomly)
EPSILON_MIN = 0.05  # Minimum value
EPSILON_DECAY = 0.9995  # Decay factor per episode, higher means a slower decay

EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_THRESHOLD = 5
EARLY_STOPPING_STARTING_EPISODE = 2000
INITIAL_PATIENCE = 300
early_stopping_patience = INITIAL_PATIENCE
best_reward = -200.
stop_training = False
reward_list = []
reward_counter = 0
main_thrust_counter = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Initialize environment WITH WIND
env = gym.make("LunarLander-v3", enable_wind=False)

n_observations = env.observation_space.shape[0]  # 8 observations in LunarLander-v3
n_actions = env.action_space.n  # 4 discrete actions

policy_net = DQN(n_observations, n_actions).to(DEVICE)
target_net = DQN(n_observations, n_actions).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # Target network in evaluation mode

replay_memory = PrioritizedReplayMemory(capacity=100000, alpha=0.6, beta_start=0.4, beta_frames=100000)

def select_action(state):
    if np.random.rand() < epsilon:
        return torch.tensor([[env.action_space.sample()]], dtype=torch.long, device=DEVICE)
    else:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

for episode in range(NUM_EPISODES):
    if stop_training:
        break
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    total_reward = 0.0
    main_thrust_counter = 0  # Reset counter per episode
    
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
        
        # Training step
        if len(replay_memory) >= BATCH_SIZE:
            transitions, indices, weights = replay_memory.sample(BATCH_SIZE)
            batch = Transition(*zip(*transitions))

            state_batch = torch.cat(batch.state).to(DEVICE)  # shape [B, obs_dim]
            action_batch = torch.cat(batch.action).to(DEVICE)  # shape [B, 1]
            reward_batch = torch.cat(batch.reward).to(DEVICE)  # shape [B]
            done_batch = torch.tensor(batch.done, device=DEVICE, dtype=torch.float32)  # shape [B]

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=DEVICE, dtype=torch.bool)
            
            non_final_next_states_list = [s for s in batch.next_state if s is not None]
            if non_final_next_states_list:
                non_final_next_states = torch.cat(non_final_next_states_list).to(DEVICE)
            else:
                non_final_next_states = torch.empty((0, n_observations), device=DEVICE)

            next_state_values_full = torch.zeros(BATCH_SIZE, device=DEVICE)

            if non_final_next_states.size(0) > 0:
                with torch.no_grad():
                    # Double DQN: policy net selects action, target net evaluates
                    next_actions = policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                    next_state_values = target_net(non_final_next_states).gather(1, next_actions).squeeze(1)
                    next_state_values_full[non_final_mask] = next_state_values

            # Compute Q values and targets
            q_policy = policy_net(state_batch).gather(1, action_batch).squeeze(1)
            q_target = reward_batch + GAMMA * next_state_values_full * (1 - done_batch)

            # Compute TD errors for PER
            with torch.no_grad():
                td_errors = (q_target - q_policy).abs()

            max_td_error = 10.0
            td_errors = torch.clamp(td_errors, min=0, max=max_td_error)
            
            # Compute loss with importance sampling weights
            loss = (weights * torch.nn.functional.smooth_l1_loss(q_policy, q_target, reduction='none')).mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update priorities in replay memory
            replay_memory.update_priorities(indices, td_errors.cpu().numpy())
        
        # Soft update target network
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(TAU * policy_param.data + (1 - TAU) * target_param.data)
        
        if done:
            reward_list.append(total_reward)
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}")
            reward_counter += 1

            if EARLY_STOPPING_ENABLED and episode > EARLY_STOPPING_STARTING_EPISODE and len(reward_list) >= 100:
                current_avg = np.mean(reward_list[-100:])
                if current_avg > best_reward + EARLY_STOPPING_THRESHOLD:
                    best_reward = current_avg
                    early_stopping_patience = INITIAL_PATIENCE
                else:
                    early_stopping_patience -= 1
                    print(f"⏳ Patience: {early_stopping_patience}/{INITIAL_PATIENCE}")
                    if early_stopping_patience <= 0:
                        print(f"🛑 Early stopping triggered. Best reward: {best_reward:.2f}")
                        stop_training = True
            break
    
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    if episode % 100 == 0 and episode > 0:
        checkpoint_path = f"/kaggle/working/checkpoint_ep{episode}.pth"
        torch.save(policy_net.state_dict(), checkpoint_path)
        print(f"💾 Checkpoint saved at episode {episode} to {checkpoint_path}")

# Save final model
model_path = "models/ddqn_lunar_lander_windy.pth"
torch.save(policy_net.state_dict(), model_path)
print(f"✅ Training completed and model saved successfully to {model_path}!")
env.close()