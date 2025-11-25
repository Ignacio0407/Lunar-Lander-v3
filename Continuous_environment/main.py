from prioritized_replay_memory import PrioritizedReplayMemory, Transition, DEVICE
import gymnasium as gym
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dqn import DQN
from preprocessing import SkipFrame # GrayScaleObservation, ResizeObservation, 
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
from gymnasium.wrappers import FrameStackObservation as FrameStack

NUM_EPISODES = 5000
BATCH_SIZE = 256 # Number of transitions sampled from the replay buffer
GAMMA = 0.99 # Discount factor of q or policy network
LR = 1e-4
TAU = 0.005 # Update rate of the target network

epsilon = 1.0  # Starting value of epsilon for epsilon greedy policy. 1 is full exploration (all actions taken randomly)
EPSILON_MIN = 0.1  # Minimum value
EPSILON_DECAY = 0.995  # Decay factor per episode, higher means a slower decay

EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_THRESHOLD = 20
EARLY_STOPPING_STARTING_EPISODE = 2000
INITIAL_PATIENCE = 300
early_stopping_patience = INITIAL_PATIENCE
best_reward = -250.
stop_training = False
reward_list = []
reward_average_100_episodes = 0.

print(f"Using device: {DEVICE}")

#env = gym.make("LunarLander-v3", render_mode="rgb_array")
env = gym.make("CarRacing-v3", continuous=False)
env = SkipFrame(env, skip=4)
env = GrayscaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=(84, 84))
env = FrameStack(env, 4)

n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n

policy_net = DQN(n_observations, n_actions).to(DEVICE)
target_net = DQN(n_observations, n_actions).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # Target network in evaluation mode

replay_memory = PrioritizedReplayMemory(100000, alpha=0.6, beta_start=0.4, beta_frames=100000)

def select_action(state):
    if np.random.rand() < epsilon:
        return torch.tensor([[env.action_space.sample()]], dtype=torch.long, device=DEVICE)
    else:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
criterion = nn.SmoothL1Loss()

for episode in range(NUM_EPISODES):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    total_reward = 0.0
    
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        
        next_state = None
        if not done:
            next_state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        
        reward_tensor = torch.tensor([reward], device=DEVICE) # Convert float to tensor
        replay_memory.push(state, action, next_state, reward_tensor, done)
        
        state = next_state
        total_reward += reward
        
        if len(replay_memory) >= BATCH_SIZE:
            transitions, indices, weights = replay_memory.sample(BATCH_SIZE)
            batch = Transition(*zip(*transitions))

            state_batch = torch.cat(batch.state).to(DEVICE)
            action_batch = torch.cat(batch.action).to(DEVICE)
            reward_batch = torch.cat(batch.reward).to(DEVICE)
            done_batch = torch.tensor(batch.done, device=DEVICE, dtype=torch.float32)

            # Arrangement to not take into account final states in the calculations
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=DEVICE, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(DEVICE)
            
            # Double DQN
            next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)

            if non_final_next_states.size(0) > 0:
                with torch.no_grad():
                    # Policy network predicts Q-values and chooses the best action in current state for all states in batch
                    next_actions = policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                    all_q_values = target_net(non_final_next_states) # Target net predicts Q-values 
                    # gather extracts Q values of the actions chosen by policy net for all states in the batch.
                    next_state_values[non_final_mask] = all_q_values.gather(1, next_actions).squeeze(1) 
            
            # # Current q_values: [10.5, 8.2, 15.0, 9.1]
            q_policy = policy_net(state_batch).gather(1, action_batch).squeeze()
            # Target q_values: [12.0, 8.3, 20.5, 9.0]
            q_target = reward_batch.squeeze() + (GAMMA * next_state_values * (1 - done_batch))

            # TD errors: [1.5, 0.1, 5.5, -0.1], high values means important transitions to learn
            td_errors = q_target.detach() - q_policy.detach() 
            # Huber Loss with weights for prioritized experience replay.
            loss = (weights * torch.nn.functional.smooth_l1_loss(q_policy, q_target, reduction='none')).mean()
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            # In-place gradient normalizing to stabilize training, better than clipping
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10)
            # torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
            optimizer.step()

            # Update priorities!
            replay_memory.update_priorities(indices, td_errors.abs().cpu().numpy())
        
        # --- SOFT UPDATE TARGET NETWORK ---
        for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(TAU * policy_param.data + (1 - TAU) * target_param.data)
        
        if done:
            reward_list.append(total_reward)
            print("Episode", episode)
            if EARLY_STOPPING_ENABLED and episode > EARLY_STOPPING_STARTING_EPISODE and len(reward_list) >= 100:
                current_avg = np.mean(reward_list[-100:])
                if current_avg > best_reward + EARLY_STOPPING_THRESHOLD:
                    best_reward = current_avg
                    early_stopping_patience = INITIAL_PATIENCE
                else:
                    early_stopping_patience -= 1
                    print(f"⏳ Patience: {early_stopping_patience}/{INITIAL_PATIENCE}")
                    if early_stopping_patience <= 0:
                        stop_training = True
            break
    
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    if episode % 200 == 0 and episode > 0:
        torch.save(policy_net.state_dict(), f"models/checkpoint_ep{episode}.pth")
        print(f"💾 Checkpoint saved at episode {episode}")

torch.save(policy_net.state_dict(), "models/lunar_lander.pth")
print("Training completed and model saved successfully!")
env.close()