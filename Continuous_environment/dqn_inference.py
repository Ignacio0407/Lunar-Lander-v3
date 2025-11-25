import gymnasium as gym
import torch
from itertools import count
from dqn_dynamic import DQN_dynamic
import os

base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "models", "wind_128_3.pth")

num_episodes = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Initialize the environment
env = gym.make("LunarLander-v3", render_mode="human")

n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n

checkpoint = torch.load(model_path)
# Initialize the model architecture
policy_net = DQN_dynamic(n_observations, n_actions, state_dict=checkpoint).to(device)
# Load the trained weights
policy_net.load_state_dict(checkpoint)
policy_net.eval() # Set the model to evaluation mode

print("Model loaded successfully!")


for episode in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    total_reward = 0.
    for t in count():
        with torch.no_grad():
            action = (policy_net(state).max(1).indices.view(1, 1))  # Exploit (best action)

        next_state, reward, terminated, truncated, info = env.step(action.item())

        state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward += reward
        done = terminated or truncated
        if done:
            print(total_reward)
            break