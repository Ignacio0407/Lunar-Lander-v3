import gymnasium as gym
import torch
import os
from itertools import count
import time
from dqn import DQN
from preprocessing import SkipFrame
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
from gymnasium.wrappers import FrameStackObservation as FrameStack

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "car_racing_2368.pth")
#MODEL_PATH = os.path.join(BASE_DIR, "models", "car_racing_fine_tune_from_4600_4700.pth")
#MODEL_PATH = os.path.join(BASE_DIR, "models", "car_racing_domain_randomize_16042.pth")

NUM_EPISODES = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🎮 Device: {device}")

env = gym.make("CarRacing-v3", continuous=False) #render_mode="human")
#env = gym.make("CarRacing-v3", continuous=False, domain_randomize=True) #render_mode="human")
env = SkipFrame(env, skip=4)
env = GrayscaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=(84, 84))
env = FrameStack(env, 4)

n_actions = env.action_space.n  # 5 discrete actions
n_observations = 4

try:
    print(f"📂 Loading model from: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    policy_net = DQN(n_observations, n_actions).to(device)
    policy_net.load_state_dict(checkpoint)
    policy_net.eval()
    
    print(f"✅ Model loaded successfully! Input shape: (4, 84, 84), Actions: {n_actions}")
    print("🚀 Game window will appear - press any key to start playing...")
    
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    print("🔄 Using random actions for demonstration...")
    policy_net = None

for episode in range(NUM_EPISODES):
    print(f"\n🏁 Episode {episode + 1}/{NUM_EPISODES}")
    obs, info = env.reset()
    total_reward = 0.0
    step_count = 0
    
    for t in count():
        # ✅ Convert observation to tensor with correct dimensions obs shape: (4, 84, 84) from FrameStack
        state_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            if policy_net is not None:
                q_values = policy_net(state_tensor)
                action = torch.argmax(q_values, dim=1).item()
            else:
                action = env.action_space.sample()
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        step_count += 1
        obs = next_obs
        
        if step_count % 50 == 0:
            print(f"  Step {step_count}: Total Reward = {total_reward:.2f}")
        
        done = terminated or truncated
        if done:
            print(f"✅ Episode {episode + 1} finished after {step_count} steps")
            print(f"🏆 Total Reward: {total_reward:.2f}")
            time.sleep(2)  # Pause to see final state
            break

env.close()
print("\n🎉 Inference completed successfully!")
print("💡 Close the game window manually when you're done watching.")