from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_observations:int, n_actions:int):
        '''
        Input:  (4, 84, 84)    # 4 canals
        ↓
        Conv1:  (32, 20, 20)   # 32 features, compacted
                ↓
        Conv2:  (64, 9, 9)     # 64 features, more compacted
                ↓
        Conv3:  (64, 7, 7)     # 64 features, super compacted
                ↓
        Flatten: (3136,)       # Vector 1D for fully-connected layers
        '''
        super().__init__()
        self.layer1 = nn.Conv2d(in_channels=n_observations, out_channels=32, kernel_size=8, stride=4) # (32, 20, 20) -> (84-8)/4 +1
        self.layer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2) # (64, 9, 9) -> (20-4)/2 + 1
        self.layer3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1) # (64, 7, 7) -> (9-3)/1 + 1
        self.layer4 = nn.Linear(64 * 7 * 7, 512)
        self.layer5 = nn.Linear(512, n_actions)

    def forward(self, x:Tensor):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.layer4(x))
        return self.layer5(x)

def calculate_conv_output_size(input_size, kernel_size, stride, padding=0):
    """Calculates output size of a convolutional layer"""
    return ((input_size - kernel_size + 2 * padding) // stride) + 1

'''
Example:
model = DQN_CNN(n_observations=4, n_actions=4).to(device)
# Input: 32 images batch, 4 frames stacked, 84x84 px
x = torch.randn(32, 4, 84, 84).to(device)
# Forward
q_values = model(x)  # Shape: (32 (pictures), 4 (q-values for actions in each frame)) ← Q-values for 4 actions
print(q_values.shape)  # torch.Size([32, 4])
'''