from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, obs_shape:tuple, n_actions:int):
        """
        obs_shape: (C,H,W), número de canales, alto y ancho
        n_actions: número de acciones discretas
        """
        super().__init__()
        c, h, w = obs_shape

        self.layer1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4)
        self.layer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.layer3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1

        conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)

        linear_input_size = conv_h * conv_w * 64
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x: Tensor):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)