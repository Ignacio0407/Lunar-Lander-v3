from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class DQN_dynamic(nn.Module):
    def __init__(self, n_observations:int, n_actions:int, state_dict:dict):
        super().__init__()
        input_size = n_observations

        # Detect hidden layers from checkpoint
        hidden_sizes = []
        for key, value in state_dict.items():
            if "weight" in key and "layer" in key and "layer3" not in key:
                hidden_sizes.append(value.shape[0])

        # Build hidden layers with the names outputted by dqn.py (layer1, layer2...)
        for i, hidden_size in enumerate(hidden_sizes):
            layer = nn.Linear(input_size, hidden_size)
            setattr(self, f"layer{i+1}", layer)
            input_size = hidden_size

        # Output layer
        output_layer = nn.Linear(input_size, n_actions)
        setattr(self, f"layer{len(hidden_sizes)+1}", output_layer)

        # Save layers number for correct forwarding
        self.num_layers = len(hidden_sizes) + 1

    def forward(self, x: Tensor):
        # Apply RELU to all hidden layers 
        for i in range(1, self.num_layers):
            layer = getattr(self, f"layer{i}")
            x = F.relu(layer(x))
        # Output layer with no activation function.
        output_layer = getattr(self, f"layer{self.num_layers}")
        return output_layer(x)