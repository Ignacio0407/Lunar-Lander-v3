from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import re

class DQN_dynamic(nn.Module):
    def __init__(self, n_observations: int, n_actions: int, state_dict: dict = None):
        super().__init__()
        
        if state_dict is None:
            self.layer1 = nn.Linear(n_observations, 128)
            self.layer2 = nn.Linear(128, 128)
            self.layer3 = nn.Linear(128, n_actions)
            self.num_layers = 3
        else:
            layer_info = []
            
            weight_keys = [k for k in state_dict.keys() if 'weight' in k and 'layer' in k]
            
            def extract_layer_number(key):
                match = re.search(r'layer(\d+)', key)
                return int(match.group(1)) if match else 0
            
            weight_keys.sort(key=extract_layer_number)
            
            prev_size = n_observations
            for i, key in enumerate(weight_keys):
                weight = state_dict[key]
                out_features = weight.shape[0]
                
                if i == len(weight_keys) - 1:
                    out_features = n_actions

                layer = nn.Linear(prev_size, out_features)
                setattr(self, f'layer{i+1}', layer)
                
                prev_size = out_features
            
            self.num_layers = len(weight_keys)
    
    def forward(self, x: Tensor):
        for i in range(1, self.num_layers):
            layer = getattr(self, f'layer{i}')
            x = F.relu(layer(x))
        last_layer = getattr(self, f'layer{self.num_layers}')
        return last_layer(x)