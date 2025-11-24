from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import re

class DQN_dynamic(nn.Module):
    def __init__(self, n_observations: int, n_actions: int, state_dict: dict = None):
        super().__init__()
        
        if state_dict is None:
            # Arquitectura por defecto
            self.layer1 = nn.Linear(n_observations, 128)
            self.layer2 = nn.Linear(128, 128)
            self.layer3 = nn.Linear(128, n_actions)
            self.num_layers = 3
        else:
            weight_keys = [k for k in state_dict.keys() if 'weight' in k and 'layer' in k]
            
            def get_layer_number(key):
                return int(re.search(r'layer(\d+)', key).group(1))
            
            weight_keys.sort(key=get_layer_number)
            
            first_weight = state_dict[weight_keys[0]]
            first_out_features = first_weight.shape[0]
            self.layer1 = nn.Linear(n_observations, first_out_features)
            
            prev_features = first_out_features
            for i in range(1, len(weight_keys) - 1):
                weight_key = weight_keys[i]
                weight = state_dict[weight_key]
                out_features = weight.shape[0]
                setattr(self, f'layer{i+1}', nn.Linear(prev_features, out_features))
                prev_features = out_features

            last_layer_idx = len(weight_keys)
            setattr(self, f'layer{last_layer_idx}', nn.Linear(prev_features, n_actions))
            
            self.num_layers = len(weight_keys)
    
    def forward(self, x: Tensor):
        for i in range(1, self.num_layers):
            layer = getattr(self, f'layer{i}')
            x = F.relu(layer(x))
        output_layer = getattr(self, f'layer{self.num_layers}')
        return output_layer(x)