from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class DQN_dynamic(nn.Module):
    def __init__(self, n_observations:int, n_actions:int, state_dict:dict):
        super().__init__()
        input_size = n_observations

        # Detectar capas en orden según el checkpoint
        layer_sizes = []
        for key, value in state_dict.items():
            if "weight" in key and "layer" in key:
                layer_sizes.append((key, value.shape))

        # Ordenar por índice de capa (layer1, layer2, ...)
        layer_sizes.sort(key=lambda x: int(x[0].split("layer")[1].split(".")[0]))

        # Construir todas las capas con los tamaños detectados
        for i, (_, shape) in enumerate(layer_sizes):
            out_features, in_features = shape
            layer = nn.Linear(in_features, out_features)
            setattr(self, f"layer{i+1}", layer)

        self.num_layers = len(layer_sizes)

    def forward(self, x: Tensor):
        # Pasar por todas las capas ocultas con ReLU
        for i in range(1, self.num_layers):
            layer = getattr(self, f"layer{i}")
            x = F.relu(layer(x))
        # Última capa sin activación
        output_layer = getattr(self, f"layer{self.num_layers}")
        return output_layer(x)