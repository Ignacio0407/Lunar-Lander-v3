from collections import deque, namedtuple
import random

import numpy as np

Transition = namedtuple("Transition", ["state", "action", "next_state", "reward", "done"])

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, min(batch_size, len(self.memory)))
    
    def __len__(self):
        return len(self.memory)
    
class PrioritizedReplayMemory:
    """
    Prioritized replay memory: samples transitions with higher TD-error more frequently
    """
    def __init__(self, capacity, alpha=0.6):
        self.memory = deque([], maxlen=capacity)
        self.priorities = deque([], maxlen=capacity)
        self.alpha = alpha  # Qué tanto priorizar (0 = uniforme, 1 = totalmente por prioridad)
        self.epsilon = 1e-6  # Para evitar prioridad cero
    
    def push(self, *args, priority=None):
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.memory.append(Transition(*args))
        self.priorities.append(priority if priority else max_priority)
    
    def sample(self, batch_size, beta=0.4):
        """
        beta: bias importance compensation (0 = no compensation, 1 = full compensation)
        """
        if len(self.memory) == 0:
            return [], [], []
        
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.memory), min(batch_size, len(self.memory)), p=probs, replace=False)
        samples = [self.memory[idx] for idx in indices]
        
        # Importance weights to correct bias
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
    
    def __len__(self):
        return len(self.memory)