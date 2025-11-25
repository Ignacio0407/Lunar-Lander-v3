from collections import namedtuple
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", ["state", "action", "next_state", "reward", "done"])

class PrioritizedReplayMemory:
    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4, beta_frames: int = 100000):
        """Args:
            capacity: Buffer size, how many transitions are stored.
            alpha: Importance given to prioritization (0 = uniforme, 1 = solo prioridades)
            beta_start: Intital value for importance sampling, since we do not know the best actions at the beginning (prevents bias)
            beta_frames: Frames for beta to reach 1.
            It works based on TD errors, if there is one quite high, it learns from it and decreases its TD error a little bit
            because it has learned something from it and therefore is less interesting after that. The decrementation is
            performed during the training, is not in this class.
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame_counter = 1
        
        self.memory = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0  # Actual position in buffer, since it is circular.

    def beta_by_frame(self, frame_idx):
        """
        Beta rises linearly from beta_start to 1.0 because at the beginning of the training, priorities are (really) imprecise.
        As the network trains more, the priorities become more precise, and thus importance sampling increases in accordance.
        Example: frame_idx=0: beta = 0.4, frame_idx=50000:   beta = 0.7, frame_idx=100000:  beta = 1.0
        """
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def push(self, *args):
        """
        Adds a new transition with maximum priority. 
        ¿Why max priority? Because we do not yer know its TD error. Guarantees that it is sampled at least once.
        """
        max_priority = self.priorities.max() if len(self.memory) > 0 else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(*args))
        else:
            self.memory[self.position] = Transition(*args) # Rewrite oldest position (circular buffer)
            
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """Returns:
            samples: Transition list
            indices: Transitions indices (to update priorities)
            weights: importance sampling weights for correcting biases.
        """
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.memory)]
        # Assign probability (between 0 and 1) based on priority.
        sampling_probabilities = priorities ** self.alpha
        sampling_probabilities /= sampling_probabilities.sum()
        
        # Actually choosing which transitions are going to be sampled
        indices = np.random.choice(len(self.memory), batch_size, p=sampling_probabilities, replace=False)
        samples = [self.memory[idx] for idx in indices]
        
        # 4. Calculate importance sampling weights because priority sampling introduces BIAS, meaning if a transition has high
        # priority, it will be selected more and its gradient has more importance than it should.
        total = len(self.memory)
        beta = self.beta_by_frame(self.frame_counter)
        self.frame += 1
        weights = (total * sampling_probabilities[indices]) ** (-beta)
        weights /= weights.max() # Normalize (between 0 and 1)
        
        return samples, indices, torch.FloatTensor(weights).to(DEVICE)
    
    def update_priorities(self, indices, td_errors):
        '''Updates priorities after calculating TD errors (TD errors = priority here)'''
        for idx, priority in zip(indices, td_errors):
            self.priorities[idx] = priority + 1e-6
    
    def __len__(self):
        return len(self.memory)