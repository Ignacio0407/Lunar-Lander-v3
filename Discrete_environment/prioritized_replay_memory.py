from collections import namedtuple
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", ["state", "action", "next_state", "reward", "done"])

class PrioritizedReplayMemory:
    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4, 
                 beta_frames: int = 100000, eps: float = 1e-6):
        """
        Prioritized Experience Replay Memory for LunarLander.
        
        Args:
            capacity: Maximum number of transitions stored
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling correction
            beta_frames: Frames for beta to reach 1.0
            eps: Small constant to avoid zero priorities
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame_counter = 1
        self.eps = eps

        self.memory = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def beta_by_frame(self, frame_idx: int) -> float:
        """Linear annealing of beta from beta_start to 1.0"""
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, 
             reward: torch.Tensor, done: bool) -> None:
        """Add transition with maximum priority to ensure it's sampled at least once."""
        if len(self.memory) > 0:
            # ✅ Get max priority from valid entries
            max_priority = self.priorities[:len(self.memory)].max()
            if max_priority <= 0:
                max_priority = 1.0
        else:
            max_priority = 1.0

        if len(self.memory) < self.capacity:
            self.memory.append(Transition(state, action, next_state, reward, done))
        else:
            self.memory[self.position] = Transition(state, action, next_state, reward, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> tuple:
        """Sample batch according to priorities with importance sampling correction."""
        n = len(self.memory)
        if n == 0:
            raise ValueError("Cannot sample from empty replay memory")

        # ✅ Get valid priorities
        priorities = self.priorities[:n].copy()
        
        # ✅ Add epsilon to avoid zeros
        priorities = priorities + self.eps
        
        # ✅ Apply alpha exponent for sampling
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        # ✅ Handle case where batch_size > memory size
        replace = batch_size > n
        indices = np.random.choice(n, min(batch_size, n), p=probs, replace=replace)
        samples = [self.memory[idx] for idx in indices]

        # ✅ Compute importance sampling weights
        beta = self.beta_by_frame(self.frame_counter)
        self.frame_counter += 1

        weights = (n * probs[indices]) ** (-beta)
        weights = weights / weights.max()  # Normalize
        
        weights_tensor = torch.FloatTensor(weights).to(DEVICE)

        return samples, indices, weights_tensor

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        Update priorities based on TD errors.
        """
        # Convert to numpy if needed
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.cpu().numpy()
        
        td_errors = np.asarray(td_errors).flatten()
        
        for idx, td_error in zip(indices, td_errors):
            priority = float(abs(td_error)) + self.eps
            self.priorities[int(idx)] = priority

    def __len__(self) -> int:
        return len(self.memory)