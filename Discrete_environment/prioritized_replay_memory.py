from collections import namedtuple
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", ["state", "action", "next_state", "reward", "done"])

class PrioritizedReplayMemory:
    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4, beta_frames: int = 100000):
        """
        Prioritized Experience Replay Memory for LunarLander (vector observations).
        
        Args:
            capacity: Buffer size, maximum number of transitions stored.
            alpha: Prioritization exponent (0 = uniform sampling, 1 = full prioritization).
            beta_start: Initial value for importance sampling correction (prevents bias).
            beta_frames: Number of frames for beta to reach 1.0 linearly.
            
        How it works:
            - Transitions with high TD errors are sampled more frequently.
            - As the agent learns, TD errors decrease naturally.
            - Beta increases over time to correct sampling bias.
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame_counter = 1
        
        self.memory = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0  # Current position in circular buffer

    def beta_by_frame(self, frame_idx: int) -> float:
        """
        Compute beta (importance sampling correction) at current frame.
        
        Beta increases linearly from beta_start to 1.0 because:
        - Early training: priorities are imprecise, use less correction
        - Late training: priorities are accurate, use full correction
        
        Example:
            frame_idx=0:       beta = 0.4 (40% correction)
            frame_idx=50000:   beta = 0.7 (70% correction)
            frame_idx=100000+: beta = 1.0 (100% correction)
        """
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def push(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, 
             reward: torch.Tensor, done: bool) -> None:
        """
        Add a new transition with maximum priority.
        
        Args:
            state: Current state tensor (e.g., shape [1, 8] for LunarLander)
            action: Action taken (tensor)
            next_state: Next state tensor (or None if terminal)
            reward: Reward received (tensor)
            done: Whether episode terminated
        
        Why maximum priority?
            We don't know the TD error yet, so assign max priority to guarantee
            the transition is sampled at least once during training.
        """
        max_priority = self.priorities.max() if len(self.memory) > 0 else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(state, action, next_state, reward, done))
        else:
            # Overwrite oldest transition (circular buffer)
            self.memory[self.position] = Transition(state, action, next_state, reward, done)
            
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> tuple:
        """
        Sample a batch of transitions according to priorities.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            samples: List of Transition namedtuples
            indices: Array of sampled indices (for priority updates)
            weights: Importance sampling weights (tensor on DEVICE)
        
        Importance sampling weights:
            High-priority transitions are sampled more often, which introduces bias.
            Weights correct this bias by down-weighting frequent samples.
            Formula: weight = (N * P(i))^(-beta)
        """
        # Get valid priorities
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.memory)]
        
        # Compute sampling probabilities: P(i) = priority(i)^alpha / sum(priorities^alpha)
        sampling_probabilities = priorities ** self.alpha
        sampling_probabilities /= sampling_probabilities.sum()
        
        # Sample indices according to probabilities
        indices = np.random.choice(len(self.memory), batch_size, p=sampling_probabilities, replace=False)
        samples = [self.memory[idx] for idx in indices]
        
        # Compute importance sampling weights
        total = len(self.memory)
        beta = self.beta_by_frame(self.frame_counter)
        self.frame_counter += 1
        
        # Weight formula: (N * P(i))^(-beta)
        # Normalize so max weight = 1.0
        weights = (total * sampling_probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, indices, torch.FloatTensor(weights).to(DEVICE)
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        Update priorities based on TD errors after training step.
        
        Args:
            indices: Indices of sampled transitions
            td_errors: Absolute TD errors |Q_target - Q_current|
        
        Why update priorities?
            - High TD error = surprising transition = learn more from it
            - Low TD error = expected transition = already learned
            
        As the network learns, TD errors naturally decrease, and priorities
        are automatically adjusted to focus on harder examples.
        """
        for idx, priority in zip(indices, td_errors):
            # Add small epsilon to avoid zero priority (would never be sampled again)
            self.priorities[idx] = priority + 1e-6
    
    def __len__(self) -> int:
        """Return current number of transitions in memory."""
        return len(self.memory)