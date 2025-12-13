import numpy as np
import torch
from collections import namedtuple

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", ["state", "action", "next_state", "reward", "done"])

class PrioritizedReplayMemory:
    def __init__(self, capacity: int, alpha: float = 0.3, beta_start: float = 0.6, 
                 beta_frames: int = 10000, eps: float = 1e-5):
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

    def push(self, state: torch.Tensor, action: torch.Tensor, next_state, 
             reward: torch.Tensor, done: bool) -> None:
        """
        Add transition to memory. Handles terminal states correctly by allowing next_state to be None.
        
        Args:
            state: Current state tensor
            action: Action taken tensor
            next_state: Next state tensor (can be None for terminal states)
            reward: Reward received tensor
            done: Whether episode terminated
        """
        # Get current max priority or use default 1.0
        if len(self.memory) > 0:
            max_priority = self.priorities[:len(self.memory)].max()
            if max_priority <= 0:
                max_priority = 1.0
        else:
            max_priority = 1.0

        # Store transition - handle terminal states correctly
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(state.cpu(), action.cpu(), next_state.cpu() if next_state is not None else None, reward.cpu(), done))
        else:
            self.memory[self.position] = Transition(state.cpu(), action.cpu(), next_state.cpu() if next_state is not None else None, reward.cpu(), done)

        # Set priority
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> tuple:
        """
        Sample batch according to priorities with importance sampling correction.
        
        Args:
            batch_size: Number of samples to return
            
        Returns:
            tuple: (samples, indices, weights)
        """
        n = len(self.memory)
        if n == 0:
            raise ValueError("Cannot sample from empty replay memory")

        # Get valid priorities and add epsilon to avoid zeros
        priorities = self.priorities[:n] + self.eps
        
        # Apply prioritization exponent alpha
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        
        # Handle case where batch_size > memory size
        replace = batch_size > n
        indices = np.random.choice(n, min(batch_size, n), p=probs, replace=replace)
        samples = [self.memory[idx] for idx in indices]

        # Compute importance sampling weights with beta annealing
        beta = self.beta_by_frame(self.frame_counter)
        self.frame_counter += 1

        weights = (n * probs[indices]) ** (-beta)
        weights = weights / weights.max()  # Normalize for stability
        
        weights_tensor = torch.FloatTensor(weights).to(DEVICE)

        return samples, indices, weights_tensor

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        Update priorities based on TD errors.
        
        Args:
            indices: Array of indices to update
            td_errors: Array of TD errors for those indices
        """
        # Ensure td_errors is numpy array
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.cpu().numpy()
        
        td_errors = np.asarray(td_errors).flatten()
        
        # Update priorities with TD errors + epsilon
        for idx, td_error in zip(indices, td_errors):
            priority = float(abs(td_error)) + self.eps
            self.priorities[int(idx)] = priority

    def __len__(self) -> int:
        """Return current size of memory"""
        return len(self.memory)