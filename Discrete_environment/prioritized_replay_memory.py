from collections import namedtuple
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", ["state", "action", "next_state", "reward", "done"])

class PrioritizedReplayMemory:
    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4, beta_frames: int = 100000, eps: float = 1e-6):
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
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, reward: torch.Tensor, done: bool) -> None:
        # Ensure new transitions get a positive priority (max current or 1.0)
        if len(self.memory) > 0:
            max_priority = float(self.priorities[:len(self.memory)].max())
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
        # Get valid priorities slice
        n = len(self.memory)
        if n == 0:
            raise ValueError("Cannot sample from an empty replay memory")

        priorities = self.priorities[:n].astype(np.float64)  # higher precision for power ops

        # Add small epsilon to avoid zeros
        priorities = priorities + self.eps

        # Compute sampling probabilities
        sampling_probabilities = priorities ** self.alpha
        sum_p = sampling_probabilities.sum()
        if sum_p <= 0 or not np.isfinite(sum_p):
            # Fallback to uniform if something went wrong
            sampling_probabilities = np.ones_like(sampling_probabilities) / len(sampling_probabilities)
        else:
            sampling_probabilities = sampling_probabilities / sum_p

        # If memory < batch_size, allow replacement
        replace = False
        if batch_size > n:
            replace = True

        indices = np.random.choice(n, batch_size, p=sampling_probabilities, replace=replace)
        samples = [self.memory[idx] for idx in indices]

        # Importance sampling weights
        beta = self.beta_by_frame(self.frame_counter)
        self.frame_counter += 1

        probs = sampling_probabilities[indices]
        # Avoid zeros in probs
        probs = np.maximum(probs, self.eps)

        weights = (n * probs) ** (-beta)
        # Normalize weights so max = 1
        max_w = weights.max()
        if max_w > 0 and np.isfinite(max_w):
            weights = weights / max_w
        else:
            weights = np.ones_like(weights, dtype=np.float32)

        weights_t = torch.from_numpy(weights.astype(np.float32)).to(DEVICE)

        return samples, indices, weights_t

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        # td_errors expected shape: (batch_size,) and dtype float32
        # Convert to numpy array if it's a tensor
        if hasattr(td_errors, "tolist"):
            # if it's numpy already it's fine; if torch tensor, convert prior
            try:
                td_errors = np.array(td_errors)
            except Exception:
                td_errors = np.asarray(td_errors)

        # Ensure 1D array
        td_errors = np.asarray(td_errors).reshape(-1)

        for idx, err in zip(indices, td_errors):
            p = float(abs(err)) + self.eps
            # store raw priority (we'll apply alpha at sampling time)
            self.priorities[int(idx)] = p

    def __len__(self) -> int:
        return len(self.memory)