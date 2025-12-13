import torch

class TorchPER:
    def __init__(self, capacity: int, state_shape, device, alpha: float = 0.6, beta_start: float = 0.4, beta_frames: int = 100_000):
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames

        self.pos = 0
        self.size = 0
        self.frame = 1

        self.states = torch.zeros((capacity, *state_shape), device=device)
        self.actions = torch.zeros((capacity, 1), dtype=torch.long, device=device)
        self.rewards = torch.zeros((capacity, 1), device=device)
        self.next_states = torch.zeros((capacity, *state_shape), device=device)
        self.dones = torch.zeros((capacity, 1), device=device)
        self.priorities = torch.zeros(capacity, device=device)

    def beta_by_frame(self):
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.size > 0 else 1.0

        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        self.priorities[self.pos] = max_prio

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        prios = self.priorities[:self.size]
        probs = prios.pow(self.alpha)
        probs /= probs.sum()
        indices = torch.multinomial(probs, batch_size, replacement=False)

        beta = self.beta_by_frame()
        self.frame += 1

        weights = (self.size * probs[indices]).pow(-beta)
        weights /= weights.max()

        return (self.states[indices], self.actions[indices], self.rewards[indices], self.next_states[indices],
                self.dones[indices], indices, weights.unsqueeze(1))

    def update_priorities(self, indices, td_errors):
        self.priorities[indices] = td_errors.abs().squeeze() + 1e-6