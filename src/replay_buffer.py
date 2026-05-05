"""
replay_buffer.py — Experience Replay Memory for DQN

Stores (state, action, reward, next_state, done) transitions
and provides random mini-batch sampling to break temporal correlation.

"""

import random
import collections
import numpy as np
import torch


class ReplayBuffer:
    """
    Fixed-capacity circular replay buffer.

    Why it matters:
    - Breaks correlation between consecutive transitions
    - Allows rare experiences to be replayed multiple times
    - Core component of DQN (Mnih et al., 2015)
    """

    def __init__(self, capacity: int, device: torch.device):
        self.buffer   = collections.deque(maxlen=capacity)
        self.capacity = capacity
        self.device   = device

    def push(self,
             state:      np.ndarray,
             action:     int,
             reward:     float,
             next_state: np.ndarray,
             done:       bool):
        """Store a single transition."""
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size: int):
        """
        Randomly sample a batch of transitions.

        Returns:
            Tuple of tensors: (states, actions, rewards, next_states, dones)
            All moved to self.device and ready for training.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)).to(self.device),
            torch.LongTensor(actions).to(self.device),
            torch.FloatTensor(rewards).to(self.device),
            torch.FloatTensor(np.array(next_states)).to(self.device),
            torch.FloatTensor(dones).to(self.device),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, min_size: int) -> bool:
        return len(self) >= min_size
