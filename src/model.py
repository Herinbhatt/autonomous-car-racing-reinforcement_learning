"""
model.py — CNN Q-Network for DQN CarRacing

Architecture:
    Input:  (batch, 4, 84, 84)  — 4 stacked grayscale frames
    Conv1:  32 filters, 8x8, stride 4
    Conv2:  64 filters, 4x4, stride 2
    Conv3:  64 filters, 3x3, stride 1
    Flatten → FC(512) → FC(n_actions)

Author: Herin Bhatt
"""

import numpy as np
import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """
    Convolutional Q-Network that maps raw pixel observations
    to Q-values for each discrete action.

    Based on: Mnih et al. (2015) — Human-level control through
    deep reinforcement learning. Nature, 518, 529-533.
    """

    def __init__(self, n_frames: int, n_actions: int, frame_size: int = 84):
        super(DQNNetwork, self).__init__()
        self.n_frames  = n_frames
        self.n_actions = n_actions
        self.frame_size = frame_size

        # Convolutional feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(n_frames, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        # Compute flattened size dynamically
        conv_out = self._get_conv_out()

        # Fully connected value head
        self.fc = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_actions),
        )

        # Weight initialization
        self._init_weights()

    def _get_conv_out(self) -> int:
        dummy = torch.zeros(1, self.n_frames, self.frame_size, self.frame_size)
        out   = self.conv(dummy)
        return int(np.prod(out.shape))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_frames, H, W) float tensor in [0, 1]
        Returns:
            Q-values: (batch, n_actions) float tensor
        """
        features = self.conv(x)
        flat      = features.view(features.size(0), -1)
        return self.fc(flat)
