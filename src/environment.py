"""
environment.py — Frame preprocessing and environment utilities

"""

import collections
import numpy as np
import cv2
import gymnasium as gym


# ── Discrete action map ───────────────────────────────────────────────────────
# CarRacing-v2 native action: [steering (-1..1), gas (0..1), brake (0..1)]
# We map these to 5 discrete actions the DQN can learn over.
ACTIONS = [
    [0.0,  0.0,  0.0],    # 0: No-op (coast)
    [-0.6, 0.0,  0.0],    # 1: Steer left
    [0.6,  0.0,  0.0],    # 2: Steer right
    [0.0,  1.0,  0.0],    # 3: Accelerate
    [0.0,  0.0,  0.8],    # 4: Brake
]
ACTION_NAMES = ['Coast', 'Left', 'Right', 'Gas', 'Brake']
N_ACTIONS    = len(ACTIONS)


def preprocess_frame(frame: np.ndarray, frame_size: int = 84) -> np.ndarray:
    """
    Convert a raw 96x96 RGB frame to a normalized grayscale frame.

    Args:
        frame:      (96, 96, 3) uint8 numpy array
        frame_size: target size (default 84)
    Returns:
        (frame_size, frame_size) float32 array in [0, 1]
    """
    gray    = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (frame_size, frame_size),
                         interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


class FrameStack:
    """
    Maintains a sliding window of the N most recent processed frames.

    Why frame stacking?
    A single frame contains no velocity information. By stacking 4 frames,
    the network can infer motion direction and speed — essential for car
    racing where braking requires predicting future position.
    """

    def __init__(self, n_frames: int, frame_size: int = 84):
        self.n_frames   = n_frames
        self.frame_size = frame_size
        self.frames     = collections.deque(maxlen=n_frames)

    def reset(self, obs: np.ndarray) -> np.ndarray:
        """Initialize stack by repeating the first frame N times."""
        frame = preprocess_frame(obs, self.frame_size)
        for _ in range(self.n_frames):
            self.frames.append(frame)
        return self._state()

    def step(self, obs: np.ndarray) -> np.ndarray:
        """Push new frame and return current stacked state."""
        frame = preprocess_frame(obs, self.frame_size)
        self.frames.append(frame)
        return self._state()

    def _state(self) -> np.ndarray:
        # Shape: (n_frames, frame_size, frame_size)
        return np.array(self.frames, dtype=np.float32)


def shape_reward(reward: float) -> float:
    """
    Reward shaping to accelerate learning.

    CarRacing default reward: +1000/N per track tile visited, -0.1 per step.
    Grass (off-track) gives negative reward.

    We amplify the grass penalty to discourage going off-track.
    """
    if reward < 0:
        return reward * 2.0   # Amplify off-track penalty
    return reward


def make_env(render_mode: str = 'rgb_array') -> gym.Env:
    """Create and return a configured CarRacing-v2 environment."""
    return gym.make(
        'CarRacing-v2',
        continuous=False,
        render_mode=render_mode,
    )
