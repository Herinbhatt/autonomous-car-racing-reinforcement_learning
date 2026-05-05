"""
agent.py — Deep Q-Network Agent

Implements:
- Epsilon-greedy action selection (exploration vs exploitation)
- Bellman equation Q-update with online and target networks
- Experience replay via ReplayBuffer
- Checkpoint saving and loading

Bellman update:
    Target: y = r + gamma * max_a' Q_target(s', a')  [if not terminal]
            y = r                                      [if terminal]
    Loss:   Huber( Q_online(s, a) - y )

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.model         import DQNNetwork
from src.replay_buffer import ReplayBuffer
from src.environment   import ACTIONS, N_ACTIONS


class DQNAgent:
    """
    DQN Agent with:
    - Online network  (updated every training step)
    - Target network  (updated every target_update_freq steps — stabilizes training)
    - Epsilon-greedy exploration with linear decay
    """

    def __init__(self, config: dict, device: torch.device):
        self.config     = config
        self.device     = device
        self.n_actions  = N_ACTIONS

        # ── Networks ──────────────────────────────────────────────────────────
        self.online_net = DQNNetwork(
            config['frame_stack'], N_ACTIONS, config['frame_size']
        ).to(device)

        self.target_net = DQNNetwork(
            config['frame_stack'], N_ACTIONS, config['frame_size']
        ).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()   # Target net: inference only, never backprop

        # ── Optimizer ─────────────────────────────────────────────────────────
        self.optimizer = optim.Adam(
            self.online_net.parameters(), lr=config['lr']
        )

        # ── Replay buffer ─────────────────────────────────────────────────────
        self.replay = ReplayBuffer(config['replay_buffer_size'], device)

        # ── Counters ──────────────────────────────────────────────────────────
        self.total_steps = 0
        self.episodes    = 0
        self.losses      = []

    # ── Epsilon ───────────────────────────────────────────────────────────────
    @property
    def epsilon(self) -> float:
        """
        Linearly decay epsilon from eps_start to eps_end over eps_decay_steps.
        High epsilon early = lots of exploration.
        Low epsilon later  = mostly exploitation of learned policy.
        """
        cfg      = self.config
        progress = min(self.total_steps / cfg['eps_decay_steps'], 1.0)
        return cfg['eps_start'] + progress * (cfg['eps_end'] - cfg['eps_start'])

    # ── Action selection ──────────────────────────────────────────────────────
    def select_action(self, state, greedy: bool = False) -> int:
        """
        Epsilon-greedy policy:
        - If greedy=True or rand > epsilon: pick argmax Q(s, a)
        - Otherwise: random action
        """
        import random
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_vals  = self.online_net(state_t)
            return q_vals.argmax(dim=1).item()

    # ── Store transition ──────────────────────────────────────────────────────
    def store(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)

    # ── Training step ─────────────────────────────────────────────────────────
    def train_step(self):
        """
        One gradient update using a sampled mini-batch.

        Returns loss value or None if buffer not ready yet.
        """
        cfg = self.config
        if not self.replay.is_ready(cfg['min_replay_size']):
            return None

        states, actions, rewards, next_states, dones = \
            self.replay.sample(cfg['batch_size'])

        # ── Current Q-values: Q_online(s, a) for taken actions ────────────────
        current_q = self.online_net(states).gather(
            1, actions.unsqueeze(1)
        ).squeeze(1)

        # ── Target Q-values: r + gamma * max_a' Q_target(s', a') ─────────────
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1)[0]
            target_q   = rewards + cfg['gamma'] * max_next_q * (1.0 - dones)

        # ── Huber loss (robust to outliers vs MSE) ────────────────────────────
        loss = F.smooth_l1_loss(current_q, target_q)

        # ── Backprop ──────────────────────────────────────────────────────────
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10)
        self.optimizer.step()

        loss_val = loss.item()
        self.losses.append(loss_val)
        return loss_val

    # ── Target network sync ───────────────────────────────────────────────────
    def update_target(self):
        """Hard copy: online → target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ── Checkpoint ────────────────────────────────────────────────────────────
    def save(self, path: str):
        torch.save({
            'online_net'  : self.online_net.state_dict(),
            'target_net'  : self.target_net.state_dict(),
            'optimizer'   : self.optimizer.state_dict(),
            'total_steps' : self.total_steps,
            'episodes'    : self.episodes,
            'config'      : self.config,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt['online_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.total_steps = ckpt.get('total_steps', 0)
        self.episodes    = ckpt.get('episodes', 0)
        print(f'Loaded: episode {self.episodes}, steps {self.total_steps:,}')
