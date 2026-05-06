# 🏎️ Autonomous Car Racing with Deep Q-Network (DQN)

> Training a neural network agent to drive a racing car using Deep Reinforcement Learning — implemented from scratch in Python with PyTorch and OpenAI Gymnasium.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch)
![Gymnasium](https://img.shields.io/badge/Gymnasium-CarRacing--v2-green?style=flat-square)
![RL](https://img.shields.io/badge/Algorithm-DQN-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## Overview

This project trains an AI agent to autonomously drive a car around a procedurally generated racetrack using **Deep Q-Network (DQN)** — the algorithm introduced by DeepMind in 2015 that achieved superhuman performance on Atari games.

The agent receives raw **96×96 pixel frames** from OpenAI Gymnasium's `CarRacing-v2` environment and learns — purely through trial and error — to steer, accelerate, and brake in order to complete laps. No human demonstrations. No labeled data. Only a reward signal.

---

## Demo
```
videos/dqn_carracing-episode-0.mp4      ← replace with GIF or video embed after training
plots/reward_curve.png
```

---

## Algorithm — Why DQN?

Standard tabular Q-learning stores one Q-value per (state, action) pair. With a 96×96×3 pixel input, the state space is astronomically large — a lookup table is impossible.

**DQN solves this** by using a Convolutional Neural Network to *approximate* the Q-function, allowing the agent to generalize across visually similar states.

### The Bellman update

```
Q(s, a)  ←  Q(s, a)  +  α · [ r  +  γ · max Q(s', a')  −  Q(s, a) ]
                                     └── target network ──┘
```

Two innovations that make DQN stable:

| Innovation | Why it matters |
|---|---|
| **Experience replay** | Breaks temporal correlation — stores 50k transitions, samples randomly |
| **Target network** | Frozen copy of the network used for computing targets, synced every 1,000 steps — prevents the "moving target" problem |

---

## Project structure

```
autonomous-car-racing-dqn/
│
├── src/
│   ├── model.py           CNN Q-Network (PyTorch)
│   ├── agent.py           DQN agent — epsilon-greedy, Bellman update, checkpointing
│   ├── replay_buffer.py   Experience replay memory
│   └── environment.py     Frame preprocessing, frame stacking, reward shaping
│
├── train.py               Main training loop with logging and checkpointing
├── evaluate.py            Run trained agent (greedy policy), report reward stats
├── record_agent.py        Record MP4 video of trained agent driving
├── DQN_CarRacing.ipynb    Google Colab notebook (full training in one file)
│
├── plot/
│   ├── reward_curve.png   Training reward history (add after training)
│ 
├── videos/
│   ├── dqn_carracing-episode-0.mp4   Video of trained agent (add after training)
│           
│
├── requirements.txt
└── README.md
```

---

## CNN Architecture

```
Input:  (batch, 4, 84, 84)   ← 4 stacked grayscale frames

Conv2d(4  → 32, kernel=8, stride=4)  + ReLU   → (batch, 32, 20, 20)
Conv2d(32 → 64, kernel=4, stride=2)  + ReLU   → (batch, 64, 9,  9 )
Conv2d(64 → 64, kernel=3, stride=1)  + ReLU   → (batch, 64, 7,  7 )

Flatten  →  3136

Linear(3136 → 512) + ReLU
Linear(512  → 5)               ← Q-value for each of 5 actions
```

### Discrete action space

| Index | Action | Continuous values |
|---|---|---|
| 0 | Coast (no-op) | [0.0, 0.0, 0.0] |
| 1 | Steer left | [-0.6, 0.0, 0.0] |
| 2 | Steer right | [0.6, 0.0, 0.0] |
| 3 | Accelerate | [0.0, 1.0, 0.0] |
| 4 | Brake | [0.0, 0.0, 0.8] |

---

## Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Learning rate α | 0.0001 | Adam optimizer, conservative for stability |
| Discount factor γ | 0.99 | High — agent values future rewards strongly |
| Epsilon start | 1.0 | Full exploration at the beginning |
| Epsilon end | 0.05 | 5% random actions maintained throughout |
| Epsilon decay | 100,000 steps | Linear decay over first 100k steps |
| Replay buffer | 50,000 | Large enough to decorrelate samples |
| Batch size | 64 | Standard for GPU training |
| Target update | 1,000 steps | Hard copy every 1k steps |
| Frame stack | 4 | Captures motion/velocity information |
| Frame size | 84×84 | Standard DQN preprocessing (Mnih 2015) |

---

## Installation and usage

```bash
# Clone
git clone https://github.com/your-username/autonomous-car-racing-reinforcement_learning
cd autonomous-car-racing-reinforcement_learning

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# Install
pip install -r requirements.txt
```

### Train
```bash
python train.py                                          # fresh training
python train.py --resume episode_0200.pth    # resume
python train.py --episodes 500 --lr 0.0001              # custom params
```

### Evaluate
```bash
python evaluate.py --model best_model.pth --episodes 10
```

### Record video
```bash
python record_agent.py --model best_model.pth
```

### Google Colab
Open `DQN_CarRacing.ipynb` in [Google Colab](https://colab.research.google.com) with a T4 GPU — no local setup needed.

---

## Training progress

| Phase | Episodes | What to expect |
|---|---|---|
| Early | 1 – 100 | Agent drives randomly, very low reward |
| Learning | 100 – 300 | Agent starts following the track |
| Improving | 300 – 600 | Agent completes partial laps |
| Solving | 600 – 1000 | Agent completes full laps consistently |

Expected training time: ~3–5 hours on Google Colab T4 GPU.

---

## Results

| Metric | Value |
|---|---|
| Best episode reward | *Add after training* |
| Average reward (last 100 episodes) | *Add after training* |
| Episodes to first complete lap | *Add after training* |
| Training time | *Add after training* |

> Reward curve: `docs/reward_curve.png`

---

## Key learnings

**Frame stacking is essential.** A single frame gives no velocity information. Stacking 4 consecutive grayscale frames allows the CNN to infer the car's speed and direction of motion.

**Target network stability.** Without a frozen target network, the Q-targets shift with every update, causing training to diverge. The target network is what separates DQN from vanilla Q-learning with neural networks.

**Reward shaping accelerates convergence.** CarRacing's default reward is sparse. Amplifying the off-track penalty encourages the agent to stay on the road rather than randomly drifting.

**Exploration-exploitation trade-off is critical.** Decaying epsilon too fast causes the agent to commit to a suboptimal policy before it has seen enough of the state space. Decaying too slowly wastes training time on random actions.

---

## Connection to other projects

| Project | Paradigm | Algorithm | Language |
|---|---|---|---|
| [Crop & Weed Detection](https://github.com/your-username/crop-weed-detection) | Supervised learning | YOLOv8 + Transfer Learning | Python |
| [Robot Navigation (JS)](https://github.com/your-username/adaptive-robot-navigation-rl) | Tabular RL | Q-learning (Bellman table) | JavaScript |
| **Autonomous Car Racing (this)** | **Deep RL** | **DQN + CNN** | **Python** |

The progression is intentional — from labeled data → table-based RL → neural network RL. Each project builds on concepts from the previous one.

---

## References

- Mnih, V. et al. (2015). [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236). *Nature*, 518, 529–533.
- Sutton, R. & Barto, A. (2018). [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html). MIT Press.
- [OpenAI Gymnasium — CarRacing-v2](https://gymnasium.farama.org/environments/box2d/car_racing/)

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

*MSc AI/ML Portfolio Project — Herin Bhatt, 2025*
