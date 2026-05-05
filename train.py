"""
train.py — Main DQN Training Script

Usage:
    python train.py                        # fresh training
    python train.py --resume checkpoints/episode_200.pth
    python train.py --episodes 500 --lr 0.0001

Author: Herin Bhatt
"""

import os
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from src.environment import make_env, FrameStack, shape_reward, ACTIONS
from src.agent       import DQNAgent

# ── Default configuration ─────────────────────────────────────────────────────
CONFIG = {
    # Environment
    'frame_stack'       : 4,
    'frame_size'        : 84,

    # Training
    'total_episodes'    : 500,
    'max_steps'         : 1000,
    'batch_size'        : 64,
    'gamma'             : 0.99,
    'lr'                : 1e-4,
    'min_replay_size'   : 5000,
    'replay_buffer_size': 50000,

    # Epsilon-greedy
    'eps_start'         : 1.0,
    'eps_end'           : 0.05,
    'eps_decay_steps'   : 100000,

    # Target network
    'target_update_freq': 1000,

    # Logging
    'save_freq'         : 50,
    'plot_freq'         : 25,
    'solve_score'       : 700,
}


def plot_rewards(rewards, avg_rewards, episode):
    os.makedirs('plots', exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(rewards, alpha=0.35, color='steelblue', label='Episode reward')
    axes[0].plot(avg_rewards, color='tomato', lw=2, label='100-ep avg')
    axes[0].axhline(CONFIG['solve_score'], color='green', ls='--',
                    alpha=0.7, label=f"Solve ({CONFIG['solve_score']})")
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title(f'DQN CarRacing-v2 — Episode {episode}')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(avg_rewards, color='darkorange', lw=1.5)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('100-ep Avg Reward')
    axes[1].set_title('Learning progress')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/reward_curve.png', dpi=120, bbox_inches='tight')
    plt.close()
    print('  Plot saved: plots/reward_curve.png')


def train(config, resume_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')

    os.makedirs('checkpoints', exist_ok=True)

    env         = make_env()
    agent       = DQNAgent(config, device)
    frame_stack = FrameStack(config['frame_stack'], config['frame_size'])

    if resume_path:
        agent.load(resume_path)

    episode_rewards = []
    avg_rewards     = []
    best_avg        = -float('inf')
    start_time      = time.time()

    print('\n' + '='*65)
    print(' DQN Training — Autonomous Car Racing (CarRacing-v2)')
    print(f' Episodes : {config["total_episodes"]}')
    print(f' Gamma    : {config["gamma"]}   LR: {config["lr"]}')
    print(f' Buffer   : {config["replay_buffer_size"]:,} transitions')
    print(f' Warmup   : {config["min_replay_size"]:,} steps before training')
    print('='*65 + '\n')

    for episode in range(agent.episodes + 1, config['total_episodes'] + 1):
        obs, _       = env.reset()
        state        = frame_stack.reset(obs)
        total_reward = 0.0
        total_loss   = 0.0
        loss_count   = 0

        for step in range(config['max_steps']):
            action_idx = agent.select_action(state)
            action     = ACTIONS[action_idx]

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            shaped     = shape_reward(reward)
            next_state = frame_stack.step(next_obs)

            agent.store(state, action_idx, shaped, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                total_loss += loss
                loss_count += 1

            agent.total_steps += 1
            if agent.total_steps % config['target_update_freq'] == 0:
                agent.update_target()

            total_reward += reward
            state = next_state
            if done:
                break

        agent.episodes += 1
        episode_rewards.append(total_reward)
        avg = np.mean(episode_rewards[-100:])
        avg_rewards.append(avg)

        elapsed  = (time.time() - start_time) / 60
        avg_loss = total_loss / max(loss_count, 1)
        print(
            f'Ep {episode:4d} | '
            f'Reward {total_reward:7.1f} | '
            f'Avg(100) {avg:7.1f} | '
            f'eps {agent.epsilon:.3f} | '
            f'loss {avg_loss:.4f} | '
            f'steps {agent.total_steps:,} | '
            f'{elapsed:.1f}m'
        )

        # Save best
        if avg > best_avg and episode >= 100:
            best_avg = avg
            agent.save('checkpoints/best_model.pth')
            print(f'  *** New best avg: {avg:.1f} — saved best_model.pth')

        # Periodic checkpoint
        if episode % config['save_freq'] == 0:
            ckpt = f'checkpoints/episode_{episode}.pth'
            agent.save(ckpt)
            print(f'  Checkpoint: {ckpt}')

        # Plot
        if episode % config['plot_freq'] == 0:
            plot_rewards(episode_rewards, avg_rewards, episode)

        # Solved?
        if avg >= config['solve_score'] and episode >= 100:
            print(f'\n SOLVED at episode {episode}! Avg reward: {avg:.1f}')
            agent.save('checkpoints/solved_model.pth')
            break

    env.close()
    plot_rewards(episode_rewards, avg_rewards, episode)
    print(f'\nTraining complete. Best avg reward: {best_avg:.1f}')
    return agent, episode_rewards


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume',   type=str,   default=None)
    parser.add_argument('--episodes', type=int,   default=CONFIG['total_episodes'])
    parser.add_argument('--lr',       type=float, default=CONFIG['lr'])
    parser.add_argument('--gamma',    type=float, default=CONFIG['gamma'])
    args = parser.parse_args()

    CONFIG['total_episodes'] = args.episodes
    CONFIG['lr']             = args.lr
    CONFIG['gamma']          = args.gamma

    train(CONFIG, resume_path=args.resume)
