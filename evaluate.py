"""
evaluate.py — Evaluate a trained DQN agent 

Usage:
    python evaluate.py --model checkpoints/best_model.pth --episodes 10

Author: Herin Bhatt
"""

import argparse
import numpy as np
import torch

from src.environment import make_env, FrameStack, ACTIONS
from src.agent       import DQNAgent
from train           import CONFIG


def evaluate(model_path: str, n_episodes: int = 10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    agent = DQNAgent(CONFIG, device)
    agent.load(model_path)
    agent.online_net.eval()

    env          = make_env()
    frame_stack  = FrameStack(CONFIG['frame_stack'], CONFIG['frame_size'])
    eval_rewards = []

    print(f'\nEvaluating {n_episodes} episodes (greedy policy)...\n')
    for ep in range(n_episodes):
        obs, _       = env.reset()
        state        = frame_stack.reset(obs)
        total_reward = 0.0
        done         = False

        while not done:
            action_idx = agent.select_action(state, greedy=True)
            obs, reward, terminated, truncated, _ = env.step(ACTIONS[action_idx])
            done          = terminated or truncated
            state         = frame_stack.step(obs)
            total_reward += reward

        eval_rewards.append(total_reward)
        print(f'  Episode {ep+1:3d}: reward = {total_reward:.1f}')

    env.close()
    print(f'\nResults over {n_episodes} episodes:')
    print(f'  Mean:   {np.mean(eval_rewards):.1f}')
    print(f'  Std:    {np.std(eval_rewards):.1f}')
    print(f'  Min:    {np.min(eval_rewards):.1f}')
    print(f'  Max:    {np.max(eval_rewards):.1f}')
    return eval_rewards


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',    type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--episodes', type=int, default=10)
    args = parser.parse_args()
    evaluate(args.model, args.episodes)
