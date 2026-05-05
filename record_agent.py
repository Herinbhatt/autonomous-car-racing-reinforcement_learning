"""
Usage:
    python record_agent.py --model checkpoints/best_model.pth

Output: videos/dqn_carracing_ep0.mp4s
"""

import os
import argparse
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from src.environment import FrameStack, ACTIONS
from src.agent       import DQNAgent
from train           import CONFIG


def record(model_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    agent = DQNAgent(CONFIG, device)
    agent.load(model_path)
    agent.online_net.eval()

    os.makedirs('videos', exist_ok=True)

    env = gym.make('CarRacing-v2', continuous=False, render_mode='rgb_array')
    env = RecordVideo(env, video_folder='videos',
                      name_prefix='dqn_carracing',
                      episode_trigger=lambda ep: True)

    frame_stack  = FrameStack(CONFIG['frame_stack'], CONFIG['frame_size'])
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

    env.close()
    print(f'Video saved to videos/  |  Episode reward: {total_reward:.1f}')
    print('Copy videos/*.mp4 to docs/demo.mp4 for your GitHub README.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth')
    args = parser.parse_args()
    record(args.model)
