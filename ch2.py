import typing as tt
from collections import Counter, defaultdict

import gymnasium as gym
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter

print("Hello from Gymnasium")

e = gym.make("CartPole-v1")

obs, info = e.reset()

print(f"Obs: {obs}")

print(e.action_space)

print(e.observation_space)

print(e.step(0))

if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = gym.wrappers.HumanRendering(env)
    total_reward = 0.0
    total_steps = 0
    obs, _ = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, is_done, is_trunc, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if is_done:
            break
    print("Episode done in %d steps, total reward %.2f" % (total_steps, total_reward))
