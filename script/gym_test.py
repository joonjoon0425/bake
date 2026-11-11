import gymnasium as gym
import numpy as np

env = gym.make_vec(
    id="CartPole-v1",
    num_envs=16,
    vectorization_mode="sync",
    vector_kwargs={ "autoreset_mode": gym.vector.AutoresetMode.DISABLED }
)

gym.pprint_registry()
print(env.observation_space.shape)
print(env.action_space.shape)
print(env.spec)