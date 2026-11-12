import gymnasium as gym
import numpy as np

import ale_py

# env = gym.make_vec(
#     id="CartPole-v1",
#     num_envs=16,
#     vectorization_mode="sync",
#     vector_kwargs={ "autoreset_mode": gym.vector.AutoresetMode.DISABLED }
# )

env = gym.wrappers.FrameStackObservation(
    gym.wrappers.AtariPreprocessing(
        gym.make(
            id="ALE/Adventure-v5",
            frameskip=1,
        ),
        frame_skip=4,
    ),
    stack_size=4,
)

gym.pprint_registry()
print(env.observation_space.shape)
print(env.action_space.shape)
print(env.spec)