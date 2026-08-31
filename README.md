# BAKE: A reinforcement learning framework built from scratch in rust, on top of [burn](https://burn.dev/)
Learn RL with rust's burn, by building a framework. This repository was created for studying reinforcement learning: tabular, and deep rl.

## Algorithms
### Tabular
- Q-Learning, Sarsa, Expected Sarsa
- n-step Sarsa, n-step Q-Learning with tree backup method
### Deep RL
- DQN, Double DQN, + dueling versions + NoisyNet versions
- Vanilla Policy Gradient (REINFORCE)
- Advantage Actor-Critic (A2C) + NoisyNet version
- Proximal Policy Optimization (PPO)

## Environment
### Tabular
- Grid world
- Blackjack
- Masked Grid world
### Deep RL
- Native CartPole-v1 referenced from [Gymnasium](https://gymnasium.farama.org)
- Gymnasium Environment Bindings with PyO3 (CartPole-v1, MountainCar-v0, Acrobot-v1, LunarLander-v3). More environments will be binded later.

## Features
- Supports json configuration (only for deep rl now)

## Examples
There are five examples in tabular algorithms, tweleve examples in deep rl algorithms, and five examples with gymnasium environments. They can be run with
```bash
git clone https://github.com/joonjoon0425/bake.git
cd bake
cargo run --release --example <example-name>
```
Followings are possible example names:
- tabular:
    - expected_sarsa
    - nstepqlearning
    - nstepsarsa
    - qlearning
    - sarsa
- deep rl:
    - a2c
    - ddqn
    - dqn
    - dueling_dqn
    - equivariant_a2c
    - equivariant_ppo
    - noisy_dqn
    - noisy_a2c
    - ppo
    - vpg
    - dqn_per
    - dueling_ddqn_per
- gymenv
    - gym_cartpole_dqn
    - gym_mountaincar_dqn_per
    - gym_lunarlander_ppo
    - gym_lunarlander_dqn
    - gym_acrobot_a2c

Example codes can be found in `crate/bake/examples`.

For gymnasium environments, follow the instructions below:
```bash
git clone https://github.com/joonjoon0425/bake.git
cd bake
uv venv --python <3.12 or 3.13>
uv sync
```

- Fix the .cargo/config.toml so that PyO3 can see the python 3.12 or 3.13 library
```toml
[env]
PYO3_PYTHON = { value = ".venv/bin/python", relative = true, force = true }
[build]
rustflags = [
  "-C", "link-arg=-Wl,-rpath,<path-to-cpython3.12-or-3.13-lib>",
]
```
- now we can run the example
```bash
source .venv/bin/activate
cargo run --release --example <gymenv-example-name>
```

- This is required since the Box2D of Gymnasium does not support Python 3.14 and PyO3 uses system's python. For more information, see https://github.com/astral-sh/uv/issues/11006.