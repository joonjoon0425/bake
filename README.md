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
- CartPole-v1 referenced from [Gymnasium](https://gymnasium.farama.org)

## Features
- Supports json configuration (only for deep rl now)

## Examples
There are five examples in tabular algorithms, and tweleve examples in deep rl algorithms. They can be run with
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

Example codes can be found in `crate/bake/examples`.
