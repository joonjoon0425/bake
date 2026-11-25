<div align="center">

# `bake` <br> [![License](https://img.shields.io/badge/License-MIT-f15a24?style=flat-square)](#license)
</div>

**`bake`** is a *reinforcement learning framework* written from scratch in **Rust**, using [`burn`](https://burn.dev/). Created to study reinforcement learning algorithms and their implementation details.

## Training Sanity Checks

## Implemented Algorithms and Features
### Algorithms
#### Deep Reinforcement Learning Algorithms
|algorithm|code|
|:---:|:---:|
|Deep Q-Network|[dqn.rs](crates/bake-rl/src/deep/algorithms/dqn.rs)|
|Double Deep Q-Network|[double_dqn.rs](crates/bake-rl/src/deep/algorithms/double_dqn.rs)|
|REINFORCE|[reinforce.rs](crates/bake-rl/src/deep/algorithms/reinforce.rs)|
|Advantage Actor Critic|[a2c.rs](crates/bake-rl/src/deep/algorithms/a2c.rs)|
|Proximal Policy Optimization|[ppo.rs](crates/bake-rl/src/deep/algorithms/ppo.rs)|

Continuous actions are not supported yet. Will be implemented soon.
#### Tabular Reinforcement Learning Algorithms
|algorithm|code|
|:---:|:---:|
|on-policy Monte Carlo|[monte_carlo/on_policy.rs](crates/bake-rl/src/tabular/algorithms/monte_carlo/on_policy.rs)|
|off-policy Monte Carlo|[monte_carlo/off_policy.rs](crates/bake-rl/src/tabular/algorithms/monte_carlo/off_policy.rs)|
|Q-Learning|[q_learning.rs](crates/bake-rl/src/tabular/algorithms/q_learning.rs)|
|SARSA|[sarsa.rs](crates/bake-rl/src/tabular/algorithms/sarsa.rs)|
|n-step SARSA|[n_step_sarsa.rs](crates/bake-rl/src/tabular/algorithms/n_step_sarsa.rs)|
|Q($`\sigma`$)|not implemented yet|

### Possible Algorithm Variants
|algorithm variants|how?|
|:---:|:---:|
|Dueling variants for DQN-like algorithms|via `#[qnet(dueling)]` attribute macro|
|Priority Experience Replay for DQN-like algorithms|via `ReplayBuffer::with_priority_sampler` function|
|NoisyNet|via [`NoisyLinear`](crates/bake-rl/src/deep/network/layer/noisy_linear.rs) layer|

### Exploration Strategies
|exploration strategies|code|
|:---:|:---:|
|Greedy|[greedy.rs](crates/bake-rl/src/exploration/greedy.rs)|
|Epsilon Greedy|[epsilon_greedy.rs](crates/bake-rl/src/exploration/epsilon_greedy.rs.rs)|
|Boltzmann|[boltzmann.rs](crates/bake-rl/src/exploration/boltzmann.rs)|

### Environments
#### Rust Native Environments
|environment|code|
|:---:|:---:|
|CartPole|[cartpole.rs](crates/bake-rl/src/env/classic/cartpole.rs)|
|CliffWalking|[cliff_walking.rs](crates/bake-rl/src/env/classic/cliff_walking.rs)|
|MaskedCliffWalking|[masked_cliff_walking.rs](crates/bake-rl/src/env/classic/masked_cliff_walking.rs)|

#### Gymnasium Binding with `pyo3`
- All `toy-text` environments
- All `box2d` environments
- All `classic-control` environments
- All `mujoco` environments
- All `ALE` environments, except `Backgammon` and `VideoCheckers`, with preprocessing (frame stack with 4 frames, and frame skip 4)

#### Vectorization of non-vectorized Environments
Possible via [`SynchronizedBatchedEnvironment`](crates/bake-rl/src/env/wrapper/sync_env.rs) wrapper. 

## Examples
For Gymnasium environment binding, python virtual environment is required. This project uses [uv](https://docs.astral.sh/uv/).
```bash
git clone https://github.com/joonjoon0425/bake.git
cd bake
cargo run --release --example a2c
cargo run --release --example ppo_test
uv sync
source .venv/bin/activate
cargo run --release --example ppo_vec_lunar_lander
```
<details>
<summary>Total example lists</summary>

#### Tabular RL
|Example name|Environment|Code|
|:---:|:---:|:---:|
|qlearning_test|`MaskedCliffWalking`|[code](crates/bake-tabular/examples/qlearning_test.rs)|
|sarsa_test|`CliffWalking`|[code](crates/bake-tabular/examples/sarsa_test.rs)|
|nstep_qlearning_test|`CliffWalking`|[code](crates/bake-tabular/examples/nstep_qlearning_test.rs)|
|nstep_sarsa_test|`MaskedCliffWalking`|[code](crates/bake-tabular/examples/nstep_sarsa_test.rs)|

#### Deep RL: Native Rust Environments
|Example name|Environment|Code|
|:---:|:---:|:---:|
|dqn_test|`CartPole`|[code](crates/bake-deep/examples/dqn_test.rs)|
|noisy_dqn_test|`CartPole`|[code](crates/bake-deep/examples/noisy_dqn_test.rs)|
|reinforce_test|`CartPole`|[code](crates/bake-deep/examples/reinforce_test.rs)|
|a2c_test|`CartPole`|[code](crates/bake-deep/examples/a2c_test.rs)|
|ppo_test|`CartPole`|[code](crates/bake-deep/examples/ppo_test.rs)|

#### Deep RL: Gymnasium Environments
|Example name|Environment|Code|
|:---:|:---:|:---:|
|dqn_taxi|`GymTaxi`|[code](crates/bake-gym/examples/dqn_taxi.rs)|
|dqn_mountaincar|`GymMountainCar`|[code](crates/bake-gym/examples/dqn_mountaincar.rs)|
|dqn_frozenlake|`GymFrozenLake`|[code](crates/bake-gym/examples/dqn_frozenlake.rs)|
|a2c_cartpole|`GymCartPole`|[code](crates/bake-gym/examples/a2c_cartpole.rs)|
|dueling_dqn_lunarlander|`GymLunarLander`|[code](crates/bake-gym/examples/dueling_dqn_lunarlander.rs)|
|ppo_lunarlander|`GymLunarLander`|[code](crates/bake-gym/examples/ppo_lunarlander.rs)|
|ppo_lunarlander_vec|`SynchronizedEnvironment<GymLunarLander>`|[code](crates/bake-gym/examples/ppo_lunarlander_vec.rs)|

</details>

## Quick Start

## License
Licensed under the [MIT License](LICENSE).