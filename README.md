# BAKE
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#license)

Reinforcement learning framework in rust, built from scratch on [Burn](https://burn.dev/).
Bake was built to study reinforcement learning algorithms.

## Results
TODO: Add Experiments Results
## Tabular
### Algorithms
|Algorithm|Implementation|
|:---:|:---:|
|Q-Learning|[qlearning.rs](crates/bake-tabular/src/agent/qlearning.rs)|
|Sarsa|[sarsa.rs](crates/bake-tabular/src/agent/sarsa.rs)|
|Expected Sarsa|[expected_sarsa.rs](crates/bake-tabular/src/agent/expected_sarsa.rs)|
|n-step Sarsa|[nstepsarsa.rs](crates/bake-tabular/src/agent/nstepsarsa.rs)|
|n-step Q-Learning with tree-backup method|[nstepqlearning.rs](crates/bake-tabular/src/agent/nstepqlearning.rs)|
### Environments
- `GridWorld`
- `Blackjack`
- `MaskedGridWorld`
### Exploration Strategies
- $\varepsilon$-greedy

## Deep
### Algorithms
|Algorithm|Implementation|
|:---:|:---:|
|DQN|[dqn.rs](crates/bake-deep/src/algorithm/dqn.rs)|
|Double DQN|[double_dqn.rs](crates/bake-deep/src/algorithm/double_dqn.rs)|
|REINFORCE|[vpg.rs](crates/bake-deep/src/algorithm/vpg.rs)|
|A2C|[a2c.rs](crates/bake-deep/src/algorithm/a2c.rs)|
|PPO|[ppo.rs](crates/bake-deep/src/algorithm/ppo.rs)|

<details>
<summary>Algorithm Extensions</summary>

**DQN and Double DQN extensions**

|Extension|Implementation|
|:---:|:---:|
|Dueling|with `DuelingQNet`|
|Priortized Experience Replay|with `PriortizedExperienceReplayBuffer`|
|NoisyNet|with `NoisyLinear`|

</details>

### Environments
- Native `CartPole` referenced from [Gymnasium](https://gymnasium.farama.org)
- Gymnasium Environment Bindings with PyO3. More environments will be binded later.
    - `GymnasiumEnv<CartPoleInfo>`: CartPole-v1
    - `GymnasiumEnv<MountainCarInfo>`: MountainCar-v0
    - `GymnasiumEnv<AcrobotInfo>`: Acrobot-v1
    - `GymnasiumEnv<LunarLanderInfo>`: LunarLander-v3

### Exploration Strategies
- $\varepsilon$-greedy
- Boltzmann
- NoisyNet

## Examples
```bash
git clone https://github.com/joonjoon0425/bake.git
cd bake
cargo run --release --example qlearning
cargo run --release --example ppo
cargo run --release --example gym_lunarlander_ppo
```
<details>
<summary>Total example lists</summary>


#### Tabular
|Example Name|Environment|
|:---:|:---:|
|qlearning|`GridWorld`|
|sarsa|`MaskedGridWorld`|
|expected_sarsa|`MaskedGridWorld`|
|nstepsarsa|`Blackjack`|
|nstepqlearning|`GridWorld`|

#### Deep: Gymnasium Environments
|Example Name|Environment|Explanation|
|:---:|:---:|:---:|
|gym_acrobot_a2c|`GymnasiumEnv<AcrobotInfo>`|A2C on Acrobot-v1|
|gym_cartpole_dqn|`GymnasiumEnv<CartPoleInfo>`|DQN on CartPole-v1|
|gym_lunarlander_dqn_per|`GymnasiumEnv<LunarLanderInfo>`|DQN on LunarLander-v3 with PER|
|gym_lunarlander_ppo|`GymnasiumEnv<LunarLanderInfo>`|PPO on LunarLander-v3|
|gym_mountaincar_dqn_per|`GymnasiumEnv<MountainCarInfo>`|DQN on MountainCar-v0 with PER|

#### Deep: Native Environments (Currently only CartPole)
|Example Name|Environment|Explanation|
|:---:|:---:|:---:|
|dqn|`CartPole`|Classic DQN with CartPole|
|ddqn|`CartPole`|Double DQN|
|dueling_dqn|`CartPole`|Dueling DQN|
|dqn_per|`CartPole`|DQN with PER|
|noisy_dqn|`CartPole`|Noisy-DQN|
|dueling_ddqn_per|`CartPole`|Dueling Double DQN with PER|
|vpg|`CartPole`|REINFORCE|
|a2c|`CartPole`|A2C|
|noisy_a2c|`CartPole`|Noisy-A2C|
|equivariant_a2c|`CartPole`|A2C with Z2-symmetric network|
|ppo|`CartPole`|PPO|
|equivariant_ppo|`CartPole`|PPO with Z2-symmetric network|
</details>

Example codes can be found [here](crates/bake/examples/).

<details>
<summary>Gymnasium environments setting</summary>

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
</details>

## License

Licensed under the [MIT License](LICENSE).