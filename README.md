# BAKE
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#license)

BAKE is a reinforcement learning framework written from scratch in Rust, using [Burn](https://burn.dev/).
It was created to study reinforcement learning algorithms and their
implementation details.

## Training Sanity Checks
The following experiments verify that the implemented algorithms learn reliably.
#### DQN variants on CartPole-v1
![Learning curves of DQN variants on native Rust CartPole-v1](docs/dqn_plots.png)
Summary
| algorithm    | steps to 475   | solved seeds   |   final return |   maximum q mean |
|:-------------|:---------------|:---------------|---------------:|-----------------:|
| DQN          | 270k           | 5 / 5          |          493.8 |            244.9 |
| Double DQN   | 230k           | 5 / 5          |          489.8 |            102.2 |
| DQN with PER | 215k           | 5 / 5          |          497.3 |            103.9 |
| Dueling DQN  | 215k           | 5 / 5          |          500   |            105.8 |
| NoisyNet-DQN | 330k           | 5 / 5          |          490.7 |            106.3 |

Hyperparameters
|$\gamma$|$\varepsilon$|loss function|optimizer|lr|warmup|update frequency|sync frequency|batch size|buffer capacity|$\alpha$ (for PER)|$\beta$ (for PER)|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|0.99|1.0 -> 0.05, linearly, from step 0 to 250000|MSE|Adam|2.5e-4|10000|10|500|128|10000|0.3|0.4 -> 1.0|

#### PPO on LunarLander-v3
![Learning curve of PPO on Gymnasium LunarLander-v3](docs/ppo_plots.png)

Hyperparameters
|$\gamma$|$\lambda$|clip $\epsilon$|loss function|entropy coefficient|rollout size|minibatch size|epoch|optimizer|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|0.99|0.95|0.1|Huber loss with $\delta$=10.0|0.05 -> 0.008 linearly, from step 0 to 1250000|2048|256|10|Adam|

For more information, see [PPO configuration file](crates/bake/configs/gymenv/ppo_lunarlander.json) and [PPO code](crates/bake/examples/gymenv/gym_lunarlander_ppo.rs)

#### Reproduction
##### DQN Variants
Training Examples
- [DQN](crates/bake/examples/deep/dqn.rs)
- [Double DQN](crates/bake/examples/deep/ddqn.rs)
- [DQN with PER](crates/bake/examples/deep/dqn_per.rs)
- [Dueling DQN](crates/bake/examples/deep/dueling_dqn.rs)
- [NoisyNet-DQN](crates/bake/examples/deep/noisy_dqn.rs)

Plotting Script: [script](docs/dqn-algs-compare.py)
##### PPO
Configuration: [configuration](crates/bake/configs/gymenv/ppo_lunarlander.json)  
Training Example: [example](crates/bake/examples/gymenv/gym_lunarlander_ppo.rs)  
Plotting Script: [script](docs/lunarlander_ppo.py)

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
|Priortized Experience Replay|with `PrioritizedExperienceReplayBuffer`|
|NoisyNet|with `NoisyLinear`|

</details>

### Environments
- Native `CartPole` modeled after [Gymnasium](https://gymnasium.farama.org)'s CartPole-v1
- Gymnasium Environment Bindings with PyO3. More environments will be supported later.
    - `GymnasiumEnv<CartPoleInfo>`: CartPole-v1
    - `GymnasiumEnv<MountainCarInfo>`: MountainCar-v0
    - `GymnasiumEnv<AcrobotInfo>`: Acrobot-v1
    - `GymnasiumEnv<LunarLanderInfo>`: LunarLander-v3

### Exploration Strategies
- $\varepsilon$-greedy
- Boltzmann
- NoisyNet

## Quick Start
Bake gives you all the components to run the training loops. What you only have to do is to implement the network structure and training loop.
```rust
use bake::deep::prelude::*;
use bake::deep::algorithm::Dqn;
use bake::deep::env::{GymnasiumEnv, LunarLanderInfo};
use bake::deep::approximator::wrapper::{ConstrainedQNet};
use bake::deep:scheduler::LinerScheduler;
use burn::prelude::*;
// your custom network structure
#[derive(Module, Debug)]
pub struct MyQNet { /* ... */ }

impl MyQNet { /* ... */ }

impl QNet for MyQNet {
    type Obs = /* observation type which your QNet can take */

    fn forward(&self, obs: Self::Obs) -> Tensor<2> {
        /* your forward logic */
    }
}

pub fn main() {
    let env = GymnasiumEnv::<LunarLanderInfo>::new(/* seed */);
    // your custom network structure is wrapped with libraries wrapper
    let online = ConstrainedQNet::new(MyQNet::new( /* ... */ ));
    let target = online.clone();
    // Experience replay buffer for DQN
    let buffer = ReplayBuffer::new(/* seed */, /* capacity */);
    // A helper for training loop. Creates transition for you.
    let tape = Tape::new(&mut env);
    // behavior policy for DQN
    let mut exploration = EpsGreedy::new(/* seed */, 1.0f32);
    // configuration of the algorithm
    let config = Dqn::new(0.99, ValueLoss::MseLoss);
    // hyperparameter schedular
    let eps_sch = LinearScheduler::new(/* start value*/, /* end value */, /* total steps */);
    // logger for updates
    let logger = Logger::new();

    // training loop
    for count in 0..500000 {
        let action = exploration.sample(&online, tape.obs.clone(), tape.constraint.clone());
        let transition = tape.step(&mut env, action);
        buffer.push(transition);

        if /* Warmup, Update frequency */ && let Some(batch) = buffer.sample(batch_size) {
            // get the loss
            let loss = Dqn::loss(&config, &online, &target, batch);
            // record the loss information
            logger.record(&loss);
            // update the value
            online = Dqn::update(online, loss, lr, &mut opt);
        }

        if /* Sync online and target network */ {
            let record = online.clone().into_record();
            target = target.load_record(record);
        }

        if tape.done() {
            // reset the environment and tape
            tape.reset(&mut env);
        }

        if count % /* logging frequency */ {
            // get the total mean of recorded logs
            let mean = logger.mean();
            // get the log and print it
            let td_error = mean.get("td_error").unwrap_or(&0f32);
            print("td error: {td_error}");
            // reset the logger
            logger.clear();
        }

        *exploration.eps_mut() = eps_sch.step();
    }
}
```
For tabular algorithms, you only have to implement your own training loop.

```rust
use bake::tabular::agent::*;
use bake::tabular::env::*;
use bake_tabular::policy::EpsGreedy;
use bake_tabular::types::Tape;

fn main() {
    /* grid world environment */
    let mut env = GridWorld::new();
    /* Q-Learning algorithm */
    let mut agent = QLearningAgent::new(env.n_states(), env.n_actions(), 0.3, 0.99);
    /* behavior policy */
    let mut policy = EpsGreedy::new(2,1f32);
    let mut tape = Tape::new(&mut env);

    for i in 0..=100000 {
        let mut episode_reward = 0f32;
        let mut n_steps = 0usize;
        tape.reset(&mut env);
        loop {
            let action = agent.action(&mut policy, tape.obs, tape.mask);
            let t = tape.step(&mut env, action);
            agent.update(t.clone());
            
            episode_reward += t.reward;
            n_steps += 1;
            if t.terminated || t.truncated { break; }
        }
        *policy.eps_mut() *= 0.9996;
        if i % 10000 == 0 { println!("Episode {i}, Steps: {n_steps}, Reward: {}, Eps: {}", episode_reward, policy.eps()) }
    }
}
```

## Examples
```bash
git clone https://github.com/joonjoon0425/bake.git
cd bake
cargo run --release --example qlearning
cargo run --release --example ppo
uv sync
# activate your virtual environment
source .venv/bin/activate
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

## License

Licensed under the [MIT License](LICENSE).