# BAKE
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#license)

Bake is a reinforcement learning framework written from scratch in Rust, using [Burn](https://burn.dev/). Created to study reinforcement learning algorithms and their implementation details.

## Training Sanity Checks
Following experiments verifies that the implemented algorithms learn reliably.
### DQN with PER and NoisyNet-DQN with PER on native Rust CartPole-v1
![Learning curves of DQN and NoisyNet-DQN on native Rust CartPole-v1](docs/figures/cartpole.png)
Codes:
- [DQN with PER](crates/bake-deep/examples/dqn_test.rs)
- [NoisyNet-DQN with PER](crates/bake-deep/examples/noisy_dqn_test.rs)
- [Plot](docs/scripts/cartpole.py)
### PPO and Dueling DQN on Gymnasium LunarLander-v3
![Learning curves of PPO and Dueling DQN with PER on Gymnasium LunarLander-v3](docs/figures/lunarlander.png)
Codes:
- [PPO](crates/bake-gym/examples/ppo_lunarlander.rs)
- [Dueling DQN with PER](crates/bake-gym/examples/dueling_dqn_lunarlander.rs)
- [Plot](docs/scripts/lunarlander.py)

## Features
### Tabular RL
#### Algorithms
|Algorithm|Implementation|
|:---:|:---:|
|Q-Learning|[qlearning.rs](crates/bake-tabular/src/algorithm/qlearning.rs)|
|Sarsa|[sarsa.rs](crates/bake-tabular/src/algorithm/sarsa.rs)|
|n-step Q-Learning|[nstep_qlearning.rs](crates/bake-tabular/src/algorithm/nstep_qlearning.rs)|
|n-step Sarsa|[nstep_sarsa.rs](crates/bake-tabular/src/algorithm/nstep_sarsa.rs)|

For the n-step Q-learning, user can choose one of two methods, tree-backup method and importance sampling method.

#### Environments
|Environment|Explanation|Implementation|
|:---:|:---:|:---:|
|`CliffWalking`|A classic S & B grid world|[cliff_walking.rs](crates/bake-tabular/src/env/cliff_walking.rs)|
|`MaskedCliffWalking`|A grid world with mask on the boundaries|[masked_cliff_walking](crates/bake-tabular/src/env/masked_cliff_walking.rs)|

#### Exploration Strategies
- greedy
- epsilon greedy
- Boltzmann

### Deep RL
#### Algorithms
|Algorithm|Implementation|
|:---:|:---:|
|DQN|[dqn.rs](crates/bake-deep/src/algorithm/dqn.rs)|
|Double DQN|[double_dqn.rs](crates/bake-deep/src/algorithm/double_dqn.rs)|
|REINFORCE|[reinforce.rs](crates/bake-deep/src/algorithm/reinforce.rs)|
|A2C|[a2c.rs](crates/bake-deep/src/algorithm/a2c.rs)|
|PPO|[ppo.rs](crates/bake-deep/src/algorithm/ppo.rs)|

<details>
<summary>Algorithm variants</summary>

**NoisyNet variant**
with `NoisyLinear`: [noisylinear.rs](crates/bake-deep/src/net/layer/noisy_linear.rs)

**DQN and Double DQN variants**
|Variant|Implementation|
|:---:|:---:|
|Dueling|with `DiscreteDuelingQNet`|
|Prioritized Experience Replay|with `ReplayBufferConfig::prioritized()`|
</details>

#### Environments
Native Rust:

These are pure Rust environments.
|Environment|Explanation|Implementation|
|:---:|:---:|:---:|
|`CartPole`|Modeled after [Gymnasium](https://gymnasium.farama.org)'s CartPole-v1|[cartpole.rs](crates/bake-deep/src/env/cartpole.rs)|
|`CliffWalking`|The one-hot encoded version of tabular environment `CliffWalking`|[cliffwalking.rs](crates/bake-deep/src/env/cliffwalking.rs)|
|`MaskedCliffWalking`|The one-hot encoded version of tabular environment `MaskedCliffWalking`|[masked_cliffwalking.rs](crates/bake-deep/src/env/masked_cliffwalking.rs)|

Gymnasium Binding:

These are environments which runs python interpreters internally. Binded with PyO3.
|Environment|Explanation|Implementation|
|:---:|:---:|:---:|
|`GymCartPole`|CartPole-v1 of Gymnasium|[cartpole.rs](crates/bake-gym/src/env/cartpole.rs)|
|`GymMountainCar`|MountainCar-v0 of Gymnasium|[mountaincar.rs](crates/bake-gym/src/env/mountaincar.rs)|
|`GymAcrobot`|Acrobot-v1 of Gymnasium|[acrobot.rs](crates/bake-gym/src/env/acrobot.rs)|
|`GymLunarLander`|LunarLander-v3 of Gymnasium|[lunarlander.rs](crates/bake-gym/src/env/lunarlander.rs)|
|`GymCliffWalking`|one-hot encoded CliffWalking-v1 of Gymnasium|[cliffwalking.rs](crates/bake-gym/src/env/cliffwalking.rs)|
|`GymTaxi`|one-hot encoded Taxi-v4 of Gymnasium|[taxi.rs](crates/bake-gym/src/env/taxi.rs)|
|`GymFrozenLake`|one-hot encoded FrozenLake-v1 of Gymnasium|[frozenlake.rs](crates/bake-gym/src/env/frozenlake.rs)|

#### Exploration Strategies
- greedy
- epsilon greedy
- Boltzmann
- NoisyNet

## Quick Start
Bake gives you all the components for making training loops. The user only have to implement the training loop and one's own network structure. (A toml configuration will be implemented later... [issue #32](https://github.com/joonjoon0425/bake/issues/32))
##### Warining
When creating a network, the user must create it on the autodiff device.

```rust
use bake_common::scheduler::{LinearScheduler, Scheduler};
use bake_deep::buffer::replay::ReplayBufferConfig;
use bake_deep::explore::{EpsGreedy, Exploration};
use bake_common::logger::MovingAvgLogger;
use bake_deep::net::basic::MlpDiscreteQNet;
use bake_deep::wrapper::DiscreteQNetWrapper;
use bake_deep::env::{CartPole, Tape};
use bake_deep::algorithm::Dqn;
use bake_deep::loss::Loss;

use burn::optim::AdamConfig;
use burn::prelude::*;

#[derive(Module, Debug)]
struct MyOwnQNet {
    /*  */,
}

impl MyOwnQNet {
    pub fn new(/* */) -> Self {
        /* */
    }
}

impl DiscreteQNet for MyOwnQNet {
    type Obs = /*  */;
    
    fn forward(&self, obs: Self::Obs) -> Tensor<2> {
        /* */
    }
}

pub fn main() {
    println!("count,reward_avg,step_avg,loss,td_error,qmean,eps");
    let seed: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    // device settings
    let device = Device::default();
    device.seed(seed);
    let autodiff_device = device.clone().autodiff();
    // environment setting
    let env = CartPole::new(seed, &device);
    // algorithm state setting
    let state = Dqn{ gamma: 0.99, loss_fn: Loss::MseLoss };
    // create yout own network on autodiff device
    // wrap it with DiscreteQNetWrapper
    let mut online = DiscreteQNetWrapper::new(MyOwnQNet::new(/* */, &autodiff_device));
    let mut target = online.clone();
    let lr = 2.5e-4;
    let mut opt = AdamConfig::new().init();

    // exploration strategy
    let mut exploration = EpsGreedy::new(seed, 1.0f32);
    // Prioritized Experience Replay with alpha = 0.6 and bets = 0.4
    let mut buffer = ReplayBufferConfig::prioritized(seed, 50000, 0.6, 0.4).with_priority_clip(1.0).init();
    // This is the helper struct which creates a Transition for the user
    let mut tape = Tape::new(env);
    // Moving average logger
    let mut logger = MovingAvgLogger::new();

    let total_steps = 500000;
    let warmup = 10000;
    let update_freq = 10;
    let sync_freq = 500;
    let batch_size = 128;

    let window = 100;
    logger.register("loss", 500);
    logger.register("mean_td_error", 500);
    logger.register("qmean", 500);
    logger.register("reward", window);
    logger.register("step", window);

    // scheduler for schedulable values
    let mut eps_sch = LinearScheduler::new(1.0, 0.05, total_steps, 0.25);
    let mut beta_sch = LinearScheduler::new(0.4, 1.0, total_steps, 1.0);

    for count in 0..=total_steps {
        let action = exploration.sample(&online, tape.obs.clone(), tape.constraint.clone());
        let t = tape.step(action);
        buffer.push(t);

        if count >= warmup && count % update_freq == 0 && let Some((batch, batch_info)) = buffer.sample(batch_size) {
            // get the dqn objective
            // the given network must be passed to the update function
            let (net, loss) = Dqn::loss(&state, online, &target, batch, batch_info.clone());
            // record the loss informations into logger
            logger.push(&loss);
            // update the priority of prioritized replay buffer
            buffer.update_priority(&batch_info.indices, loss.td_error.clone());
            // update the network
            online = Dqn::update(net, loss, lr, &mut opt);
        }

        if count % sync_freq == 0 {
            let record = online.clone().into_record();
            target = target.load_record(record);
        }

        if tape.done() {
            logger.push_single("reward", tape.episode_reward);
            logger.push_single("step", tape.steps as f32);
            tape.reset();
        }

        if count % 5000 == 0 {
            let reward = logger.emit("reward");
            let step = logger.emit("step");
            let loss = logger.emit("loss");
            let mean_td_error = logger.emit("mean_td_error");
            let qmean = logger.emit("qmean");
            println!("{count},{reward},{step},{loss},{mean_td_error},{qmean},{}", exploration.eps());
        }

        *exploration.eps_mut() = eps_sch.step() as f32;
        *buffer.beta_mut() = beta_sch.step();
    }   
}

```

## Examples
To use Gymnasium environments, python virtual environment is required. This project uses [uv](https://docs.astral.sh/uv/).
```bash
git clone https://github.com/joonjoon0425/bake.git
cd bake
cargo run --release --example qlearning_test
cargo run --release --example ppo_test
uv sync
source .venv/bin/activate
cargo run --release --example ppo_lunarlander
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
|tabular_test|`MaskedCliffWalking`|[code](crates/bake-deep/examples/tabular_test.rs)|

#### Deep RL: Gymnasium Environments
|Example name|Environment|Code|
|:---:|:---:|:---:|
|dqn_taxi|`GymTaxi`|[code](crates/bake-gym/examples/dqn_taxi.rs)|
|dqn_mountaincar|`GymMountainCar`|[code](crates/bake-gym/examples/dqn_mountaincar.rs)|
|dqn_frozenlake|`GymFrozenLake`|[code](crates/bake-gym/examples/dqn_frozenlake.rs)|
|a2c_cartpole|`GymCartPole`|[code](crates/bake-gym/examples/a2c_cartpole.rs)|
|dueling_dqn_lunarlander|`GymLunarLander`|[code](crates/bake-gym/examples/dueling_dqn_lunarlander.rs)|
|ppo_lunarlander|`GymLunarLander`|[code](crates/bake-gym/examples/ppo_lunarlander.rs)|

</details>

## License
Licensed under the [MIT License](LICENSE).