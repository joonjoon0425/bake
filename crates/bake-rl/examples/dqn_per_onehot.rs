use bake_rl::env::space_transform::one_hot::BatchedOneHotTransform;
use bake_rl::env::wrapper::{BatchedObservationTransform, BatchedRecordEpisodeStatistics, SynchronizedBatchedEnvironment};
use bake_rl::deep::prelude::*;
use bake_rl::core::scheduler::LinearScheduler;
use bake_rl::core::scheduler::Scheduler;
use bake_rl::env::classic::masked_cliff_walking::MaskedCliffWalkingConfig;

use burn::nn::activation::ActivationConfig;
use burn::prelude::*;
use burn::nn::*;
use burn::nn::activation::Activation;
use burn::optim::*;

pub fn main() {
    let seed = 140;
    let device = Device::default();
    device.seed(seed);

    let n_envs = 3;
    
    let env = {
        let mut vec = vec![];
        for _ in 0..n_envs as u64 { vec.push(MaskedCliffWalkingConfig.init()); }
        vec
    };
    let env = SynchronizedBatchedEnvironment::new(env, &device);
    let env = BatchedObservationTransform::new(env, BatchedOneHotTransform::new());
    let env = BatchedRecordEpisodeStatistics::new(env);
    
    let mut tape = BatchedTape::new(env, true);

    let state = DQNState { gamma: 0.99, loss_fn: LossFn::MseLoss };
    let mut online = TestNet::new(vec![tape.obs_space().shape()[1], 64, 64, tape.action_space().n()], ActivationConfig::Relu, &device);
    let mut target = online.clone();
    let lr = 1e-3;
    let mut opt = AdamConfig::new().init();
        
    let mut exploration = BatchedEpsilonGreedy::new(1.0f32);
    let mut buffer = ReplayBuffer::with_prioritized_sampler(0.6, 0.4, Some(1.0), 100000, n_envs, &device);

    let total_steps = 500000;
    let warmup = 10000;
    let update_freq = 10;
    let sync_freq = 500;
    let batch_size = 128;

    let mut eps_sch = LinearScheduler::new(1.0, 0.05, total_steps, 0.25);
    let mut beta_sch = LinearScheduler::new(0.4, 1.0, total_steps, 1.0);

    for count in 0..total_steps {
        let action = online.action(&mut exploration, tape.obs.clone(), tape.constraint.clone());
        let (batch, info) = tape.step(action);

        buffer.push(batch);

        if count >= warmup && count % update_freq == 0 && let Some(sample) = buffer.sample(batch_size) {
            let (net, loss) = dqn_loss(&state, online, &target, sample.sample, sample.info);
            buffer.update_priorities(sample.indices, loss.td_error.clone());
            online = net.update(loss, lr, &mut opt);
        }

        if count % sync_freq == 0 {
            target = online.clone();
        }

        if count % 1 == 0 {
            let (_, stats) = info;
            // just show naively
            for stat in stats {
                println!("count:{}, environment id: {}, reward: {}, step: {}", count, stat.id, stat.r, stat.s);
            }
        }

        *exploration.epsilon_mut() = eps_sch.step() as f32;
        *buffer.beta_mut() = beta_sch.step() as f32;
    }

    // // evaluation
    // // training in native cartpole, evaluation in gymnasium environment... it's like sim2sim
    // let env = GymnasiumEnvironmentConfig::cart_pole().kwarg("render_mode", "human").seed(10).init();
    // let env = SynchronizedBatchedEnvironment::new(vec![env], &device);

    // let mut tape = BatchedTape::new(env, false);
    // let mut eval = BatchedGreedy;
    // for _ in 0..5 {
    //     loop {
    //         let action = online.action(&mut eval, tape.obs.clone(), tape.constraint.clone());
    //         tape.step(action);
    //         if tape.done().clone().into_scalar() {
    //             tape.reset(None);
    //             break;
    //         }
    //     }
    // }
}

#[derive(Module, Debug)]
pub struct TestNet {
    linear: Vec<Linear>,
    activation: Activation
}

#[qnet]
impl TestNet {
    pub fn new(dims: Vec<usize>, activation: ActivationConfig, device: &Device) -> Self {
        let n = dims.len();
        let mut lv = Vec::with_capacity(n);
        for i in 0..(n - 1) {
            lv.push(LinearConfig::new(dims[i], dims[i + 1]).init(device));
        }
        Self {
            linear: lv,
            activation: activation.init(device),
        }
    }

    pub fn forward(&self, mut obs: Tensor<2>) -> Tensor<2> {
        let n = self.linear.len();
        for i in 0..(n - 1) {
            obs = self.activation.forward(self.linear[i].forward(obs));
        }
        obs = self.linear[n - 1].forward(obs);
        obs
        // self.linear[n - 1].forward(obs)
    }
}