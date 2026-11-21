use bake_rl::env::wrapper::{BatchedRecordEpisodeStatistics, SynchronizedBatchedEnvironment};
use bake_rl::core::scheduler::LinearScheduler;
use bake_rl::core::scheduler::Scheduler;
use bake_rl::env::gym::GymnasiumEnvironmentConfig;
use bake_rl::deep::prelude::*;
use bake_rl::logger::{LogCollector, StdoutSink};

use std::time::Instant;

use burn::nn::activation::ActivationConfig;
use burn::prelude::*;
use burn::nn::*;
use burn::nn::activation::Activation;
use burn::optim::*;

pub fn main() {
    let s = Instant::now();

    let seed = 140;
    let device = Device::default();
    device.seed(seed);

    let mut collector = LogCollector::new();
    let mut stdout = StdoutSink::new("Dueling Double DQN - CartPole Log");

    let env = GymnasiumEnvironmentConfig::cart_pole().seed(seed).init();
    let env = SynchronizedBatchedEnvironment::new(vec![env], &device);
    let env = BatchedRecordEpisodeStatistics::new(env, 100);

    let state = DoubleDQNState { gamma: 0.99, loss_fn: LossFn::MseLoss };
    let mut online = DuelingTestNet::new(vec![env.obs_space().shape()[1], 64, 64, env.action_space().n()], ActivationConfig::Relu, &device);
    let mut target = online.clone();
    let lr = 1e-3;
    let mut opt = AdamConfig::new().init();

    let mut tape = BatchedTape::new(env, true);
        
    let mut exploration = BatchedEpsilonGreedy::new(1.0f32);
    let mut buffer = ReplayBuffer::with_uniform_sampler(100000, 1, &device);

    let total_steps = 300000;
    let warmup = 10000;
    let update_freq = 10;
    let sync_freq = 500;
    let batch_size = 128;

    let mut eps_sch = LinearScheduler::new(1.0, 0.05, total_steps, 2.0 / 3.0);

    for count in 0..total_steps {
        let action = online.action(&mut exploration, tape.obs.clone(), tape.constraint.clone());
        let (batch, _) = tape.step(action);

        buffer.push(batch);

        if count >= warmup && count % update_freq == 0 && let Some(sample) = buffer.sample(batch_size) {
            let (net, loss) = double_dqn_loss(&state, online, &target, sample.sample, sample.info);
            online = net.update(loss.clone(), lr, &mut opt);
            // log
            collector.record(2, "loss", loss.loss);
            collector.record(2, "q_mean", loss.q_mean);
            collector.record(2, "td_error_abs_mean", loss.td_error.abs().mean());
        }

        if count % sync_freq == 0 {
            target = online.clone();
        }

        if count % 5000 == 0 {
            // log
            collector.record(1, "ep/reward_mean", tape.env().reward_mean());
            collector.record(1, "ep/length_mean", tape.env().length_mean());
            collector.record(1, "ep/time_mean", tape.env().time_mean());
            collector.record(5, "elapsed time", s.elapsed().as_secs_f64());
            collector.dump_to(count, &mut stdout).expect("unexpected error");    
        }

        *exploration.epsilon_mut() = eps_sch.step() as f32;
    }

    // evaluation
    let env = GymnasiumEnvironmentConfig::cart_pole().kwarg("render_mode", "human").seed(10).init();
    let env = SynchronizedBatchedEnvironment::new(vec![env], &device);

    let mut tape = BatchedTape::new(env, false);
    let mut eval = BatchedGreedy;
    for _ in 0..5 {
        loop {
            let action = online.action(&mut eval, tape.obs.clone(), tape.constraint.clone());
            tape.step(action);
            if tape.done().clone().into_scalar() {
                tape.reset(None);
                break;
            }
        }
    }
}

#[derive(Module, Debug)]
pub struct DuelingTestNet {
    linear: Vec<Linear>,
    adv: Linear,
    val: Linear,
    activation: Activation
}

#[qnet(dueling)]
impl DuelingTestNet {
    pub fn new(dims: Vec<usize>, activation: ActivationConfig, device: &Device) -> Self {
        let n = dims.len();
        let mut lv = Vec::with_capacity(n);
        for i in 0..(n - 2) {
            lv.push(LinearConfig::new(dims[i], dims[i + 1]).init(device));
        }
        let adv = LinearConfig::new(dims[n - 2], dims[n - 1]).init(device); 
        let val = LinearConfig::new(dims[n - 2], 1).init(device);
        Self {
            linear: lv,
            adv,
            val,
            activation: activation.init(device),
        }
    }

    pub fn forward(&self, mut obs: Tensor<2>) -> (Tensor<1>, Tensor<2>) {
        let n = self.linear.len();
        for i in 0..n {
            obs = self.activation.forward(self.linear[i].forward(obs));
        }
        let val = self.val.forward(obs.clone()).squeeze_dim(1);
        let adv = self.adv.forward(obs);
        (val, adv)
        // self.linear[n - 1].forward(obs)
    }
}