use bake_rl::env::wrapper::{BatchedRecordEpisodeStatistics, SynchronizedBatchedEnvironment};
use bake_rl::scheduler::{Scheduler, LinearScheduler};
use bake_rl::deep::prelude::*;
use bake_rl::logger::{LogCollector, StdoutSink};
use bake_rl::env::classic::cartpole::CartPoleConfig;

use burn::prelude::*;
use burn::nn::*;
use burn::nn::activation::{Activation, ActivationConfig};
use burn::optim::*;

use std::time::Instant;

pub fn main() {
    let s = Instant::now();

    let seed: u64 = 120;
    let device = Device::default();
    device.seed(seed);

    let mut collector = LogCollector::new();
    let mut stdout = StdoutSink::new("REINFORCE - CartPole Log");

    let mut state =
        REINFORCEState {
            gamma: 0.99,
            c_e: 0.02,
            baseline: REINFORCEBaseline::LeaveOneOut,
        };

    let n_envs = 16;
    let env = Vec::from_fn(n_envs, |i| CartPoleConfig::new(i as u64).init());
    let env = SynchronizedBatchedEnvironment::new(env, &device);
    let env = BatchedRecordEpisodeStatistics::new(env, 100);

    let mut policy = TestNet::new(vec![env.obs_space().shape()[1], 64, 64, env.action_space().n()], ActivationConfig::Relu, &device);
    let lr = 1e-3;
    let mut opt = AdamConfig::new().init();
    
    let mut buffer = EpisodeBuffer::new(n_envs, &device);
    let mut tape = BatchedTape::new(env, true);

    let total_steps = 200000;
    let c_e_sch = LinearScheduler::new(state.c_e, 0.002, total_steps, 1.0);

    for count in 0..total_steps {
        let action = policy.action(tape.obs.clone(), tape.constraint.clone());
        let (t, _) = tape.step(action);

        buffer.push(t);
        
        if buffer.is_ready() {
            tape.reset(None);
            let batch = buffer.pop();
            let (p, loss) = reinforce_loss(&state, policy, batch);
            policy = p.update(loss.clone(), lr, &mut opt);
            collector.record(2, "loss", loss.loss);
            collector.record(2, "entropy", loss.entropy);
        }
        
        if count % 5000 == 0 {
            // log
            collector.record(1, "ep/reward_mean", tape.env().reward_mean());
            collector.record(1, "ep/length_mean", tape.env().length_mean());
            collector.record(1, "ep/time_mean", tape.env().time_mean());
            collector.record(-1, "elapsed time", s.elapsed().as_secs_f64());
            collector.dump_to(count, &mut stdout).expect("unexpected error");
        }

        state.c_e = c_e_sch.value(count) as f32;
    }

    // let env = GymnasiumEnvironmentConfig::cart_pole().kwarg("render_mode", "human").init();
    // let env = SynchronizedVectorEnvironment::new(vec![env], &device);

    // let mut tape = VectorTape::new(env, false);
    // for _ in 0..5 {
    //     loop {
    //         let action = ac.dist(tape.obs.clone(), tape.constraint.clone()).mode();
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

#[policy(distribution = Categorical)]
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