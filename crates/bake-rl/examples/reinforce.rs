use bake_rl::env::wrapper::BatchedRecordEpisodeStatistics;
use bake_rl::env::wrapper::SynchronizedBatchedEnvironment;
use bake_rl::core::scheduler::LinearScheduler;
use bake_rl::core::scheduler::Scheduler;
use bake_rl::deep::prelude::*;

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

    let mut state =
        REINFORCEState {
            gamma: 0.99,
            c_e: 0.02,
            baseline: REINFORCEBaseline::LeaveOneOut,
        };

    let n_envs = 4;
    let env = {
        let mut vec = vec![];
        for i in 0..n_envs as u64 { vec.push(CartPoleConfig::new(seed + i).init()); }
        vec
    };
    let env = SynchronizedBatchedEnvironment::new(env, &device);
    let env = BatchedRecordEpisodeStatistics::new(env);

    let mut policy = TestNet::new(vec![env.obs_space().shape()[1], 128, env.action_space().n()], ActivationConfig::Relu, &device);
    let lr = 1e-3;
    let mut opt = AdamConfig::new().init();
    
    let mut buffer = EpisodeBuffer::new();
    let mut tape = BatchedTape::new(env, true);

    let total_steps = 500000;
    let mut c_e_sch = LinearScheduler::new(state.c_e as f64, 0.002, total_steps, 1.0);

    // used to ensure all environments has finished at least once
    let mut done = Tensor::zeros([n_envs], &device);

    for count in 0..total_steps {
        let action = policy.action(tape.obs.clone(), tape.constraint.clone());
        let (t, info) = tape.step(action);

        done.inplace(|d| d.bool_or(t.terminated.clone().bool_or(t.truncated.clone())));
        buffer.push(t);
        
        if done.clone().all().into_scalar() {
            tape.reset(None);
            let batch = buffer.pop();
            done = Tensor::zeros([n_envs], &device);
            let (p, loss) = reinforce_loss(&state, policy, batch);
            policy = p.update(loss, lr, &mut opt);
        }
        
        if count % 1 == 0 {
            let (_, stats) = info;
            // just show naively
            for stat in stats {
                println!("count:{}, environment id: {}, reward: {}, step: {}", count, stat.id, stat.r, stat.s);
            }
        }

        state.c_e = c_e_sch.step() as f32;
    }

    let d = s.elapsed();

    println!("It took {}s", d.as_secs());

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