use bake_rl::env::wrapper::BatchedRecordEpisodeStatistics;
use bake_rl::scheduler::{Scheduler, LinearScheduler};
use bake_rl::env::gym::BatchedGymnasiumEnvironmentConfig;
use bake_rl::deep::prelude::*;
use bake_rl::logger::{LogCollector, StdoutSink, CsvSink};
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
    let stdout = StdoutSink::new("PPO - LunarLander Log");
    let csv = CsvSink::new(std::path::Path::new("logs/ppo.csv"), &["ep/reward_mean", "ep/length_mean", "ep/time_mean", "loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "elapsed time"]).unwrap();
    
    let mut sink = (stdout, csv);

    let mut state =
        PPOState {
            gamma: 0.99,
            c_e: 0.02,
            c_c: 0.0, 
            eps: 0.2,
            advantage: AdvantageEstimator::Gae { lambda: 0.95 },
            loss_fn: LossFn::MseLoss,
        };
    let env = BatchedGymnasiumEnvironmentConfig::lunar_lander().seed(seed).init(16, &device);
    let env = BatchedRecordEpisodeStatistics::new(env, 100);

    let mut ac = TestNet::new(vec![env.obs_space().shape()[1], 128, env.action_space().n()], ActivationConfig::Relu, &device);
    let lr_a = 1e-3;
    let lr_c = 2.5e-3;
    let mut opt_a = AdamConfig::new().init();
    let mut opt_c = AdamConfig::new().init();

    let mut buffer = RolloutBuffer::new(256);
    let sampler = ShuffleSampler;
    let mut tape = BatchedTape::new(env, true);

    let total_steps = 100000;
    let c_e_sch = LinearScheduler::new(state.c_e, 0.002, total_steps, 1.0);

    for count in 0..total_steps {
        let dist = ac.dist(tape.obs.clone(), tape.constraint.clone());
        let action = dist.sample();
        let (mut t, _) = tape.step(action.clone());
        t.insert(batched_key::log_prob, dist.log_probs(action)).unwrap();

        buffer.push(t);

        if buffer.is_full() {
            let mut batch = buffer.pop();
            let (adv, ret) = state.advantage.normalized_advantage(&ac, batch.clone(), state.gamma);
            batch.insert(batched_key::adv, adv).unwrap(); batch.insert(batched_key::ret, ret).unwrap();
            let batch = batch.flatten();
            for _ in 0..4 {
                for minibatch in sampler.iter(128, &batch, &device) {
                    let (net, loss) = ppo_loss(&state, ac, minibatch.sample);
                    ac = net.update(loss.clone(), lr_a, lr_c, &mut opt_a, &mut opt_c);
                    collector.record_mean(2, "loss", loss.loss);
                    collector.record_mean(2, "value_loss", loss.value_loss);
                    collector.record_mean(2, "entropy", loss.entropy);
                    collector.record_mean(2, "approx_kl", loss.approx_kl);
                    collector.record_mean(2, "clip_fraction", loss.clip_fraction);
                }
            }
        }

        if count % 5000 == 0 {
            // log
            collector.record(1, "ep/reward_mean", tape.env().reward_mean());
            collector.record(1, "ep/length_mean", tape.env().length_mean());
            collector.record(1, "ep/time_mean", tape.env().time_mean());
            collector.record(-1, "elapsed time", s.elapsed().as_secs_f64());
            collector.dump_to(count, &mut sink).expect("unexpected error");
        }

        state.c_e = c_e_sch.value(count) as f32;
    }

    let d = s.elapsed();

    println!("It took {}s", d.as_secs());

    let env = BatchedGymnasiumEnvironmentConfig::lunar_lander().kwarg("render_mode", "human").init(1, &device);

    let mut tape = BatchedTape::new(env, false);
    for _ in 0..5 {
        loop {
            let action = ac.dist(tape.obs.clone(), tape.constraint.clone()).mode();
            tape.step(action);
            if tape.done().clone().into_scalar() {
                tape.reset(None);
                break;
            }
        }
    }
}

#[derive(Module, Debug)]
pub struct TestNet {
    a_lv: Vec<Linear>,
    c_lv: Vec<Linear>,
    activation: Activation
}

#[actor_critic(distribution = Categorical)]
impl TestNet {
    pub fn new(dims: Vec<usize>, activation: ActivationConfig, device: &Device) -> Self {
        let n = dims.len();
        let mut a_lv = Vec::with_capacity(n);
        let mut c_lv = Vec::with_capacity(n);
        for i in 0..(n - 1) {
            a_lv.push(LinearConfig::new(dims[i], dims[i + 1]).init(device));
            if i == n - 2 {
                c_lv.push(LinearConfig::new(dims[i], 1).init(device));
            } else {
                c_lv.push(LinearConfig::new(dims[i], dims[i + 1]).init(device));
            }
        }

        Self {
            a_lv,
            c_lv,
            activation: activation.init(device),
        }
    }

    pub fn actor(&self, mut obs: Tensor<2>) -> Tensor<2> {
        let n = self.a_lv.len();
        for i in 0..(n - 1) {
            obs = self.activation.forward(self.a_lv[i].forward(obs));
        }
        obs = self.a_lv[n - 1].forward(obs);
        obs
    }

    pub fn critic(&self, mut obs: Tensor<2>) -> Tensor<1> {
        let n = self.c_lv.len();
        for i in 0..(n - 1) {
            obs = self.activation.forward(self.c_lv[i].forward(obs));
        }
        obs = self.c_lv[n - 1].forward(obs);
        obs.squeeze_dim(1)
    }
}