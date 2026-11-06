use bake_rl::core::env::wrapper::sync_env::SynchronizedVectorEnvironment;
use bake_rl::core::env::wrapper::record_episode_statistics::RecordEpisodeStatistics;
use bake_rl::core::scheduler::LinearScheduler;
use bake_rl::core::scheduler::Scheduler;
use bake_rl::env::gym::GymnasiumEnvironmentConfig;
use bake_rl::deep::prelude::*;
use burn::prelude::*;
use burn::nn::*;
use burn::nn::activation::{Activation, ActivationConfig};
use burn::optim::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;

pub fn main() {
    let seed: u64 = 120;
    let mut rng = SmallRng::seed_from_u64(seed);
    let device = Device::default();
    device.seed(seed);

    let mut state =
        PPOState {
            gamma: 0.99,
            c_e: 0.02,
            c_c: 0.0, 
            eps: 0.2,
            advantage: AdvantageEstimator::Gae { lambda: 0.95 },
            loss_fn: LossFn::MseLoss,
        };
    let mut env = vec![];
    for i in 0..16 { env.push(GymnasiumEnvironmentConfig::lunar_lander().seed(i).init()) }
    let env = SynchronizedVectorEnvironment::new(env, &device);
    let env = RecordEpisodeStatistics::new(env);

    let mut ac = TestNet::new(vec![env.obs_space().shape()[1], 64, 64, env.action_space().n()], ActivationConfig::Relu, &device);
    let lr_a = 1e-3;
    let lr_c = 2.5e-3;
    let mut opt_a = AdamConfig::new().init();
    let mut opt_c = AdamConfig::new().init();

    let mut buffer = RolloutBuffer::new();
    let mut tape = VectorTape::new(env, true);

    let total_steps = 100000;
    let mut c_e_sch = LinearScheduler::new(state.c_e as f64, 0.002, total_steps, 1.0);

    for count in 0..total_steps {
        let dist = ac.dist(tape.obs.clone(), tape.constraint.clone());
        let action = dist.sample();
        let (mut t, info) = tape.step(action.clone());
        t.insert(key::log_prob, dist.log_probs(action)).unwrap();

        buffer.push(t);

        if buffer.len() >= 1024 {
            let mut batch = buffer.pop();
            let (adv, ret) = state.advantage.normalized_advantage(&ac, batch.clone(), state.gamma);
            batch.insert(key::adv, adv).unwrap(); batch.insert(key::ret, ret).unwrap();
            let batch = batch.flatten();
            for _ in 0..4 {
                for minibatch in batch.minibatch(0, 128, &mut rng, &device) {
                    let (net, loss) = ppo_loss(&state, ac, minibatch);
                    ac = net.update(loss, lr_a, lr_c, &mut opt_a, &mut opt_c);
                }
            }
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
    let env = GymnasiumEnvironmentConfig::lunar_lander().kwarg("render_mode", "human").init();
    let env = SynchronizedVectorEnvironment::new(vec![env], &device);

    let mut tape = VectorTape::new(env, false);
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