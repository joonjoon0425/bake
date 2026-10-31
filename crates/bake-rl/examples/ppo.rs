use bake_rl::core::distribution::Categorical;
use bake_rl::core::env::wrapper::sync_env::SynchronizedVectorEnvironmentConfig;
use bake_rl::env::gym::GymnasiumEnvironmentConfig;
use bake_rl::prelude::*;
use bake_rl::deep::prelude::*;
use burn::prelude::*;
use burn::nn::*;
use burn::nn::activation::{Activation, ActivationConfig};
use burn::optim::*;

pub fn main() {
    let seed: u64 = 120;
    let device = Device::default();
    device.seed(seed);

    let mut state =
        PPOState {
            gamma: 0.99,
            c_e: 0.02,
            c_c: 0.0, 
            eps: 0.2,
            advantage: AdvantageEstimator::Gae { lambda: 0.96 },
            loss_fn: LossFn::MseLoss,
        };
    let env = SynchronizedVectorEnvironmentConfig::new(
        vec![GymnasiumEnvironmentConfig::lunar_lander()]
    ).init(&device);

    let mut ac = TestNet::new(vec![env.obs_space().shape()[1], 128, 84, env.action_space().n()], ActivationConfig::Relu, &device);
    let lr_a = 1e-3;
    let lr_c = 2.5e-3;
    let mut opt_a = AdamConfig::new().init();
    let mut opt_c = AdamConfig::new().init();

    let mut buffer = RolloutBuffer::new();
    let mut tape = Tape::new(env, false);

    let total_steps = 1000000;
    let mut c_e_sch = LinearScheduler::new(state.c_e as f64, 0.002, total_steps, 1.0);
        
    // temporary, until the logger is implemented...
    let mut reward = 0f32;
    let mut step = 0f32;
    
    // temporary, until the logger is implemented...
    let mut reward_vec = vec![];
    let mut step_vec = vec![];


    for count in 0..total_steps {
        let dist = ac.dist(tape.obs.clone(), tape.constraint.clone());
        let action = dist.sample();
        let mut t = tape.step(action.clone());
        t.insert(key::log_prob, dist.log_probs(action)).unwrap();

        reward = reward + t.reward.clone().into_scalar::<f32>();
        step += 1.;

        buffer.push(t);

        if buffer.len() >= 2048 {
            let mut batch = buffer.pop();
            let (adv, ret) = state.advantage.normalized_advantage(&ac, batch.clone(), state.gamma);
            batch.insert(key::adv, adv).unwrap(); batch.insert(key::ret, ret).unwrap();
            for _ in 0..4 {
                for minibatch in batch.minibatch(0, 128, seed, &device) {
                    let (net, loss) = ppo_loss(&state, ac, minibatch);
                    ac = net.update(loss, lr_a, lr_c, &mut opt_a, &mut opt_c);
                }
            }
        }

        if tape.done().clone().into_scalar() {
            reward_vec.push(reward);
            step_vec.push(step);
            tape.reset(None);
            reward = 0.;
            step = 0.;
        }

        if count % 5000 == 0 {
            let avg_reward = reward_vec.iter().rev().take(100).sum::<f32>() / 100.;
            let avg_step = step_vec.iter().rev().take(100).sum::<f32>() / 100.;
            println!("{count},{avg_reward},{avg_step}");
        }

        state.c_e = c_e_sch.step() as f32;
    }

    let env = SynchronizedVectorEnvironmentConfig::new(
        vec![GymnasiumEnvironmentConfig::lunar_lander().kwarg("render_mode", "human")]
    ).init(&device);

    let mut tape = Tape::new(env, false);
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