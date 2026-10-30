use bake_rl::env::gym::GymnasiumEnvironmentConfig;
use bake_rl::prelude::*;
use bake_rl::deep::prelude::*;
use bake_rl::core::env::wrapper::sync_env::SynchronizedVectorEnvironmentConfig;
use burn::nn::activation::ActivationConfig;
use burn::prelude::*;
use burn::nn::*;
use burn::nn::activation::Activation;
use burn::optim::*;

pub fn main() {
    let seed = 12;
    let device = Device::default();
    device.seed(seed);
    let env = SynchronizedVectorEnvironmentConfig::new(vec![
        GymnasiumEnvironmentConfig::cart_pole().seed(seed)
    ]).init(&device);
    
    let mut tape = Tape::new(env, false);

    let state = DQNState { gamma: 0.99, loss_fn: LossFn::MseLoss };
    let mut online = TestNet::new(vec![tape.obs_space().shape()[1], 128, 84, tape.action_space().n()], ActivationConfig::Relu, &device);
    let mut target = online.clone();
    let lr = 2.5e-4;
    let mut opt = AdamConfig::new().init();
        
    let mut exploration = EpsGreedy::new(seed, 1.0f32);
    let mut buffer = ReplayBufferConfig::prioritized(seed, 50000, 0.6, 0.4).with_priority_clip(1.0).init();

    let total_steps = 500000;
    let warmup = 10000;
    let update_freq = 10;
    let sync_freq = 500;
    let batch_size = 128;

    // temporary, until the logger is implemented...
    let mut reward = 0f32;
    let mut step = 0f32;
    
    // temporary, until the logger is implemented...
    let mut reward_vec = vec![];
    let mut step_vec = vec![];


    let mut eps_sch = LinearScheduler::new(1.0, 0.05, total_steps, 0.5);
    let mut beta_sch = LinearScheduler::new(0.4, 1.0, total_steps, 1.0);

    for count in 0..total_steps {
        let action = exploration.sample(&online, tape.obs.clone(), tape.constraint.clone());
        let batch = tape.step(action);
        
        reward = reward + batch.reward.clone().detach_batch_dim().0.into_scalar::<f32>();
        step += 1.;

        buffer.push(batch);

        if count >= warmup && count % update_freq == 0 && let Some((batch, batch_info)) = buffer.sample(batch_size) {
            let (net, loss) = dqn_loss(&state, online, &target, batch, batch_info.clone());
            buffer.update_priority(&batch_info.indices, loss.td_error.clone());
            online = net.update(loss, lr, &mut opt);
        }

        if count % sync_freq == 0 {
            target = online.clone();
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
            println!("{count},{avg_reward},{avg_step},{}", exploration.eps());
        }

        *exploration.eps_mut() = eps_sch.step() as f32;
        *buffer.beta_mut() = beta_sch.step();
    }

    // evaluation
    let env = SynchronizedVectorEnvironmentConfig::new(vec![
        GymnasiumEnvironmentConfig::cart_pole().kwarg("render_mode", "human").seed(12) ,
    ]).init(&device);

    let mut tape = Tape::new(env, false);
    let mut eval = Greedy;
    for _ in 0..5 {
        loop {
            let action = eval.sample(&online, tape.obs.clone(), tape.constraint.clone());
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

    pub fn forward(&self, obs: BatchTensor<2>) -> BatchTensor<2> {
        let (mut obs, bs) = obs.detach_batch_dim();
        let n = self.linear.len();
        for i in 0..(n - 1) {
            obs = self.activation.forward(self.linear[i].forward(obs));
        }
        obs = self.linear[n - 1].forward(obs);
        obs.attach_batch_dim(bs)
    }
}