use bake_rl::env::gym::GymnasiumEnvironmentConfig;
use bake_rl::prelude::*;
use bake_rl::core::env::wrapper::sync_env::SynchronizedVectorEnvironmentConfig;
use burn::nn::activation::ActivationConfig;
use burn::prelude::*;
use burn::nn::*;
use burn::nn::activation::Activation;

pub fn main() {
    let device = Device::default();
    let env = SynchronizedVectorEnvironmentConfig::new(vec![
        GymnasiumEnvironmentConfig::bipedalwalker().kwarg("render_mode", "human").seed(10) ,
    ]).init(&device);

    let mut tape = Tape::new(env, true);

    for _ in 0..100 {
        let _ = tape.step(
            tape.action_space().sample(1)
        );
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
        obs.attach_batch_dim(bs)
    }
}