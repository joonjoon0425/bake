use std::marker::PhantomData;

use bake::{agent::{dqn::DqnAgent}, buffer::ReplayBuffer, encoderhead::{Encoder, EncoderHead, Head}, environment::{CartPole, Environment}, exploration::EpsGreedy, transition::Transition};
use burn::{backend::{NdArray, ndarray::NdArrayDevice::Cpu}, nn::{Linear, LinearConfig, Relu}, optim::{AdamConfig}, prelude::*};
use burn::module::Module;
use burn_autodiff::Autodiff;

type Engine = NdArray;
type AutodiffEngine = Autodiff<Engine>;

fn main() {
    let device = Cpu;
    AutodiffEngine::seed(&device, 12);
    let mut env = CartPole::new(123);
    let mut agent = DqnAgent::new(
        0.99f32, 
        EncoderHead::new(IdentityEnc::new(), QHead::<AutodiffEngine>::new(4, 2, &device)),
        AdamConfig::new().init(),
        1e-3,
        device
    );
    let mut policy = EpsGreedy::new(1.0, 123);
    let mut buffer = ReplayBuffer::new(123, 10000, device.clone());

    for episode in 0..=4000 {
        let mut obs = env.reset();
        let mut steps = 0;
        let mut count = 0;
        let mut loss = 0f32;
        let mut td_error = 0f32;
        let mut q_mean = 0f32;

        loop {
            let action = agent.select_action(&mut policy, obs);
            let (next_obs, reward, terminated, truncated) = env.step(action);

            buffer.push(Transition {
                observation: obs,
                action,
                reward,
                next_observation: next_obs,
                terminated,
                truncated,
                extra: (),
                _backend: PhantomData
            });

            if let Some(batch) = buffer.sample(64) {
                (agent, loss, td_error, q_mean) = agent.update(batch);
            }

            if count % 400 == 0 {
                agent.sync();
            }

            obs = next_obs;
            count += 1;
            steps += 1;
            if terminated || truncated { break; }
        }
        *policy.eps_mut() *= 0.999;
        if episode % 100 == 0 { println!("episode: {episode}, steps: {steps}, loss: {loss}, td_error: {td_error}, q_mean: {q_mean}, eps: {}", policy.eps()); }
    }
}

#[derive(Module, Debug)]
struct QHead<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    activation: Relu,
}

impl<B: Backend> QHead<B> {
    pub fn new(d_input: usize, d_output: usize, device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(d_input, 128).init(device),
            fc2: LinearConfig::new(128, d_output).init(device),
            activation: Relu::new()
        }
    }
}

impl<B: Backend> Head<B, 2> for QHead<B> {
    type Output = Tensor<B, 2>;

    fn forward(&self, encoded: Tensor<B, 2>) -> Self::Output {
        let x = self.fc1.forward(encoded);
        let x = self.activation.forward(x);
        let x = self.fc2.forward(x);

        x
    }
}

#[derive(Module, Debug)]
struct IdentityEnc<B: Backend> {
    _backend: PhantomData<B>
}

impl<B: Backend> IdentityEnc<B> {
    pub fn new() -> Self {
        Self {
            _backend: PhantomData
        }
    }
}

impl<B: Backend> Encoder<B, 2> for IdentityEnc<B> {
    type Obs = [f32; 4];

    fn forward(&self, batched_obs: <Self::Obs as bake::traits::Batchable<B>>::Batched) -> Tensor<B, 2> {
        batched_obs    
    }
}