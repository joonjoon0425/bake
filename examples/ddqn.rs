use std::marker::PhantomData;

use bake::{agent::DoubleDqnAgent, buffer::ReplayBuffer, encoderhead::{DeulingHead, EncoderHead, LinearHead, MlpEncoder}, environment::{CartPole, Environment}, exploration::EpsGreedy, transition::Transition};
use burn::{backend::{NdArray, ndarray::NdArrayDevice::Cpu}, nn::Relu, optim::{AdamConfig}, prelude::*};
use burn_autodiff::Autodiff;

type Engine = NdArray;
type AutodiffEngine = Autodiff<Engine>;

fn main() {
    let device = Cpu;
    AutodiffEngine::seed(&device, 12);
    let mut env = CartPole::new(123);
    let mut agent = DoubleDqnAgent::new(
        0.99f32, 
        EncoderHead::new(MlpEncoder::new(vec![4, 128], nn::activation::Activation::Relu(Relu), &device), DeulingHead::<AutodiffEngine>::new(128, 2, &device)),
        AdamConfig::new().init(),
        2.5e-4,
        device
    );
    let mut policy = EpsGreedy::new(1.0, 123);
    let mut buffer = ReplayBuffer::new(123, 10000, device.clone());

    for episode in 0..=4000 {
        let (mut obs, _) = env.reset();
        let mut steps = 0;
        let mut count = 0;
        let mut loss = 0f32;
        let mut td_error = 0f32;
        let mut q_mean = 0f32;

        loop {
            let action = agent.select_action(&mut policy, obs);
            let ((next_obs, _), reward, terminated, truncated) = env.step(action);

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

            if count % 1000 == 0 {
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