use std::marker::PhantomData;

use bake::{agent::A2CAgent, buffer::RolloutBuffer, encoderhead::{EncoderHead, LinearHead, MlpEncoder}, environment::{CartPole, Environment}, transition::Transition};
use burn::{backend::{NdArray, ndarray::NdArrayDevice::Cpu}, nn::Relu, optim::AdamConfig, prelude::*};
use burn_autodiff::Autodiff;

type Engine = NdArray;
type AutodiffEngine = Autodiff<Engine>;

fn main() {
    let device = Cpu;
    // let device = CudaDevice::new(0);
    AutodiffEngine::seed(&device, 12);
    let mut env = CartPole::new(123);
    let mut agent = A2CAgent::new(
        0.99f32,
        0.01,
        1e-3,
        1e-3,
        device.clone(),
        EncoderHead::new(MlpEncoder::new(vec![4, 128], nn::activation::Activation::Relu(Relu), &device), LinearHead::new(128, 2, &device)),
        EncoderHead::new(MlpEncoder::new(vec![4, 128], nn::activation::Activation::Relu(Relu), &device), LinearHead::<AutodiffEngine>::new(128, 1, &device)),
        AdamConfig::new().init(),
        AdamConfig::new().init(),
    );
    let mut buffer = RolloutBuffer::new(64, device.clone());

    for episode in 0..=4000 {
        let (mut obs, _) = env.reset();
        let mut steps = 0;
        let mut loss = 0f32;
        let mut entropy = 0f32;

        loop {
            let action = agent.select_action(obs);
            let ((next_obs, _), reward, terminated, truncated) = env.step(action);

            buffer.push(Transition {
                observation: obs,
                action,
                reward,
                next_observation: next_obs,
                terminated,
                truncated,
                mask: (),
                next_mask: (),
                extra: (),
                _backend: PhantomData
            });

            obs = next_obs;
            steps += 1;

            if buffer.is_full() {
                let steps = buffer.pop();
                (agent, entropy, loss) = agent.update(next_obs, steps);
            }


            if terminated || truncated {
                break;
            }
        }
        if episode % 100 == 0 { println!("episode: {episode}, steps: {steps}, entropy: {entropy}, value loss: {loss}"); }
    }
}