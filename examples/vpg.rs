use std::marker::PhantomData;

use bake::{agent::vpg::{Baseline, VpgAgent}, buffer::EpisodeBuffer, encoderhead::{EncoderHead, LinearHead, MlpEncoder}, environment::{CartPole, Environment}, transition::Transition};
use burn::{backend::{NdArray, ndarray::NdArrayDevice::Cpu}, nn::Relu, optim::{AdamConfig}, prelude::*};
use burn_autodiff::Autodiff;

type Engine = NdArray;
type AutodiffEngine = Autodiff<Engine>;

fn main() {
    let device = Cpu;
    AutodiffEngine::seed(&device, 12);
    let mut env = CartPole::new(123);
    let mut agent = VpgAgent::new(
        0.99f32,
        Baseline::Mean,
        EncoderHead::new(MlpEncoder::new(vec![4, 128], nn::activation::Activation::Relu(Relu), &device), LinearHead::<AutodiffEngine>::new(128, 2, &device)),
        AdamConfig::new().init(),
        1e-3,
        device
    );
    let mut buffer = EpisodeBuffer::new(device.clone());

    for episode in 0..=4000 {
        let mut obs = env.reset();
        let mut steps = 0;
        let mut loss = 0f32;

        loop {
            let action = agent.select_action(obs);
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

            obs = next_obs;
            steps += 1;
            if terminated || truncated {
                let batch = buffer.pop();
                (agent, loss) = agent.update(batch);
                break;
            }
        }
        if episode % 100 == 0 { println!("episode: {episode}, steps: {steps}, loss: {loss}"); }
    }
}