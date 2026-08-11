use std::marker::PhantomData;

use bake::{agent::vpg::{Baseline, VpgAgent}, buffer::EpisodeBuffer, encoderhead::{EncoderHead, LinearHead, MlpEncoder}, environment::{CartPole, Environment}, transition::Transition};
use burn::{backend::{Cuda, NdArray, cuda::CudaDevice, ndarray::NdArrayDevice::Cpu}, nn::Relu, optim::AdamConfig, prelude::*};
use burn_autodiff::Autodiff;

type Engine = NdArray;
type AutodiffEngine = Autodiff<Engine>;

fn main() {
    let device = Cpu;
    // let device = CudaDevice::new(0);
    AutodiffEngine::seed(&device, 12);
    let mut env = CartPole::new(123);
    let mut agent = VpgAgent::new(
        12,
        0.99f32,
        0.0,
        device.clone(),
        Baseline::Mean,
        EncoderHead::new(MlpEncoder::new(vec![4, 128], nn::activation::Activation::Relu(Relu), &device), LinearHead::<AutodiffEngine>::new(128, 2, &device)),
        AdamConfig::new().init(),
        1e-3,
        
    );
    let mut buffer = EpisodeBuffer::new(device.clone());

    for episode in 0..=4000 {
        let (mut obs, _) = env.reset();
        let mut steps = 0;
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
                extra: (),
                _backend: PhantomData
            });

            obs = next_obs;
            steps += 1;
            if terminated || truncated {
                let batch = buffer.pop();
                (agent, entropy) = agent.update(batch);
                break;
            }
        }
        if episode % 100 == 0 { println!("episode: {episode}, steps: {steps}, entropy: {entropy}"); }
    }
}