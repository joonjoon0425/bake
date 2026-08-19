use bake_deep::{agent::{A2CAgent, A2CConfig}, buffer::RolloutBuffer, encoder::MLPEncoder, env::CartPole, head::{CategoricalHead, LinearValueHead}, network::SequentialActorCriticNetwork, types::Tape};
use burn::{Tensor, nn::activation::Activation, optim::{AdamConfig, RmsPropConfig}, tensor::Device};


pub fn main() {
    let device = Device::default().autodiff();
    device.seed(4);
    let mut env = CartPole::new(12, &device);
    let mut agent = A2CAgent::new(
        0.99,
        0.02,
        A2CConfig::separated(
            1e-4,
            RmsPropConfig::new().init(),
            1e-3,
            RmsPropConfig::new().init()
        ),
        SequentialActorCriticNetwork::new(
            MLPEncoder::new(vec![4, 128], Activation::Relu(burn::nn::Relu), &device),
            MLPEncoder::new(vec![4, 128], Activation::Relu(burn::nn::Relu), &device),
            CategoricalHead::new(128, 2, &device),
            LinearValueHead::new(128, 1, &device)
        )
    );
    let mut buffer = RolloutBuffer::new(160.into(), device.clone());
    let mut tape = Tape::new(&mut env);

    for i in 0..=4000 {
        let mut entropy: Tensor<1> = Tensor::zeros([1], &device);
        let mut loss: Tensor<1> = Tensor::zeros([1], &device);
        let mut step = 0;
        tape.reset(&mut env);
        loop {
            let action = agent.action(tape.obs.clone(), tape.constraint.clone());
            let t = tape.step(&mut env, action);
            let done = t.terminated || t.truncated;

            buffer.push(t);

            step += 1;
            if buffer.is_full() {
                let batch = buffer.pop();
                (agent, loss, entropy) = agent.update(batch);
            }
            if done { break; }
        }

        if i % 100 == 0 {
            println!("Episode: {i}, Steps: {step}, Loss: {}, Entropy: {}", loss.into_scalar::<f32>(), entropy.into_scalar::<f32>());
        }
    }
}