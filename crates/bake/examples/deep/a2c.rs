use bake_deep::{agent::A2CAgent, buffer::RolloutBuffer, encoder::MLPEncoder, env::CartPole, head::{CategoricalHead, ValueHead}, network::SequentialActorCriticNetwork, types::Tape};
use burn::{Tensor, nn::activation::Activation, optim::AdamConfig, tensor::Device};


pub fn main() {
    let device = Device::default().autodiff();
    device.seed(12);
    let mut env = CartPole::new(12, &device);
    let mut agent = A2CAgent::new(
        0.99,
        1.0,
        0.03,
        1e-3,
        AdamConfig::new().init(),
        SequentialActorCriticNetwork::new(
            MLPEncoder::new(vec![4, 128], Activation::Relu(burn::nn::Relu), &device),
            MLPEncoder::new(vec![4, 128], Activation::Relu(burn::nn::Relu), &device),
            CategoricalHead::new(128, 2, &device),
            ValueHead::new(128, 1, &device)
        )
    );
    let mut buffer = RolloutBuffer::new(64.into(), device.clone());
    let mut tape = Tape::new(&mut env);

    for i in 0..=4000 {
        let mut entropy: Tensor<1> = Tensor::zeros([1], &device);
        let mut loss: Tensor<1> = Tensor::zeros([1], &device);
        let mut policy_loss: Tensor<1> = Tensor::zeros([1], &device);
        let mut step = 0;
        tape.reset(&mut env);
        loop {
            let action = agent.action(tape.obs.clone(), tape.mask.clone());
            let t = tape.step(&mut env, action);
            let done = t.terminated || t.truncated;

            buffer.push(t);

            step += 1;
            if buffer.is_full() {
                let batch = buffer.pop();
                (agent, loss, policy_loss, entropy) = agent.update(batch);
            }
            if done { break; }
        }

        if i % 100 == 0 {
            println!("Episode: {i}, Steps: {step}, Loss: {}, Policy Loss: {}, Entropy: {}", loss.into_scalar::<f32>(), policy_loss.mean().into_scalar::<f32>(), entropy.into_scalar::<f32>());
        }
    }
}