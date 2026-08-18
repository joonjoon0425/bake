use bake_deep::{agent::{Baseline, VPGAgent}, buffer::RolloutBuffer, encoder::MLPEncoder, env::CartPole, head::CategoricalHead, network::SequentialLogitNetwork, types::Tape};
use burn::{Tensor, nn::activation::Activation, optim::AdamConfig, tensor::Device};


pub fn main() {
    let device = Device::default().autodiff();
    device.seed(1);
    let mut env = CartPole::new(12, &device);
    let mut agent = VPGAgent::new(
        0.99,
        Baseline::Mean,
        SequentialLogitNetwork::new(
            MLPEncoder::new(vec![4, 128], Activation::Relu(burn::nn::Relu), &device),
            CategoricalHead::new(128, 2, &device)
        ),
        0.00,
        AdamConfig::new().init(),
        1e-3
    );
    let mut buffer = RolloutBuffer::new(None, device.clone());
    let mut tape = Tape::new(&mut env);

    for i in 0..=4000 {
        let mut entropy: Tensor<1> = Tensor::zeros([1], &device);
        let mut step = 0;
        tape.reset(&mut env);
        loop {
            let action = agent.action(tape.obs.clone(), tape.mask.clone());
            let t = tape.step(&mut env, action);
            let done = t.terminated || t.truncated;

            buffer.push(t);

            step += 1;
            if done {
                let batch = buffer.pop();
                (agent, entropy) = agent.update(batch);
                break;
            }
        }

        if i % 100 == 0 {
            println!("Episode: {i}, Steps: {step}, Entropy: {}", entropy.mean().into_scalar::<f32>());
        }
    }
}