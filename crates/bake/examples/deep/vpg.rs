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
    let mut buffer = RolloutBuffer::new();
    let mut tape = Tape::new(&mut env);
    let mut log = Default::default();

    for i in 0..=4000 {
        let mut step = 0;
        tape.reset(&mut env);
        loop {
            let action = agent.action(tape.obs.clone(), tape.constraint.clone());
            let (t, _, terminated, truncated) = tape.step(&mut env, action);
            buffer.push(t);

            step += 1;
            if terminated || truncated {
                let batch = buffer.pop();
                (agent, log) = agent.update(batch);
                break;
            }
        }

        if i % 100 == 0 {
            println!("Episode: {i}, Steps: {step}, Entropy: {}, Surrogate loss: {}", log.entropy(), log.surrogate_loss());
        }
    }
}