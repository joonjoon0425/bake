use bake_deep::{agent::DQNAgent, buffer::ReplayBuffer, encoder::MLPEncoder, env::CartPole, head::QHead, policy::EpsGreedy, network::SequentialQNetwork, types::Tape};
use burn::{Tensor, nn::{Relu}, optim::AdamConfig, tensor::Device};
use burn::nn::activation::Activation;
pub fn main() {
    let device = Device::default().autodiff();
    device.seed(12);
    let mut env = CartPole::new(12, &device);
    let mut agent = DQNAgent::new(0.99,    
        SequentialQNetwork::new(
            MLPEncoder::new(vec![4, 128], Activation::Relu(Relu), &device),
            QHead::new(128, 2, &device)
        ),
        1e-3,
            AdamConfig::new().init()
        );
    let mut policy = EpsGreedy::new(123, 1.0f32);
    let mut buffer = ReplayBuffer::new(12, 10000, device.clone());
    let mut tape = Tape::new(&mut env);
    let mut count = 0;
    for episode in 0..=4000 {
        let mut steps = 0;
        let mut loss: Tensor<1> = Tensor::zeros([1], &device);
        let mut td_error: Tensor<1> = Tensor::zeros([1], &device);
        let mut q_mean: Tensor<1> = Tensor::zeros([1], &device);
        tape.reset(&mut env);
        loop {
            let action = agent.action(&mut policy, tape.obs.clone(), tape.mask.clone());
            let t = tape.step(&mut env, action);
            let done = t.terminated || t.truncated;
            buffer.push(t);

            if let Some(batch) = buffer.sample(64) {
                (agent, q_mean, loss, td_error) = agent.update(batch);
            }

            if count % 1000 == 0 {
                agent.sync();
            }
            count += 1;
            steps += 1;
            if done { break; }
        }
        *policy.eps_mut() *= 0.999;
        if episode % 100 == 0 { println!("episode: {episode}, steps: {steps}, loss: {}, q_mean: {}, td_error: {}, eps: {}", loss.into_scalar::<f32>(), q_mean.into_scalar::<f32>(), td_error.mean().into_scalar::<f32>(), policy.eps()); }
    }
}