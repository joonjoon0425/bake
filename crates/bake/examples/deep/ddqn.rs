use bake_deep::{agent::DoubleDQNAgent, buffer::ReplayBuffer, approximator::encoder::MLPEncoder, env::CartPole, approximator::head::{LinearQHead}, approximator::ComposedQFunction, exploration::EpsGreedy, types::Tape};
use burn::{grad_clipping::GradientClippingConfig, nn::Relu, optim::AdamConfig, tensor::Device};
use burn::nn::activation::Activation;
pub fn main() {
    let seed: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    let device = Device::default().autodiff();
    device.seed(seed);
    println!("# optimizer=adam lr=1e-3 buffer_capacity=10000 sync_freq=1000 update_freq=10 T=64 seed={seed}");
    println!("episode,total_steps,step,loss,q_mean,td_error,eps");
    let mut env = CartPole::new(seed, &device);
    let mut agent = DoubleDQNAgent::new(0.99,    
        ComposedQFunction::new(
            MLPEncoder::new(vec![4, 128], Activation::Relu(Relu), &device),
            LinearQHead::new(128, 2, &device)
        ),
        1e-3,
            AdamConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(10.0))).init()
        );
    let mut policy = EpsGreedy::new(seed, 1.0f32);
    let mut buffer = ReplayBuffer::new(12, 10000);
    let mut tape = Tape::new(&mut env);
    let mut count = 0;
    let mut log = Default::default();
    for episode in 0..=4000 {
        let mut steps = 0;
        tape.reset(&mut env);
        loop {
            let action = agent.action(&mut policy, tape.obs.clone(), tape.constraint.clone());
            let t= tape.step(&mut env, action);
            buffer.push(t);

            if count % 10 == 0 && let Some(batch) = buffer.sample(64) {
                (agent, log) = agent.update(batch);
            }

            if count % 1000 == 0 {
                agent.sync();
            }
            count += 1;
            steps += 1;
            if tape.done() { break; }
        }
        *policy.eps_mut() = (policy.eps() * 0.999).max(0.05);
        if episode % 10 == 0 {
            let loss = log.loss();
            let q_mean = log.q_mean();
            let td_error = log.mean_td_error();
            let eps = policy.eps();
            println!("{episode} {count} {steps} {loss} {q_mean} {td_error} {eps}");
            if episode % 100 == 0 {
                eprintln!("Episode: {episode} Total steps: {count} Steps: {steps} Loss: {loss} Q-Mean: {q_mean} TD-error: {td_error} Eps: {eps}");
            }
        }
    }
}