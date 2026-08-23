use bake_deep::{agent::DQNAgent, buffer::ReplayBuffer, approximator::encoder::MLPEncoder, env::CartPole, approximator::head::LinearDuelingQHead, exploration::EpsGreedy, approximator::ComposedQFunction, types::Tape};
use burn::{nn::{Relu}, optim::AdamConfig, tensor::Device};
use burn::nn::activation::Activation;
pub fn main() {
    let device = Device::default().autodiff();
    device.seed(12);
    let mut env = CartPole::new(12, &device);
    let mut agent = DQNAgent::new(0.99,    
        ComposedQFunction::new(
            MLPEncoder::new(vec![4, 128], Activation::Relu(Relu), &device),
            LinearDuelingQHead::new(128, 2, &device)
        ),
        1e-3,
            AdamConfig::new().init()
        );
    let mut policy = EpsGreedy::new(123, 1.0f32);
    let mut buffer = ReplayBuffer::new(12, 10000);
    let mut tape = Tape::new(&mut env);
    let mut count = 0;
    let mut log = Default::default();
    for episode in 0..=4000 {
        let mut steps = 0;
        tape.reset(&mut env);
        loop {
            let action = agent.action(&mut policy, tape.obs.clone(), tape.constraint.clone());
            let t = tape.step(&mut env, action);
            buffer.push(t);

            if let Some(batch) = buffer.sample(64) {
                (agent, log) = agent.update(batch);
            }

            if count % 1000 == 0 {
                agent.sync();
            }
            count += 1;
            steps += 1;
            if tape.done() { break; }
        }
        *policy.eps_mut() *= 0.999;
        if episode % 100 == 0 { println!("episode: {episode}, steps: {steps}, loss: {}, q_mean: {}, td_error: {}, eps: {}", log.loss(), log.q_mean(), log.mean_td_error(), policy.eps()); }
    }
}