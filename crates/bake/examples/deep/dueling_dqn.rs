use bake_deep::{algorithm::{Dqn, dqn::ValueLoss}, approximator::{ComposedQFunction, LinearDuelingQHead, encoder::MlpEncoder}, buffer::ReplayBuffer, env::CartPole, exploration::{EpsGreedy, Exploration}, types::{Logger, Tape}};
use burn::{grad_clipping::GradientClippingConfig, nn::activation::ActivationConfig::Relu, optim::AdamConfig, tensor::Device};
pub fn main() {
    let seed: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    let device = Device::default().autodiff();
    device.seed(seed);
    let mut env = CartPole::new(seed, &device);
    let config = Dqn::new(0.99, ValueLoss::MseLoss);
    let mut online = ComposedQFunction::new(
            MlpEncoder::new(vec![4, 128], Relu, &device),
            LinearDuelingQHead::new(128, 2, &device)
        );
    let mut target = online.clone();
    let lr = 1e-3;
    let mut opt = AdamConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(10.0))).init();

    let mut exploration = EpsGreedy::new(seed, 1.0f32);
    let mut buffer = ReplayBuffer::new(12, 10000);
    let mut tape = Tape::new(&mut env);
    let mut count = 0;
    let mut logger = Logger::default();
    for episode in 0..=4000 {
        let mut steps = 0;
        tape.reset(&mut env);
        loop {
            let action = exploration.sample(&online, tape.obs.clone(), tape.constraint.clone());
            let t = tape.step(&mut env, action);
            buffer.push(t);

            if count % 10 == 0 && let Some(batch) = buffer.sample(64) {
                let loss = Dqn::loss(&config, &online, &target, batch);
                logger.record(&loss);
                online = Dqn::update(online, loss, lr, &mut opt);
            }

            if count % 1000 == 0 {
                target = online.clone();
            }
            count += 1;
            steps += 1;
            if tape.done() { break; }
        }
        *exploration.eps_mut() = (exploration.eps() * 0.999).max(0.05);
        
        if episode % 10 == 0 {
            let mean = logger.mean();
            let loss = mean.get("loss").unwrap_or(&0f32);
            let q_mean = mean.get("qmean").unwrap_or(&0f32);
            let td_error = mean.get("td_error").unwrap_or(&0f32);
            let eps = exploration.eps();
            println!("{episode} {count} {steps} {loss} {q_mean} {td_error} {eps}");
            if episode % 100 == 0 {
                eprintln!("Episode: {episode} Total steps: {count} Steps: {steps} Loss: {loss} Q-Mean: {q_mean} TD-error: {td_error} Eps: {eps}");
            }
            logger.clear();
        }
    }
}