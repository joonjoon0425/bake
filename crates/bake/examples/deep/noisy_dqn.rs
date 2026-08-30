use bake_deep::{algorithm::{Dqn, dqn::ValueLoss}, approximator::ConstrainedQNet, buffer::ReplayBuffer, env::CartPole, exploration::{Exploration, Greedy, NoiseReset}, network::NoisyMlpQNet, types::{Logger, Tape}};
use burn::{module::Module, nn::activation::ActivationConfig::Relu, optim::AdamConfig, tensor::Device};
pub fn main() {
    let seed: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    let device = Device::default().autodiff();
    device.seed(seed);
    let mut env = CartPole::new(seed, &device);
    let config = Dqn::new(0.99, ValueLoss::HuberLoss { delta: 10.0f32 });
    let mut online = ConstrainedQNet::new(NoisyMlpQNet::new(&[4, 128, 2], Relu, &device));
    let mut target = online.clone();
    let lr = 1e-3;
    let mut opt = AdamConfig::new().init();

    let mut exploration = Greedy;
    let mut buffer = ReplayBuffer::new(12, 10000);
    let mut tape = Tape::new(&mut env);
    let mut count = 0;
    let mut logger = Logger::default();
    for episode in 0..=4000 {
        let mut steps = 0;
        tape.reset(&mut env);
        loop {
            online.reset_noise();
            let action = exploration.sample(&online, tape.obs.clone(), tape.constraint.clone());
            let t = tape.step(&mut env, action);
            buffer.push(t);

            if count % 4 == 0 && let Some(batch) = buffer.sample(64) {
                online.reset_noise();
                target.reset_noise();
                let loss = Dqn::loss(&config, &online, &target, batch);
                logger.record(&loss);
                online = Dqn::update(online, loss, lr, &mut opt);
            }

            if count % 1000 == 0 {
                let record = online.clone().into_record();
                target = target.load_record(record);
            }
            count += 1;
            steps += 1;
            if tape.done() { break; }
        }
        if episode % 10 == 0 {
            let mean = logger.mean();
            let loss = mean.get("loss").unwrap_or(&0f32);
            let q_mean = mean.get("qmean").unwrap_or(&0f32);
            let td_error = mean.get("td_error").unwrap_or(&0f32);
            println!("{episode} {count} {steps} {loss} {q_mean} {td_error}");
            if episode % 100 == 0 {
                eprintln!("Episode: {episode} Total steps: {count} Steps: {steps} Loss: {loss} Q-Mean: {q_mean} TD-error: {td_error}");
            }
            logger.clear();
        }
    }
}