use bake_deep::{algorithm::{Baseline, Vpg}, approximator::{ComposedPolicy, Policy, encoder::MLPEncoder, head::CategoricalHead}, buffer::RolloutBuffer, env::CartPole, types::{Logger, Tape}};
use burn::{nn::activation::Activation, optim::AdamConfig, tensor::Device};


pub fn main() {
    let device = Device::default().autodiff();
    device.seed(1);
    let mut env = CartPole::new(12, &device);
    let config = Vpg::new(0.99, Baseline::Normalized);
    let mut policy = ComposedPolicy::new(
            MLPEncoder::new(vec![4, 128], Activation::Relu(burn::nn::Relu), &device),
            CategoricalHead::new(128, 2, &device)
        );
    let c_e = 0.00;
    let lr = 1e-3;
    let mut opt = AdamConfig::new().init();

    let mut buffer = RolloutBuffer::new();
    let mut tape = Tape::new(&mut env);
    let mut logger = Logger::default();

    for i in 0..=4000 {
        let mut step = 0;
        tape.reset(&mut env);
        loop {
            let action = policy.action(tape.obs.clone(), tape.constraint.clone());
            let t = tape.step(&mut env, action);
            buffer.push(t);

            step += 1;
            if tape.done() {
                let batch = buffer.pop();
                let loss = Vpg::loss(&config, &policy, batch);
                logger.record(&loss);
                policy = Vpg::update(policy, loss, c_e, lr, &mut opt);
                break;
            }
        }

        if i % 100 == 0 {
            let mean = logger.mean();
            println!("Episode: {i}, Steps: {step}, Entropy: {}, Surrogate loss: {}", mean.get("entropy").unwrap_or(&0f32), mean.get("loss").unwrap_or(&0f32));
        }
        logger.clear();
    }
}