use bake_deep::{algorithm::Vpg, approximator::{CategoricalPolicy, Policy}, buffer::RolloutBuffer, config::VpgConfig, env::CartPole, network::MlpPolicyNet, types::{Logger, Tape}};
use burn::{config::Config, nn::activation::ActivationConfig::Relu, tensor::Device};

pub fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "crates/bake/configs/deep/vpg_cartpole.json".to_string());
    let config = VpgConfig::load(path).expect("Failed to read config");

    let device = Device::default().autodiff();
    device.seed(config.seed);
    let mut env = CartPole::new(config.seed, &device);
    let mut policy = CategoricalPolicy::new(MlpPolicyNet::new(&[4, 128, 2], Relu, &device));
    let mut opt = config.opt_config.init();

    let mut buffer = RolloutBuffer::new();
    let mut tape = Tape::new(&mut env);
    let mut logger = Logger::default();

    for i in 0..=config.total_episode {
        let mut step = 0;
        tape.reset(&mut env);
        loop {
            let action = policy.action(tape.obs.clone(), tape.constraint.clone());
            let t = tape.step(&mut env, action);
            buffer.push(t);

            step += 1;
            if tape.done() {
                let batch = buffer.pop();
                let loss = Vpg::loss(&config.vpg, &policy, batch);
                logger.record(&loss);
                policy = Vpg::update(policy, loss, config.coeff_entropy, config.lr, &mut opt);
                break;
            }
        }

        if i % config.log_interval == 0 {
            let mean = logger.mean();
            println!("Episode: {i}, Steps: {step}, Entropy: {}, Surrogate loss: {}", mean.get("entropy").unwrap_or(&0f32), mean.get("loss").unwrap_or(&0f32));
        }
        logger.clear();
    }
}