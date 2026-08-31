use std::collections::VecDeque;

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

    let window = 20;
    let mut ep_rewards = VecDeque::with_capacity(window);
    let mut ep_reward = 0f32;
    let mut logger = Logger::default();

    for count in 0..=config.total_steps {
        let action = policy.action(tape.obs.clone(), tape.constraint.clone());
        let t = tape.step(&mut env, action);
        buffer.push(t);
        ep_reward += tape.reward;

        if tape.done() {
            let batch = buffer.pop();
            let loss = Vpg::loss(&config.vpg, &policy, batch);
            logger.record(&loss);
            policy = Vpg::update(policy, loss, config.coeff_entropy, config.lr, &mut opt);

            tape.reset(&mut env);
            if ep_rewards.len() >= window {
                ep_rewards.pop_front();
            }
            ep_rewards.push_back(ep_reward);
            ep_reward = 0f32;
        }

        if count % config.log_interval == 0 {
            let reward_avg = ep_rewards.iter().sum::<f32>() / ep_rewards.len() as f32;
            let mean = logger.mean();
            println!("count: {count}, reward_avg: {reward_avg}, entropy: {}, surrogate loss: {}", mean.get("entropy").unwrap_or(&0f32), mean.get("loss").unwrap_or(&0f32));
            logger.clear();
        }
        
    }
}