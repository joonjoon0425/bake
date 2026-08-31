use std::collections::VecDeque;

use bake_deep::{algorithm::*, approximator::{ActorCritic, CategoricalActorCritic}, buffer::RolloutBuffer, config::{A2CConfig, ActorCriticEncoderConfig}, env::CartPole, exploration::NoiseReset, network::NoisyMlpActorCriticNet, types::{Logger, Tape}};
use burn::{config::Config, nn::activation::ActivationConfig::Relu, tensor::Device};


pub fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "crates/bake/configs/deep/noisy_a2c_cartpole.json".to_string());
    let config = A2CConfig::load(&path).expect("failed to load config");
    
    let device = Device::default().autodiff();
    device.seed(config.seed);
    let mut env = CartPole::new(config.seed, &device);
    let mut actor_critic = CategoricalActorCritic::new(NoisyMlpActorCriticNet::new(&[4, 128, 2], Relu, &device));

    let (lr_a, lr_c, mut opt_a, mut opt_c) = match &config.encoder_config {
        ActorCriticEncoderConfig::Separated { lr_actor, opt_actor_config, lr_critic, opt_critic_config } => {
            (*lr_actor, *lr_critic, opt_actor_config.init(), opt_critic_config.init())
        },
        _ => { panic!("Current code uses encoder-separated actor and critic") }
    };

    let mut buffer = RolloutBuffer::new();
    let mut tape = Tape::new(&mut env);

    let mut ep_rewards = VecDeque::with_capacity(20);
    let mut ep_reward = 0f32;
    let mut logger = Logger::default();

    for count in 0..=config.total_steps {
        let action = actor_critic.action(tape.obs.clone(), tape.constraint.clone());
        let t = tape.step(&mut env, action);
        buffer.push(t);
        ep_reward += tape.reward;

        if buffer.len() >= config.rollout_size {
            let batch = buffer.pop();
            let loss = A2C::loss(&config.a2c, &actor_critic, batch);
            logger.record(&loss);
            actor_critic = A2C::update_separated(actor_critic, loss, config.coeff_entropy, lr_a, &mut opt_a, lr_c, &mut opt_c);
            actor_critic.reset_noise();
        }

        if tape.done() {
            tape.reset(&mut env);
            if ep_rewards.len() >= 20 {
                ep_rewards.pop_front();
            }
            ep_rewards.push_back(ep_reward);
            ep_reward = 0f32;
        }

        if count % config.log_interval == 0 {
            let ep_reward_average = ep_rewards.iter().sum::<f32>() / ep_rewards.len() as f32;
            let mean = logger.mean();
            let loss = mean.get("critic_loss").unwrap_or(&0f32);
            let entropy = mean.get("entropy").unwrap_or(&0f32);
            println!("count: {count}, reward_avg: {ep_reward_average}, loss: {loss}, entropy: {entropy}");
            logger.clear();
        }
    }
}