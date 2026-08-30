use bake_deep::{algorithm::*, approximator::{ActorCritic, CategoricalActorCritic}, buffer::RolloutBuffer, config::{A2CConfig, ActorCriticEncoderConfig}, env::CartPole, exploration::NoiseReset, network::{MlpActorCriticNet, NoisyMlpActorCriticNet}, types::{Logger, Tape}};
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
    let mut total_steps = 0;
    let mut logger = Logger::default();

    for i in 0..= config.total_episode {
        let mut step = 0;
        tape.reset(&mut env);
        loop {
            let action = actor_critic.action(tape.obs.clone(), tape.constraint.clone());
            let t = tape.step(&mut env, action);
            buffer.push(t);

            step += 1;
            total_steps += 1;
            if buffer.len() >= config.rollout_size {
                let batch = buffer.pop();
                let loss = A2C::loss(&config.a2c, &actor_critic, batch);
                logger.record(&loss);
                actor_critic = A2C::update_separated(actor_critic, loss, config.coeff_entropy, lr_a, &mut opt_a, lr_c, &mut opt_c);
                actor_critic.reset_noise();
            }
            if tape.done() { break; }
        }

        if i % config.log_interval == 0 {
            let mean = logger.mean();
            let loss = mean.get("critic_loss").unwrap_or(&0f32);
            let entropy = mean.get("entropy").unwrap_or(&0f32);
            println!("{i},{total_steps},{step},{entropy},{loss}");
            if i % 100 == 0 {
                eprintln!("Episode: {i}, Total steps: {total_steps} Steps: {step}, Entropy: {entropy}, Loss: {loss}");
            }
            logger.clear()
        }
    }
}