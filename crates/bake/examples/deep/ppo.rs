use bake_deep::{algorithm::{Ppo, PpoExtra}, approximator::{ActorCritic, CategoricalActorCritic}, buffer::RolloutBuffer, config::{ActorCriticEncoderConfig, PpoConfig}, distribution::Distribution, env::CartPole, network::MlpActorCriticNet, types::{Batchable, Logger, Tape}, utils::gae};
use burn::{Tensor, config::Config, nn::activation::ActivationConfig::Relu, tensor::{Device, Int, TensorData}};
use rand::{SeedableRng, seq::SliceRandom};


pub fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "crates/bake/configs/deep/ppo_cartpole.json".to_string());
    let config = PpoConfig::load(&path).expect("failed to load config");

    let device = Device::default().autodiff();
    device.seed(config.seed);
    let mut env = CartPole::new(config.seed, &device);
    let mut actor_critic = CategoricalActorCritic::new(MlpActorCriticNet::new(&[4, 128, 2], Relu, &device));
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
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);

    for i in 0..=config.total_episode {    
        let mut step = 0;
        tape.reset(&mut env);
        loop {
            let action = actor_critic.action(tape.obs.clone(), tape.constraint.clone());
            let dist = actor_critic.dist(tape.obs.clone(), tape.constraint.clone());
            let t = tape.step(&mut env, action.clone()).add_extra(dist.log_probs(action));

            buffer.push(t);

            step += 1;
            total_steps += 1;
            if buffer.len() >= config.rollout_size {
                let mut batch = buffer.pop();
                let (adv, ret) = gae(&actor_critic, batch.clone(), config.ppo.gamma, config.ppo.lambda);
                let gae = (adv.clone() - adv.clone().mean()) / (adv.var(0) + 1e-9).sqrt();
                let old_log_probs = std::mem::replace(&mut batch.extras, Tensor::zeros([1], &device));
                let batch = batch.add_extra(PpoExtra { gae, ret, old_log_probs });
                for _ in 0..config.epoch {
                    let mut perm: Vec<i64> = (0..batch.batch_size() as i64).collect();
                    perm.shuffle(&mut rng);
                    for chunk in perm.chunks(config.minibatch_size) {
                        let idx = Tensor::<1, Int>::from_data(TensorData::new(chunk.to_vec(), [chunk.len()]), &device);
                        let loss = Ppo::loss(&config.ppo, &actor_critic, batch.clone().select(idx));
                        logger.record(&loss);
                        actor_critic = Ppo::update_separated(actor_critic, loss, config.coeff_entropy, lr_a, &mut opt_a, lr_c, &mut opt_c)
                    }
                }
                
            }
            if tape.done() { break; }
        }

        if i % config.log_interval == 0 && i != 0 {
            let mean = logger.mean();
            let entropy = mean.get("entropy").unwrap_or(&0f32);
            let approx_kl = mean.get("approx_kl").unwrap_or(&0f32);
            let clip_ratio = mean.get("clip_ratio").unwrap_or(&0f32);
            println!("{i},{total_steps},{step},{entropy},{approx_kl},{clip_ratio}");
            if i % 100 == 0 {
                eprintln!("Episode: {i}, Total steps: {total_steps} Steps: {step}, Entropy: {entropy}, Approx KL: {approx_kl} Clip ratio: {clip_ratio}");
            }
            logger.clear();
        }
    }
}