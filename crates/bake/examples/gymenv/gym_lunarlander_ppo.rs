use std::collections::VecDeque;

use bake_common::LinearScheduler;
use bake_deep::{algorithm::{Ppo, PpoExtra}, approximator::{ActorCritic, CategoricalActorCritic}, buffer::RolloutBuffer, config::{ActorCriticEncoderConfig, PpoConfig}, distribution::Distribution, env::*, network::MlpActorCriticNet, types::{Batchable, Logger, Tape}, utils::gae};
use burn::{Tensor, config::Config, nn::activation::ActivationConfig::*, tensor::{Device, Int, TensorData}};
use rand::{SeedableRng, seq::SliceRandom};


pub fn main() {
    println!("count,ep_reward_average,ep_step_average");
    let seed: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    let path = std::env::args().nth(2).unwrap_or_else(|| "crates/bake/configs/gymenv/ppo_lunarlander.json".to_string());
    let config = PpoConfig::load(&path).expect("failed to load config");

    let device = Device::default().autodiff();
    device.seed(seed);
    let mut env = GymnasiumEnv::<LunarLanderInfo>::new(seed, &device, false);
    let mut actor_critic = CategoricalActorCritic::new(MlpActorCriticNet::new(&[env.obs_dim(), 64, 64, env.n_actions()], Tanh, &device));
    let (lr_a, lr_c, mut opt_a, mut opt_c) = match &config.encoder_config {
        ActorCriticEncoderConfig::Separated { lr_actor, opt_actor_config, lr_critic, opt_critic_config } => {
            (*lr_actor, *lr_critic, opt_actor_config.init(), opt_critic_config.init())
        },
        _ => { panic!("Current code uses encoder-separated actor and critic") }
    };

    let mut buffer = RolloutBuffer::new();
    let mut tape = Tape::new(&mut env);

    let window = 100;
    let mut ep_rewards = VecDeque::with_capacity(window);
    let mut ep_reward = 0f32;
    let mut ep_steps = VecDeque::with_capacity(window);
    let mut ep_step = 0usize;
    let mut logger = Logger::default();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut c_e = config.coeff_entropy;
    let mut c_e_sch = LinearScheduler::new(c_e as f64, 0.008, (config.total_steps as f32 * (9. / 16.)) as usize);

    for count in 0..=config.total_steps {
        let action = actor_critic.action(tape.obs.clone(), tape.constraint.clone());
        let dist = actor_critic.dist(tape.obs.clone(), tape.constraint.clone());
        let t = tape.step(&mut env, action.clone()).add_extra(dist.log_probs(action));

        buffer.push(t);
        ep_reward += tape.reward;
        ep_step += 1;

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
                    actor_critic = Ppo::update_separated(actor_critic, loss, c_e, lr_a, &mut opt_a, lr_c, &mut opt_c)
                }
            }
            
        }

        if tape.done() {
            tape.reset(&mut env);
            if ep_rewards.len() >= window {
                ep_rewards.pop_front();
            }
            ep_rewards.push_back(ep_reward);
            ep_reward = 0f32;
            ep_steps.push_back(ep_step);
            ep_step = 0usize;
        }

        if count % config.log_interval == 0 {
            let ep_reward_avgerage = if ep_rewards.len() != 0 { ep_rewards.iter().sum::<f32>() / ep_rewards.len() as f32 } else { 0f32 };
            let ep_step_average = if ep_steps.len() != 0 { ep_steps.iter().sum::<usize>() / ep_steps.len() } else { 0usize };
            let mean = logger.mean();
            let actor_loss = mean.get("actor_loss").unwrap_or(&0f32);
            let critic_loss = mean.get("critic_loss").unwrap_or(&0f32);
            let entropy = mean.get("entropy").unwrap_or(&0f32);
            let approx_kl = mean.get("approx_kl").unwrap_or(&0f32);
            let clip_ratio = mean.get("clip_ratio").unwrap_or(&0f32);
            println!("{count},{ep_reward_avgerage},{ep_step_average}");
            eprintln!("count: {count}, reward_avg: {ep_reward_avgerage}, step_average: {ep_step_average}, actor_loss: {actor_loss}, critic_loss: {critic_loss}, entropy: {entropy}, approx KL: {approx_kl}, clip ratio: {clip_ratio}, c_e: {c_e}");
            
            logger.clear();
        }

        c_e = c_e_sch.step() as f32;
    }

    // Evaluation
    let mut env = GymnasiumEnv::<LunarLanderInfo>::new(seed, &device, true);
    for _ in 0..5 {
        let (mut obs, mut constraint) = env.reset();
        loop {
            let action = actor_critic.dist(obs, constraint).mode();
            let ((next_obs, next_constraint), _, terminated, truncated) = env.step(action);
            obs = next_obs;
            constraint = next_constraint;

            if terminated || truncated {
                break;
            }
        }
    }

}