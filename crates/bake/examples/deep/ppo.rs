use bake_deep::{algorithm::{Ppo, PpoExtra, dqn::ValueLoss}, approximator::{ActorCritic, SeparatedActorCritic, encoder::MLPEncoder, head::{CategoricalHead, LinearVHead}}, buffer::RolloutBuffer, distribution::Distribution, env::CartPole, types::{Batchable, Logger, Tape}, utils::gae};
use burn::{Tensor, nn::activation::Activation, optim::RmsPropConfig, tensor::{Device, Int, TensorData}};
use rand::{SeedableRng, seq::SliceRandom};


pub fn main() {
    let seed: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    let device = Device::default().autodiff();
    device.seed(seed);
    let mut env = CartPole::new(seed, &device);
    let config = Ppo::new(0.99, 0.98, 0.2, ValueLoss::MseLoss);
    let mut actor_critic = SeparatedActorCritic::new(
            MLPEncoder::new(vec![4, 128], Activation::Relu(burn::nn::Relu), &device),
            MLPEncoder::new(vec![4, 128], Activation::Relu(burn::nn::Relu), &device),
            CategoricalHead::new(128, 2, &device),
            LinearVHead::new(128, 1, &device)
        );
    let lr_a = 1e-4;
    let lr_c = 1e-3;
    let c_e = 0.02;
    let mut opt_a = RmsPropConfig::new().init();
    let mut opt_c = RmsPropConfig::new().init();

    const BATCH_SIZE: usize = 512;
    const MINIBATCH_SIZE: usize = 64;
    const EPOCH: usize = 4;

    let mut buffer = RolloutBuffer::new();
    let mut tape = Tape::new(&mut env);
    let mut total_steps = 0;
    let mut logger = Logger::default();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    for i in 0..=4000 {    
        let mut step = 0;
        tape.reset(&mut env);
        loop {
            let action = actor_critic.action(tape.obs.clone(), tape.constraint.clone());
            let dist = actor_critic.actor(tape.obs.clone(), tape.constraint.clone());
            let t = tape.step(&mut env, action.clone()).add_extra(dist.log_probs(action));

            buffer.push(t);

            step += 1;
            total_steps += 1;
            if buffer.len() >= BATCH_SIZE {
                let mut batch = buffer.pop();
                let (adv, ret) = gae(&actor_critic, batch.clone(), config.gamma, config.lambda);
                let gae = (adv.clone() - adv.clone().mean()) / (adv.var(0) + 1e-9).sqrt();
                let old_log_probs = std::mem::replace(&mut batch.extras, Tensor::zeros([1], &device));
                let batch = batch.add_extra(PpoExtra { gae, ret, old_log_probs });
                for _ in 0..EPOCH {
                    let mut perm: Vec<i64> = (0..batch.batch_size() as i64).collect();
                    perm.shuffle(&mut rng);
                    for chunk in perm.chunks(MINIBATCH_SIZE) {
                        let idx = Tensor::<1, Int>::from_data(TensorData::new(chunk.to_vec(), [chunk.len()]), &device);
                        let loss = Ppo::loss(&config, &actor_critic, batch.clone().select(idx));
                        logger.record(&loss);
                        actor_critic = Ppo::update_separated(actor_critic, loss, c_e, lr_a, &mut opt_a, lr_c, &mut opt_c)
                    }
                }
                
            }
            if tape.done() { break; }
        }

        if i % 10 == 0 && i != 0 {
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