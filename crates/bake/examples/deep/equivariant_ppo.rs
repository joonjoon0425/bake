use std::collections::VecDeque;

use bake_deep::{algorithm::{Ppo, PpoExtra}, approximator::{ActorCritic, CategoricalActorCritic}, buffer::RolloutBuffer, config::{ActorCriticEncoderConfig, PpoConfig}, distribution::Distribution, env::CartPole, network::{ActorCriticNet, EncoderType::Separated, MlpActorCriticNet}, types::{Batchable, Logger, Tape}, utils::gae};
use burn::{Tensor, config::Config, module::Module, nn::activation::ActivationConfig::Relu, tensor::{Device, Int, TensorData}};
use rand::{SeedableRng, seq::SliceRandom};


pub fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "crates/bake/configs/deep/ppo_cartpole.json".to_string());
    let config = PpoConfig::load(&path).expect("failed to load config");

    let device = Device::default().autodiff();
    device.seed(config.seed);
    let mut env = CartPole::new(config.seed, &device);
    let mut actor_critic = CategoricalActorCritic::new(Z2SymNet::new(MlpActorCriticNet::new(&[4, 128, 2], Relu, &device)));
    let (lr_a, lr_c, mut opt_a, mut opt_c) = match &config.encoder_config {
        ActorCriticEncoderConfig::Separated { lr_actor, opt_actor_config, lr_critic, opt_critic_config } => {
            (*lr_actor, *lr_critic, opt_actor_config.init(), opt_critic_config.init())
        },
        _ => { panic!("Current code uses encoder-separated actor and critic") }
    };

    let mut buffer = RolloutBuffer::new();
    let mut tape = Tape::new(&mut env);

    let window = 20;
    let mut ep_rewards = VecDeque::with_capacity(window);
    let mut ep_reward = 0f32;
    let mut logger = Logger::default();
    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);

    for count in 0..=config.total_steps {
        let action = actor_critic.action(tape.obs.clone(), tape.constraint.clone());
        let dist = actor_critic.dist(tape.obs.clone(), tape.constraint.clone());
        let t = tape.step(&mut env, action.clone()).add_extra(dist.log_probs(action));

        buffer.push(t);
        ep_reward += tape.reward;

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
        if tape.done() {
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
            let entropy = mean.get("entropy").unwrap_or(&0f32);
            let approx_kl = mean.get("approx_kl").unwrap_or(&0f32);
            let clip_ratio = mean.get("clip_ratio").unwrap_or(&0f32);
            // println!("{i},{total_steps},{step},{entropy},{approx_kl},{clip_ratio}");
            eprintln!("count: {count}, reward_avg: {reward_avg}, entropy: {entropy}, approx KL: {approx_kl}, clip ratio: {clip_ratio}");
            
            logger.clear();
        }
    }
}


#[derive(Module, Debug)]
pub struct Z2SymNet<Ac: ActorCriticNet> {
    net: Ac,
}

impl<Ac: ActorCriticNet> Z2SymNet<Ac> {
    pub fn new(net: Ac) -> Self {
        Self {
            net
        }
    }
}

impl<Ac: ActorCriticNet<Obs = Tensor<2>, Params = Tensor<2>>> ActorCriticNet for Z2SymNet<Ac> {
    type Obs = Tensor<2>;
    type Params = Tensor<2>;

    fn params(&self, obs: Self::Obs) -> Self::Params {
        let logits_pos = self.net.params(obs.clone());
        let logits_neg = self.net.params(-obs);
        (logits_pos + logits_neg.flip([1])) * 0.5
    }

    fn values(&self, obs: Self::Obs) -> Tensor<1> {
        let v_pos = self.net.values(obs.clone());
        let v_neg = self.net.values(-obs);
        (v_pos + v_neg) * 0.5
    }

    fn encoder_type(&self) -> bake_deep::network::EncoderType {
        Separated
    }
}