use bake_deep::logger::MovingAvgLogger;
use bake_deep::scheduler::{LinearScheduler, Scheduler};
use bake_deep::{
    algorithm::{AdvantageEstimator, Ppo},
    buffer::RolloutBuffer,
    contract::ActorCritic,
    data::{Batchable, extras::{ Advantage, LogProb, Return } },
    distribution::{Categorical, Distribution},
    env::Tape,
    loss::Loss,
    net::basic::MlpSeparatedActorCriticNet,
    wrapper::ActorCriticWrapper
};
use bake_gym::env::KwArgs;
use bake_gym::env::GymLunarLander;
use burn::{nn::activation::ActivationConfig::Relu, optim::RmsPropConfig, prelude::*};
use rand::{SeedableRng, rngs::SmallRng, seq::SliceRandom};

pub fn main() {
    println!("count,reward_avg,step_avg,entropy,approx_KL,clip_fraction");
    let seed: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    let device = Device::default();
    device.seed(seed);
    let autodiff_device = device.clone().autodiff();

    let mut rng = SmallRng::seed_from_u64(seed);
    
    let state = Ppo { gamma: 0.99, eps: 0.2, advantage: AdvantageEstimator::Gae { lambda: 0.95 }, loss_fn: Loss::MseLoss };
    let env = GymLunarLander::new(seed, &device, KwArgs::new());
    let mut actor_critic: ActorCriticWrapper<_, Categorical> = ActorCriticWrapper::new(MlpSeparatedActorCriticNet::new(&[env.obs_shape()[1], 64, 64, env.n_actions()], Relu, &autodiff_device));

    let mut c_e = 0.02;
    let lr_a = 1e-4;
    let lr_c = 1e-3;
    let mut opt_a = RmsPropConfig::new().init();
    let mut opt_c = RmsPropConfig::new().init();

    let mut buffer = RolloutBuffer::new();
    let mut tape = Tape::new(env);

    let mut logger = MovingAvgLogger::new();
    logger.register("reward", 100);
    logger.register("step", 100);
    logger.register("actor_loss", 100);
    logger.register("critic_loss", 100);
    logger.register("entropy", 100);
    logger.register("approx_kl", 100);
    logger.register("clip_fraction", 100);

    let total_steps = 1000000;
    let mut c_e_sch = LinearScheduler::new(c_e as f64, 0.002, total_steps, 1.0);

    for count in 0..=total_steps {
        let dist = actor_critic.dist(tape.obs.clone(), tape.constraint.clone());
        let action = dist.sample();
        let mut t = tape.step(action.clone());
        t.insert::<LogProb>(dist.log_probs(action));

        buffer.push(t);

        if buffer.len() >= 2048 {
            let mut batch = buffer.pop();
            let (adv, ret) = state.advantage.advantage(&actor_critic, batch.clone(), state.gamma);
            let adv = (adv.clone() - adv.clone().mean()) / (adv.var(0) + 1e-9).sqrt();
            batch.insert::<Advantage>(adv); batch.insert::<Return>(ret);
            for _ in 0..4 {
                let mut perm: Vec<i64> = (0..batch.batch_size().unwrap() as i64).collect();
                perm.shuffle(&mut rng);
                for chunk in perm.chunks(128) {
                    let idx = Tensor::<1, Int>::from_data(TensorData::new(chunk.to_vec(), [chunk.len()]), &device);
                    let (net, loss) = Ppo::loss(&state, actor_critic, batch.clone().select(idx));
                    logger.push(&loss);
                    actor_critic = Ppo::update_separated(net, loss, c_e, lr_a, &mut opt_a, lr_c, &mut opt_c);
                }
            }
            
        }
        if tape.done() {
            logger.push_single("reward", tape.episode_reward);
            logger.push_single("step", tape.steps as f32);
            tape.reset();
        }
        
        if count % 5000 == 0 {
            let reward_avg = logger.emit("reward");
            let step_avg = logger.emit("step");
            let entropy = logger.emit("entropy");
            let approx_kl = logger.emit("approx_kl");
            let clip_fraction = logger.emit("clip_fraction");
            println!("{count},{reward_avg},{step_avg},{entropy},{approx_kl},{clip_fraction}");
        }

        c_e = c_e_sch.step() as f32;
    }

    // evaluation
    let env = GymLunarLander::new(seed, &device, KwArgs::new().add("render_mode", "human"));
    let mut tape = Tape::new(env);
    for _ in 0..5 {
        tape.reset();
        loop {
            let action = actor_critic.dist(tape.obs.clone(), tape.constraint.clone()).mode();
            tape.step(action);

            if tape.done() {
                break;
            }
        }
    }
}