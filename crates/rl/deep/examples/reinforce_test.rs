use bake::rl::deep::algorithm::Reinforce;
use bake::rl::deep::contract::basic::Policy;
use bake::rl::deep::buffer::RolloutBuffer;
use bake::rl::deep::net::basic::MlpPolicy;
use bake::rl::deep::algorithm::reinforce::Baseline;
use bake::rl::deep::logger::MovingAvgLogger;

use bake::rl::env::vectorized::{Tape, sync_env::SynchronizedEnvironment};
use bake::rl::env::collection::{CartPole, CartPoleConfig};

use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::nn::activation::ActivationConfig::Relu;

pub fn main() {
    println!("count,reward_avg,step_avg,entropy, surrogate_loss");
    let seed: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    let device = Device::default();
    device.seed(seed);

    let env = SynchronizedEnvironment::<CartPole>::new(vec![CartPoleConfig::new()], &device);
    let state = Reinforce{ gamma: 0.99, c_e: 0.02, baseline: Baseline::Normalized };
    let mut policy = MlpPolicy::new(&[4, 128, 2], Relu, &device);
    let mut opt = AdamConfig::new().init();

    let mut buffer = RolloutBuffer::new();
    let mut tape = Tape::new(env);

    let total_steps = 500000;
    let mut logger = MovingAvgLogger::new();
    logger.register("reward", 100);
    logger.register("step", 100);
    for name in Reinforce::log_names() {
        logger.register(name, 100);
    }

    for count in 0..=total_steps {
        let action = policy.action(tape.obss.clone(), tape.constraints.clone());
        let t = tape.step(action);
        buffer.push(t);

        if tape.done().into_scalar() {
            let rollout = buffer.pop();
            let (net, loss) = Reinforce::loss(&state, policy, rollout);
            logger.push(&loss, 100.into());
            policy = net.update(loss, 1e-3, &mut opt);
        }

        // logging episodic rewards and steps
        let (r, s) = tape.finished_reward_steps();
        for (reward, step) in r.iter().zip(s.iter()) {
            logger.push_single("reward", *reward, None);
            logger.push_single("step", *step, None);
        }

        if count % 5000 == 0 {
            let reward = logger.emit("reward");
            let step = logger.emit("step");
            let entropy = logger.emit("entropy");
            let policy_loss = logger.emit("policy_loss");
            println!("count: {count}, reward: {reward}, step: {step} entropy: {entropy}, policy loss: {policy_loss}");
        }
        
    }
}