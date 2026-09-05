use bake::deep::algorithm::Reinforce;
use bake::deep::env::{CartPole, Tape};
use bake::deep::buffer::RolloutBuffer;
use bake::deep::net::basic::MlpPolicyNet;
use bake_deep::algorithm::reinforce::Baseline;
use bake_deep::contract::Policy;
use bake_deep::logger::MovingAvgLogger;
use bake_deep::wrapper::PolicyWrapper;

use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::nn::activation::ActivationConfig::Relu;

pub fn main() {
    println!("count,ep_reward_average,ep_step_average,loss,td_error,qmean,eps");
    let seed: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    let device = Device::default();
    device.seed(seed);
    let autodiff_device = device.clone().autodiff();

    let mut env = CartPole::new(seed, &device);
    let state = Reinforce{ gamma: 0.99, c_e: 0.2, baseline: Baseline::Mean };
    let mut policy = PolicyWrapper::new(MlpPolicyNet::new(&[4, 128, 2], Relu, &autodiff_device));
    let mut opt = AdamConfig::new().init();

    let mut buffer = RolloutBuffer::new();
    let mut tape = Tape::new(&mut env);

    let total_steps = 500000;
    let mut logger = MovingAvgLogger::new();
    logger.register("reward", 100);
    logger.register("step", 100);
    logger.register("surrogate_loss", 20);
    logger.register("entropy", 20);

    for count in 0..=total_steps {
        let action = policy.action(tape.obs.clone(), tape.constraint.clone());
        let t = tape.step(&mut env, action);
        buffer.push(t);

        if tape.done() {
            let rollout = buffer.pop();
            let loss = Reinforce::loss(&state, &policy, rollout);
            logger.push(&loss);
            policy = Reinforce::update(policy, loss, state.c_e, 1e-3, &mut opt);

            logger.push_single("reward", tape.episode_reward);
            logger.push_single("step", tape.steps as f32);
            tape.reset(&mut env);
        }

        if count % 5000 == 0 {
            let reward = logger.emit("reward");
            let step = logger.emit("step");
            let entropy = logger.emit("entropy");
            let surrogate_loss = logger.emit("surrogate_loss");
            println!("count: {count}, reward: {reward}, step: {step} entropy: {entropy}, surrogate loss: {surrogate_loss}");
        }
        
    }
}