use std::collections::VecDeque;

use bake_common::LinearScheduler;
use bake_deep::{algorithm::{DoubleDqn, dqn::ValueLoss}, approximator::ConstrainedQNet, buffer::ReplayBuffer, env::CartPole, exploration::{EpsGreedy, Exploration}, network::MlpQNet, types::{Logger, Tape}};
use burn::{module::Module, nn::{activation::ActivationConfig::Relu}, optim::AdamConfig, tensor::Device};

pub fn main() {
    println!("count,ep_reward_average,loss,td_error,qmean,eps");
    let seed: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    let device = Device::default().autodiff();
    device.seed(seed);
    let mut env = CartPole::new(seed, &device);
    let config = DoubleDqn::new(0.99, ValueLoss::MseLoss);
    let mut online = ConstrainedQNet::new(MlpQNet::new(&[4, 128, 84, 2], Relu, &device));
    let mut target = online.clone();
    let lr = 2.5e-4;
    let mut opt = AdamConfig::new().init();

    let mut exploration = EpsGreedy::new(seed, 1.0f32);
    let mut buffer = ReplayBuffer::new(seed, 10000);
    let mut tape = Tape::new(&mut env);
    let mut logger = Logger::default();

    let total_steps = 500000;
    let warmup = 10000;
    let update_freq = 10;
    let sync_freq = 500;
    let batch_size = 128;

    let window = 20;
    let mut ep_rewards = VecDeque::with_capacity(window);
    let mut ep_reward = 0f32;

    let mut eps_sch = LinearScheduler::new(1.0, 0.05, total_steps / 4);

    for count in 0..=total_steps {
        let action = exploration.sample(&online, tape.obs.clone(), tape.constraint.clone());
        let t = tape.step(&mut env, action);
        buffer.push(t);
        ep_reward += tape.reward;

        if count >= warmup && count % update_freq == 0 && let Some(batch) = buffer.sample(batch_size) {
            let loss = DoubleDqn::loss(&config, &online, &target, batch);
            logger.record(&loss);
            online = DoubleDqn::update(online, loss, lr, &mut opt);
        }

        if count % sync_freq == 0 {
            let record = online.clone().into_record();
            target = target.load_record(record);
        }

        if tape.done() {
            tape.reset(&mut env);
            if ep_rewards.len() >= window {
                ep_rewards.pop_front();
            }
            ep_rewards.push_back(ep_reward);
            ep_reward = 0f32;
        }

        if count % 5000 == 0 {
            let ep_reward_average = ep_rewards.iter().sum::<f32>() / ep_rewards.len() as f32;
            let mean = logger.mean();
            let loss = mean.get("loss").unwrap_or(&0f32);
            let td_error = mean.get("td_error").unwrap_or(&0f32);
            let qmean = mean.get("qmean").unwrap_or(&0f32);
            println!("{count},{ep_reward_average},{loss},{td_error},{qmean},{}", exploration.eps());
            logger.clear();
        }

        *exploration.eps_mut() = eps_sch.step() as f32;
    }
}
