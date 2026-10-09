use bake::rl::deep::buffer::replay::ReplayBufferConfig;
use bake::rl::deep::explore::{Greedy, Exploration};
use bake::rl::deep::logger::MovingAvgLogger;
use bake::rl::deep::net::basic::NoisyMlpDiscreteQNet;
use bake::rl::deep::net::layer::NoiseReset;
use bake::rl::deep::scheduler::{LinearScheduler, Scheduler};

use bake::rl::deep::algorithm::Dqn;
use bake::rl::deep::loss::LossFn;
use bake::rl::env::vectorized::{Tape, sync_env::SynchronizedEnvironment};
use bake::rl::env::collection::{CartPole, CartPoleConfig};

use burn::optim::AdamConfig;
use burn::prelude::*;
use nn::activation::ActivationConfig::Relu;

pub fn main() {
    println!("count,reward_avg,step_avg,loss,td_error,qmean");
    let seed: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    let device = Device::default();
    device.seed(seed);
    let env = SynchronizedEnvironment::<CartPole>::new(vec![CartPoleConfig::new()], &device);
    let config = Dqn{ gamma: 0.99, loss_fn: LossFn::MseLoss };
    let mut online = NoisyMlpDiscreteQNet::new(&[4, 128, 84, 2], Relu, &device);
    let mut target = online.clone();
    let lr = 2.5e-4;
    let mut opt = AdamConfig::new().init();

    let mut exploration = Greedy;
    let mut buffer = ReplayBufferConfig::prioritized(seed, 50000, 0.6, 0.4).with_priority_clip(1.0).init();
    let mut tape = Tape::new(env);
    let mut logger = MovingAvgLogger::new();

    let total_steps = 500000;
    let warmup = 10000;
    let update_freq = 10;
    let sync_freq = 500;
    let batch_size = 128;

    let window = 100;
    logger.register("reward", window);
    logger.register("step", window);

    let mut beta_sch = LinearScheduler::new(0.4, 1.0, total_steps, 1.0);

    for count in 0..=total_steps {
        online.reset_noise();
        let action = exploration.sample(&online, tape.obss.clone(), tape.constraints.clone());
        let t = tape.step(action);
        buffer.push(t);

        if count >= warmup && count % update_freq == 0 && let Some((batch, batch_info)) = buffer.sample(batch_size) {
            online.reset_noise();
            target.reset_noise();
            let (net, loss) = Dqn::loss(&config, online, &target, batch, batch_info.clone());
            logger.push(&loss, window.into());
            buffer.update_priority(&batch_info.indices, loss.td_error.clone());
            online = net.update(loss, lr, &mut opt);
        }

        if count % sync_freq == 0 {
            target = online.clone();
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
            let loss = logger.emit("loss");
            let mean_td_error = logger.emit("mean_td_error");
            let qmean = logger.emit("qmean");
            println!("{count},{reward},{step},{loss},{mean_td_error},{qmean}");
        }

        *buffer.beta_mut() = beta_sch.step();
    }
}
