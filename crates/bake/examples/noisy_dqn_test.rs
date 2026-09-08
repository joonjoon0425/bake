use bake_deep::buffer::replay::ReplayBufferConfig;
use bake_deep::explore::{Greedy, Exploration};
use bake::logger::MovingAvgLogger;
use bake_deep::net::basic::NoisyMlpDiscreteQNet;
use bake_deep::net::layer::NoiseReset;
use bake_deep::wrapper::DiscreteQNetWrapper;
use burn::optim::AdamConfig;
use burn::prelude::*;
use nn::activation::ActivationConfig::Relu;

use bake::deep::env::{CartPole, Tape};
use bake::deep::algorithm::Dqn;
use bake_deep::loss::Loss;

use bake::scheduler::{LinearScheduler, Scheduler};

pub fn main() {
    println!("count,ep_reward_average,ep_step_average,loss,td_error,qmean");
    let seed: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    let device = Device::default();
    device.seed(seed);
    let autodiff_device = device.clone().autodiff();
    let mut env = CartPole::new(seed, &device);
    let config = Dqn{ gamma: 0.99, loss_fn: Loss::MseLoss };
    let mut online = DiscreteQNetWrapper::new(NoisyMlpDiscreteQNet::new(&[4, 128, 84, 2], Relu, &autodiff_device));
    let mut target = online.clone();
    let lr = 2.5e-4;
    let mut opt = AdamConfig::new().init();

    let mut exploration = Greedy;
    let mut buffer = ReplayBufferConfig::prioritized(seed, 50000, 0.6, 0.4).with_priority_clip(1.0).init();
    let mut tape = Tape::new(&mut env);
    let mut logger = MovingAvgLogger::new();

    let total_steps = 500000;
    let warmup = 10000;
    let update_freq = 10;
    let sync_freq = 500;
    let batch_size = 128;

    let window = 100;
    logger.register("loss", 500);
    logger.register("mean_td_error", 500);
    logger.register("qmean", 500);
    logger.register("reward", window);
    logger.register("step", window);

    let mut beta_sch = LinearScheduler::new(0.4, 1.0, total_steps, 1.0);

    for count in 0..=total_steps {
        online.reset_noise();
        let action = exploration.sample(&online, tape.obs.clone(), tape.constraint.clone());
        let t = tape.step(&mut env, action);
        buffer.push(t);

        if count >= warmup && count % update_freq == 0 && let Some((batch, batch_info)) = buffer.sample(batch_size) {
            online.reset_noise();
            target.reset_noise();
            let (net, loss) = Dqn::loss(&config, online, &target, batch, batch_info.clone());
            logger.push(&loss);
            buffer.update_priority(&batch_info.indices, loss.td_error.clone());
            online = Dqn::update(net, loss, lr, &mut opt);
        }

        if count % sync_freq == 0 {
            let record = online.clone().into_record();
            target = target.load_record(record);
        }

        if tape.done() {
            logger.push_single("reward", tape.episode_reward);
            logger.push_single("step", tape.steps as f32);
            tape.reset(&mut env);
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
