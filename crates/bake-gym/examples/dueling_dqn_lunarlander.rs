use bake_deep::scheduler::{LinearScheduler, Scheduler};
use bake_deep::buffer::replay::ReplayBufferConfig;
use bake_deep::explore::{EpsGreedy, Exploration, Greedy};
use bake_deep::logger::MovingAvgLogger;
use bake_deep::net::basic::MlpDiscreteDuelingQNet;
use bake_deep::wrapper::DiscreteDuelingQNetWrapper;
use bake_gym::env::KwArgs;
use burn::optim::AdamConfig;
use burn::prelude::*;
use nn::activation::ActivationConfig::Relu;

use bake_deep::env::Tape;
use bake_deep::algorithm::Dqn;
use bake_deep::loss::Loss;

use bake_gym::env::lunarlander::GymLunarLander;

pub fn main() {
    println!("count,ep_reward_average,ep_step_average,loss,td_error,qmean,eps");
    let seed: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    let device = Device::default();
    device.seed(seed);
    let autodiff_device = device.clone().autodiff();
    let env = GymLunarLander::new(seed, &device, KwArgs::new());
    let config = Dqn{ gamma: 0.99, loss_fn: Loss::MseLoss };
    let mut online = DiscreteDuelingQNetWrapper::new(MlpDiscreteDuelingQNet::new(&[env.obs_shape()[1], 128, 84, env.n_actions()], Relu, &autodiff_device));
    let mut target = online.clone();
    let lr = 2.5e-4;
    let mut opt = AdamConfig::new().init();

    let mut exploration = EpsGreedy::new(seed, 1.0f32);
    let mut buffer = ReplayBufferConfig::prioritized(seed, 50000, 0.6, 0.4).with_priority_clip(1.0).init();
    let mut tape = Tape::new(env);
    let mut logger = MovingAvgLogger::new();

    let total_steps = 1000000;
    let warmup = 10000;
    let update_freq = 10;
    let sync_freq = 500;
    let batch_size = 512;

    let window = 100;
    logger.register("loss", 500);
    logger.register("mean_td_error", 500);
    logger.register("qmean", 500);
    logger.register("reward", window);
    logger.register("step", window);

    let mut eps_sch = LinearScheduler::new(1.0, 0.05, total_steps, 0.6);
    let mut beta_sch = LinearScheduler::new(0.4, 1.0, total_steps, 1.0);

    for count in 0..=total_steps {
        let action = exploration.sample(&online, tape.obs.clone(), tape.constraint.clone());
        let t = tape.step(action);
        buffer.push(t);

        if count >= warmup && count % update_freq == 0 && let Some((batch, batch_info)) = buffer.sample(batch_size) {
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
            tape.reset();
        }

        if count % 5000 == 0 {
            let reward = logger.emit("reward");
            let step = logger.emit("step");
            let loss = logger.emit("loss");
            let mean_td_error = logger.emit("mean_td_error");
            let qmean = logger.emit("qmean");
            println!("{count},{reward},{step},{loss},{mean_td_error},{qmean},{}", exploration.eps());
        }

        *exploration.eps_mut() = eps_sch.step() as f32;
        *buffer.beta_mut() = beta_sch.step();
    }

    // evaluation
    let env = GymLunarLander::new(seed, &device, KwArgs::new().add("render_mode", "human"));
    let mut tape = Tape::new(env);
    let mut greedy = Greedy;
    for _ in 0..5 {
        tape.reset();
        loop {
            let action = greedy.sample(&online, tape.obs.clone(), tape.constraint.clone());
            tape.step(action);

            if tape.done() {
                break;
            }
        }
    }
}