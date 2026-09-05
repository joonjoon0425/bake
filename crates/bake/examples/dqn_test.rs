use bake_deep::buffer::replay::ReplayBufferConfig;
use bake_deep::explore::{EpsGreedy, Exploration};
use bake_deep::logger::MovingAvgLogger;
use bake_deep::net::basic::MlpDiscreteQNet;
use bake_deep::wrapper::DiscreteQNetWrapper;
use burn::optim::AdamConfig;
use burn::prelude::*;
use nn::activation::ActivationConfig::Relu;

use bake::deep::env::{CartPole, Tape};
use bake::deep::algorithm::dqn::Dqn;
use bake_deep::algorithm::loss_enum::Loss;

pub fn main() {
    println!("count,ep_reward_average,ep_step_average,loss,td_error,qmean,eps");
    let seed: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    let device = Device::default();
    device.seed(seed);
    let autodiff_device = device.clone().autodiff();
    let mut env = CartPole::new(seed, &device);
    let config = Dqn{ gamma: 0.99, loss_fn: Loss::MseLoss };
    let mut online = DiscreteQNetWrapper::new(MlpDiscreteQNet::new(&[4, 128, 84, 2], Relu, &autodiff_device));
    let mut target = online.clone();
    let lr = 2.5e-4;
    let mut opt = AdamConfig::new().init();

    let mut exploration = EpsGreedy::new(seed, 1.0f32);
    let mut buffer = ReplayBufferConfig::prioritized(seed, 10000, 0.6, 0.4).with_priority_clip(1.0).with_max_priority_within_buffer(true).init();
    let mut tape = Tape::new(&mut env);
    let mut logger = MovingAvgLogger::new();

    let total_steps = 500000;
    let warmup = 10000;
    let update_freq = 10;
    let sync_freq = 500;
    let batch_size = 128;

    let mut ep_step = 0;
    let mut ep_reward = 0.;

    let window = 100;
    logger.register("loss", 500);
    logger.register("mean_td_error", 500);
    logger.register("qmean", 500);
    logger.register("reward", window);
    logger.register("step", window);

    let mut eps_sch = LinearScheduler::new(1.0, 0.05, total_steps / 2);
    let mut beta_sch = LinearScheduler::new(0.4, 1.0, total_steps);

    for count in 0..=total_steps {
        let action = exploration.sample(&online, tape.obs.clone(), tape.constraint.clone());
        let t = tape.step(&mut env, action);
        buffer.push(t);
        ep_step += 1;
        ep_reward += tape.reward;

        if count >= warmup && count % update_freq == 0 && let Some((batch, batch_info)) = buffer.sample(batch_size) {
            let loss = Dqn::loss(&config, &online, &target, batch, batch_info);
            logger.push(&loss);
            online = Dqn::update(online, loss, lr, &mut opt);
        }

        if count % sync_freq == 0 {
            let record = online.clone().into_record();
            target = target.load_record(record);
        }

        if tape.done() {
            tape.reset(&mut env);
            logger.push_single("reward", ep_reward);
            logger.push_single("step", ep_step as f32);
            ep_step = 0;
            ep_reward = 0.;
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

    
}


pub struct LinearScheduler {
    start: f64,
    steps: usize,

    cur_step: usize,
    slope: f64,
}

impl LinearScheduler {
    pub fn new(start: f64, end: f64, steps: usize) -> Self {
        Self {
            start,
            steps,
            cur_step: 0,
            slope: (end - start) / steps as f64
        }
    }

    pub fn step(&mut self) -> f64 {
        if self.cur_step < self.steps { self.cur_step += 1; }
        let ret = self.slope * self.cur_step as f64 + self.start;
        ret
    }

    pub fn reset(&mut self) {
        self.cur_step = 0;
    }
}