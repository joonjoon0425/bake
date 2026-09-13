use bake_deep::net::DiscreteQNet;
use bake_deep::scheduler::{LinearScheduler, Scheduler};
use bake_deep::buffer::replay::ReplayBufferConfig;
use bake_deep::explore::{EpsGreedy, Exploration};
use bake_deep::logger::MovingAvgLogger;
use bake_deep::wrapper::DiscreteQNetWrapper;
use burn::nn::{Linear, LinearConfig, Initializer::Zeros};
use burn::optim::AdamConfig;
use burn::prelude::*;

use bake_deep::env::{MaskedCliffWalking, Tape};
use bake_deep::algorithm::Dqn;
use bake_deep::loss::Loss;

pub fn main() {
    println!("count,reward_avg,step_avg,loss,td_error,qmean,eps");
    let seed: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    let device = Device::default();
    device.seed(seed);
    let autodiff_device = device.clone().autodiff();
    let env = MaskedCliffWalking::new(&device);
    let config = Dqn{ gamma: 0.99, loss_fn: Loss::HuberLoss { delta: 10. } };
    let mut online = DiscreteQNetWrapper::new(LinearQNet::new(env.n_obs(), env.n_actions(), &autodiff_device));
    let mut target = online.clone();
    let lr = 2.5e-4;
    let mut opt = AdamConfig::new().init();

    let mut exploration = EpsGreedy::new(seed, 1.0f32);
    let mut buffer = ReplayBufferConfig::uniform(seed, 50000).init();
    let mut tape = Tape::new(env);
    let mut logger = MovingAvgLogger::new();

    let total_steps = 1000000;
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

    let mut eps_sch = LinearScheduler::new(1.0, 0.05, total_steps, 0.25);

    for count in 0..=total_steps {
        let action = exploration.sample(&online, tape.obs.clone(), tape.constraint.clone());
        let t = tape.step(action);
        buffer.push(t);

        if count >= warmup && count % update_freq == 0 && let Some((batch, batch_info)) = buffer.sample(batch_size) {
            let (net, loss) = Dqn::loss(&config, online, &target, batch, batch_info.clone());
            logger.push(&loss);
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
    }
}

#[derive(Module, Debug)]
struct LinearQNet {
    linear: Linear,
}

impl LinearQNet {
    pub fn new(d_input: usize, d_output: usize, device: &Device) -> Self {
        Self {
            linear: LinearConfig::new(d_input, d_output).with_initializer(Zeros).init(device)
        }
    }
}

impl DiscreteQNet for LinearQNet {
    type Obs = Tensor<2>;
    
    fn forward(&self, obs: Self::Obs) -> Tensor<2> {
        self.linear.forward(obs)
    }
}