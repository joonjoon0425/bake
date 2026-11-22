use bake_rl::env::space_transform::one_hot::BatchedOneHotTransform;
use bake_rl::env::wrapper::{BatchedObservationTransform, BatchedRecordEpisodeStatistics, SynchronizedBatchedEnvironment};
use bake_rl::deep::prelude::*;
use bake_rl::scheduler::{Scheduler, LinearScheduler};
use bake_rl::env::classic::masked_cliff_walking::MaskedCliffWalkingConfig;
use bake_rl::logger::{LogCollector, StdoutSink};

use std::time::Instant;

use burn::prelude::*;
use burn::nn::*;
use burn::optim::*;

pub fn main() {
    let s = Instant::now();

    let seed = 140;
    let device = Device::default();
    device.seed(seed);

    let mut collector = LogCollector::new();
    let mut stdout = StdoutSink::new("DQN PER - MaskedCliffWalking Log");

    let n_envs = 4;
    
    let env = Vec::from_fn(n_envs, |_| MaskedCliffWalkingConfig.init());

    let env = SynchronizedBatchedEnvironment::new(env, &device);
    let env = BatchedObservationTransform::new(env, BatchedOneHotTransform::new());
    let env = BatchedRecordEpisodeStatistics::new(env, 100);

    let state = DQNState { gamma: 0.99, loss_fn: LossFn::MseLoss };
    let online = seq!(
        LinearConfig::new(env.obs_space().shape()[1], 64).init(&device),
        Relu,
        LinearConfig::new(64, 64).init(&device),
        Relu,
        LinearConfig::new(64, env.action_space().n()).init(&device)
    );
    let mut online = QNetWrapper::new(online);
    let mut target = online.clone();
    let lr = 1e-3;
    let mut opt = AdamConfig::new().init();
    
    let mut tape = BatchedTape::new(env, true);

    let mut exploration = BatchedEpsilonGreedy::new(1.0f32);
    let mut buffer = ReplayBuffer::with_prioritized_sampler(0.6, 0.4, Some(1.0), 100000, n_envs, &device);

    let total_steps = 100000;
    let warmup = 10000;
    let update_freq = 10;
    let sync_freq = 500;
    let batch_size = 128;

    let eps_sch = LinearScheduler::new(1.0, 0.05, total_steps, 0.25);
    let beta_sch = LinearScheduler::new(0.4, 1.0, total_steps, 1.0);

    for count in 0..total_steps {
        // for action mask check, line 70
        // let constraint = tape.constraint.clone();
        let action = online.action(&mut exploration, tape.obs.clone(), tape.constraint.clone());
        let (batch, _) = tape.step(action.clone());

        buffer.push(batch);

        if count >= warmup && count % update_freq == 0 && let Some(sample) = buffer.sample(batch_size) {
            let (net, loss) = dqn_loss(&state, online, &target, sample.sample, sample.info);
            buffer.update_priorities(sample.indices, loss.td_error.clone());
            online = net.update(loss.clone(), lr, &mut opt);
            // log
            collector.record(2, "loss", loss.loss);
            collector.record(2, "q_mean", loss.q_mean);
            collector.record(2, "td_error_abs_mean", loss.td_error.abs().mean());
        }

        if count % sync_freq == 0 {
            target = online.clone();
        }
        
        if count % 5000 == 0 {
            // log
            collector.record(1, "ep/reward_mean", tape.env().reward_mean());
            collector.record(1, "ep/length_mean", tape.env().length_mean());
            collector.record(1, "ep/time_mean", tape.env().time_mean());
            collector.record(5, "elapsed time", s.elapsed().as_secs_f64());
            collector.dump_to(count, &mut stdout).expect("unexpected error while logging to stdout");    
        }

        *exploration.epsilon_mut() = eps_sch.value(count);
        *buffer.beta_mut() = beta_sch.value(count);
    }

    // // evaluation
    // // training in native cartpole, evaluation in gymnasium environment... it's like sim2sim
    // let env = GymnasiumEnvironmentConfig::cart_pole().kwarg("render_mode", "human").seed(10).init();
    // let env = SynchronizedBatchedEnvironment::new(vec![env], &device);

    // let mut tape = BatchedTape::new(env, false);
    // let mut eval = BatchedGreedy;
    // for _ in 0..5 {
    //     loop {
    //         let action = online.action(&mut eval, tape.obs.clone(), tape.constraint.clone());
    //         tape.step(action);
    //         if tape.done().clone().into_scalar() {
    //             tape.reset(None);
    //             break;
    //         }
    //     }
    // }
}