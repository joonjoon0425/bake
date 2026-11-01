use bake_rl::core::env::wrapper::record_episode_statistics::RecordEpisodeStatistics;
use bake_rl::core::env::{Tape, VectorEnvironment, VectorEnvironmentConfig};
use bake_rl::core::env::wrapper::sync_env::SynchronizedVectorEnvironmentConfig;
use bake_rl::core::exploration::{EpsGreedy, Greedy};
use bake_rl::core::scheduler::{LinearScheduler, Scheduler};
use bake_rl::env::gym::GymnasiumEnvironmentConfig;
use burn::prelude::*;
use bake_rl::tabular::qtable::QTable;
use bake_rl::tabular::algorithms::q_learning::*;

pub fn main() {
    let seed= 12;
    let device = Device::default();
    device.seed(seed);
    let env = SynchronizedVectorEnvironmentConfig::new(vec![GymnasiumEnvironmentConfig::cliff_walking().seed(seed)]).init(&device);
    let env = RecordEpisodeStatistics::new(env);
    let state = QLearningState { gamma: 0.99, alpha: 0.4 };
    let mut q = QTable::new(env.obs_space().n(), env.action_space().n());
    let mut tape = Tape::new(env, true);

    let total_steps = 100000;
    let mut eg = EpsGreedy::new(1.0);
    let mut eps_sch = LinearScheduler::new(1.0, 0.05, total_steps, 0.75);

    for count in 0..total_steps {
        let action = q.action(&mut eg, tape.obs.clone(), tape.constraint.clone());
        let (t, info) = tape.step(action);

        qlearning_update(&state, &mut q, t.destruct().0);

        for stat in info.1 {
            println!("count: {count}, environment: {}, reward: {}, step: {}", stat.id, stat.r, stat.s);
        }
        *eg.eps_mut() = eps_sch.step() as f32;
    }

    let env = SynchronizedVectorEnvironmentConfig::new(vec![GymnasiumEnvironmentConfig::cliff_walking().kwarg("render_mode", "human")]).init(&device);
    let mut tape = Tape::new(env, false);
    let mut g = Greedy;
    for _ in 0..5 {
        loop {
            let action = q.action(&mut g, tape.obs.clone(), tape.constraint);
            tape.step(action);
            if tape.done().clone().into_scalar() {
                tape.reset(None);
                break;
            }
        }
    }
}