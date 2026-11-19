use bake_rl::tabular::prelude::*;
use bake_rl::env::gym::GymnasiumEnvironmentConfig;
use bake_rl::core::scheduler::{LinearScheduler, Scheduler};
use bake_rl::env::wrapper::RecordEpisodeStatistics;

use rand::prelude::*;

pub fn main() {
    let total_step = 1000000;
    let seed = 12;
    let mut rng = SmallRng::seed_from_u64(seed);
    let env = GymnasiumEnvironmentConfig::frozen_lake().seed(seed).init();
    let env = RecordEpisodeStatistics::new(env, 100);
    let mut qtable = QTable::new(env.obs_space().n(), env.action_space().n());
    let mut tape = Tape::new(env);
    let state = OnPolicyMonteCarloState { gamma: 0.99, visit: MonteCarloVisit::EveryVisit };
    let mut stat = OnPolicyMonteCarloStat::new(&qtable);

    let mut exploration = EpsilonGreedy::new(1.0);
    let mut eps_sch = LinearScheduler::new(1.0, 0.05, total_step, 0.5);

    let mut buffer = EpisodeBuffer::new();

    for count in 0..total_step {
        let action = qtable.action(&mut rng, &mut exploration, tape.obs, tape.constraint);
        let (t, _) = tape.step(action);
        buffer.push(t);

        if buffer.is_ready() {
            let episode = buffer.pop();
            on_policy_monte_carlo_update(&state, &mut stat, &mut qtable, episode);
            tape.reset();
        }

        if count % 1000 == 0 {
            println!("count: {count}, ep/reward_mean: {}, ep/length_mean: {}, ep/time_mean: {}", tape.env().reward_mean(), tape.env().length_mean(), tape.env().time_mean());
        }

        *exploration.epsilon_mut() = eps_sch.step() as f32;
    }

    let env = GymnasiumEnvironmentConfig::frozen_lake().kwarg("render_mode", "human").init();
    let mut tape = Tape::new(env);

    for _ in 0..5 {
        loop {
            let action = qtable.action(&mut rng, &mut exploration, tape.obs, tape.constraint);
            tape.step(action);
            if tape.done() {
                tape.reset();
                break;
            }
        }
    }
}