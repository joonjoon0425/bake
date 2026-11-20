use bake_rl::tabular::prelude::*;
use bake_rl::env::gym::GymnasiumEnvironmentConfig;
use bake_rl::core::scheduler::{LinearScheduler, Scheduler};
use bake_rl::env::wrapper::RecordEpisodeStatistics;
use bake_rl::logger::{LogCollector, StdoutSink};

use rand::prelude::*;

pub fn main() {
    let total_step = 1000000;
    let seed = 12;
    let mut rng = SmallRng::seed_from_u64(seed);

    let mut collector = LogCollector::new();
    let mut stdout = StdoutSink::new("off-policy Monte Carlo - Taxi Log");

    let env = GymnasiumEnvironmentConfig::taxi().seed(seed).init();
    let env = RecordEpisodeStatistics::new(env, 100);
    
    let mut qtable = QTable::new(env.obs_space().n(), env.action_space().n());
    let mut tape = Tape::new(env);
    let state = OffPolicyMonteCarloState { gamma: 0.99 };
    let mut stat = OffPolicyMonteCarloStat::new(&qtable);

    let mut exploration = EpsilonGreedy::new(1.0);
    let mut eps_sch = LinearScheduler::new(1.0, 0.05, total_step, 0.5);

    let mut buffer = EpisodeBuffer::new();

    for count in 0..total_step {
        // log
        collector.record(0, "count", count);
        collector.record(1, "ep/reward_mean", tape.env().reward_mean());
        collector.record(1, "ep/length_mean", tape.env().length_mean());
        collector.record(1, "ep/time_mean", tape.env().time_mean());

        let action = qtable.action(&mut rng, &mut exploration, tape.obs, tape.constraint);
        let action_prob = qtable.prob(&mut exploration, tape.obs, tape.constraint, action);
        let (mut t, _) = tape.step(action);
        t.dict.insert_str("action_prob", action_prob);
        buffer.push(t);

        if buffer.is_ready() {
            let episode = buffer.pop();
            off_policy_monte_carlo_update(&state, &mut stat, &mut qtable, episode);
            tape.reset();
        }

        if count % 1000 == 0 {
            collector.dump_to(&mut stdout).expect("unexpected error");
        }

        *exploration.epsilon_mut() = eps_sch.step() as f32;
    }

    let env = GymnasiumEnvironmentConfig::taxi().kwarg("render_mode", "human").init();
    let mut tape = Tape::new(env);
    let mut greedy = Greedy;
    for _ in 0..5 {
        loop {
            let action = qtable.action(&mut rng, &mut greedy, tape.obs, tape.constraint);
            tape.step(action);
            if tape.done() {
                tape.reset();
                break;
            }
        }
    }
}