use bake_rl::tabular::prelude::*;
use bake_rl::env::gym::GymnasiumEnvironmentConfig;
use bake_rl::core::scheduler::{LinearScheduler, Scheduler};
use bake_rl::env::wrapper::RecordEpisodeStatistics;
use bake_rl::logger::{LogCollector, StdoutSink};

use rand::prelude::*;

pub fn main() {
    let total_step = 100000;
    let seed = 12;
    let mut rng = SmallRng::seed_from_u64(seed);

    let mut collector = LogCollector::new();
    let mut stdout = StdoutSink::new("n-step SARSA - Taxi Log");

    let env = GymnasiumEnvironmentConfig::taxi().seed(seed).init();
    let env = RecordEpisodeStatistics::new(env, 100);
    
    let mut qtable = QTable::new(env.obs_space().n(), env.action_space().n());
    let mut tape = Tape::new(env);
    let state = NStepSARSAState { gamma: 0.99, alpha: 0.15 };
    
    let mut exploration = EpsilonGreedy::new(1.0);
    let mut eps_sch = LinearScheduler::new(1.0, 0.05, total_step, 0.6);

    let mut buffer = WindowBuffer::new(3);

    // pick action first
    let mut action = qtable.action(&mut rng, &mut exploration, tape.obs, tape.constraint);
    for count in 0..total_step {
        // log
        collector.record(0, "count", count);
        collector.record(1, "ep/reward_mean", tape.env().reward_mean());
        collector.record(1, "ep/length_mean", tape.env().length_mean());
        collector.record(1, "ep/time_mean", tape.env().time_mean());

        let (mut t, _) = tape.step(action);
        action = qtable.action(&mut rng, &mut exploration, tape.obs, tape.constraint);
        t.dict.insert_str("next_action", action);

        buffer.push(t);

        if buffer.is_ready() {
            let transitions = buffer.peek();
            n_step_sarsa_update(&state, &mut qtable, transitions);
            buffer.pop();
        }

        if count % 1000 == 0 {
            collector.dump_to(&mut stdout).expect("unexpected error");
        }

        if tape.done() {
            tape.reset();
            // update leftovers
            for transitions in buffer.drain() {
                n_step_sarsa_update(&state, &mut qtable, transitions);
            }
        }

        *exploration.epsilon_mut() = eps_sch.step() as f32;
    }

    let env = GymnasiumEnvironmentConfig::taxi().kwarg("render_mode", "human").init();
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