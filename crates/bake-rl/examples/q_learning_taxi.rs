use bake_rl::core::scheduler::{LinearScheduler, Scheduler};
use bake_rl::core::env::wrapper::RecordEpisodeStatistics;
use bake_rl::env::gym::GymnasiumEnvironmentConfig;
use bake_rl::tabular::prelude::*;

use rand::prelude::*;

pub fn main() {
    let seed = 129;
    let mut rng = SmallRng::seed_from_u64(seed);    
    let env = GymnasiumEnvironmentConfig::taxi().seed(seed).init();
    let env = RecordEpisodeStatistics::new(env);
    let state = QLearningState { gamma: 0.99, alpha: 0.4 };
    let mut q = QTable::new(env.obs_space().n(), env.action_space().n());
    let mut tape = Tape::new(env);

    let total_steps = 100000;
    let mut eg = EpsilonGreedy::new(1.0);
    let mut eps_sch = LinearScheduler::new(1.0, 0.05, total_steps, 0.5);

    for count in 0..total_steps {
        let action = q.action(&mut rng, &mut eg, tape.obs.clone(), tape.constraint.clone());
        let (t, (_, stat)) = tape.step(action);

        qlearning_update(&state, &mut q, t);

        if tape.done() && let Some(stat) = stat {
            if count % 50 == 0 {
                println!("count: {count}, reward: {}, step: {}", stat.r, stat.s);
            }
            tape.reset();
        }
        
        *eg.epsilon_mut() = eps_sch.step() as f32;
    }

    let env = GymnasiumEnvironmentConfig::taxi().kwarg("render_mode", "human").init();
    let mut tape = Tape::new(env);
    let mut g = Greedy;
    for _ in 0..5 {
        loop {
            let action = q.action(&mut rng, &mut g, tape.obs.clone(), tape.constraint);
            tape.step(action);
            if tape.done() {
                tape.reset();
                break;
            }
        }
    }
}