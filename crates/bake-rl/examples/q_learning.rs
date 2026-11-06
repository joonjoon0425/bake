use bake_rl::core::scheduler::{LinearScheduler, Scheduler};
use bake_rl::env::gym::GymnasiumEnvironmentConfig;
use bake_rl::tabular::prelude::*;

use burn::prelude::*;
use bake_rl::tabular::qtable::QTable;

pub fn main() {
    let seed= 129;
    let device = Device::default();
    device.seed(seed);
    let env = GymnasiumEnvironmentConfig::taxi().seed(seed).init();
    let state = QLearningState { gamma: 0.99, alpha: 0.4 };
    let mut q = QTable::new(env.obs_space().n(), env.action_space().n());
    let mut tape = Tape::new(env);

    let total_steps = 100000;
    let mut eg = EpsGreedy::native(1.0, seed);
    let mut eps_sch = LinearScheduler::new(1.0, 0.05, total_steps, 0.75);

    for count in 0..total_steps {
        let action = q.action(&mut eg, tape.obs.clone(), tape.constraint.clone());
        let (t, _) = tape.step(action);
        let r = t.reward;
        qlearning_update(&state, &mut q, t);

        println!("count: {count}, environment: {}, reward: {}", 0, r);
        
        *eg.eps_mut() = eps_sch.step() as f32;
    }

    let env = GymnasiumEnvironmentConfig::taxi().kwarg("render_mode", "human").init();
    let mut tape = Tape::new(env);
    let mut g = Greedy;
    for _ in 0..5 {
        loop {
            let action = q.action(&mut g, tape.obs.clone(), tape.constraint);
            tape.step(action);
            if tape.done() {
                tape.reset();
                break;
            }
        }
    }
}