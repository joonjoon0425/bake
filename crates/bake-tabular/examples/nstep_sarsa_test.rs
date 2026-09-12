use bake_tabular::algorithm::NStepSarsa;
use bake_tabular::qtable::QTable;
use bake_tabular::explore::EpsGreedy;

use bake_common::logger::MovingAvgLogger;
use bake_common::scheduler::{LinearScheduler, Scheduler};
use bake_tabular::buffer::window::WindowBuffer;
use bake_tabular::env::{MaskedCliffWalking, Tape};
use bake_tabular::explore::Exploration;

pub fn main() {
    let state = NStepSarsa { n: 1, gamma: 0.99, alpha: 0.4 };
    let env = MaskedCliffWalking::new();
    let mut qtable = QTable::new(env.n_states(), env.n_actions());
    let mut exploration = EpsGreedy::new(12, 1.0);
    
    let total_steps = 100000;
    let mut eps_sch = LinearScheduler::new(1.0, 0.005, total_steps, 0.4);
    let mut tape = Tape::new(env);
    let mut window = WindowBuffer::new();

    let mut logger = MovingAvgLogger::new();
    logger.register("reward", 20);
    logger.register("steps", 20);
    
    let mut action = exploration.sample(&qtable, tape.obs, tape.constraint);
    for count in 0..=total_steps {
        let mut t = tape.step(action);
        action = exploration.sample(&qtable, tape.obs, tape.constraint);
        t.insert("next_action", action as f32);
        window.push(t);
        if window.len() >= state.n {
            let sample = window.sample();
            NStepSarsa::update(&state, &mut qtable, sample);
        }
        
        if tape.done() {
            logger.push_single("reward", tape.episode_reward);
            logger.push_single("steps", tape.steps as f32);
            tape.reset();
            window.clear();
            // drain the window buffer
            for t in window.drain() {
                NStepSarsa::update(&state, &mut qtable, t);
            }
            action = exploration.sample(&qtable, tape.obs, tape.constraint);
        }

        if count % 10000 == 0 {
            println!("count: {count}, reward: {}, steps: {}, eps: {}", logger.emit("reward"), logger.emit("steps"), exploration.eps());
        }

        *exploration.eps_mut() = eps_sch.step() as f32;
    }
}