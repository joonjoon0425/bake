use bake::tabular::algorithm::NStepQLearning;
use bake::tabular::env::CliffWalking;
use bake::tabular::qtable::QTable;
use bake::tabular::explore::EpsGreedy;

use bake::logger::MovingAvgLogger;
use bake::scheduler::{LinearScheduler, Scheduler};
use bake_tabular::buffer::window::WindowBuffer;
use bake_tabular::env::Tape;
use bake_tabular::explore::Exploration;

pub fn main() {
    let state = NStepQLearning { n: 3, gamma: 0.99, alpha: 0.02 };
    let env = CliffWalking::new();
    let mut qtable = QTable::new(env.n_states(), env.n_actions());
    let mut exploration = EpsGreedy::new(12, 1.0);
    
    let total_steps = 100000;
    let mut eps_sch = LinearScheduler::new(1.0, 0.0, total_steps, 0.6);
    let mut tape = Tape::new(env);
    let mut window = WindowBuffer::new();

    let mut logger = MovingAvgLogger::new();
    logger.register("reward", 20);
    logger.register("steps", 20);

    for count in 0..=total_steps {
        let action = exploration.sample(&qtable, tape.obs, tape.constraint);
        let b_log_prob = exploration.prob(&qtable, tape.obs, action, tape.constraint).ln();
        let mut t = tape.step(action);
        t.insert("b_log_prob", b_log_prob);
        window.push(t);

        if window.len() >= state.n {
            let t = window.sample();
            NStepQLearning::update_is(&state, &mut qtable, t);
        }

        if tape.done() {
            logger.push_single("reward", tape.episode_reward);
            logger.push_single("steps", tape.steps as f32);
            tape.reset();
            // drain the window buffer
            for t in window.drain() {
                NStepQLearning::update_is(&state, &mut qtable, t);
            }
        }

        if count % 10000 == 0 {
            println!("count: {count}, reward: {}, steps: {}, eps: {}", logger.emit("reward"), logger.emit("steps"), exploration.eps());
        }

        *exploration.eps_mut() = eps_sch.step() as f32;
    }
}