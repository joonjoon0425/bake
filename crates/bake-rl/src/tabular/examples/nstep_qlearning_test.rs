use bake::rl::tabular::algorithm::NStepQLearning;
use bake::rl::tabular::qtable::QTable;
use bake::rl::tabular::explore::EpsGreedy;
use bake::rl::tabular::buffer::window::WindowBuffer;
use bake::rl::tabular::explore::Exploration;

use bake::rl::env::{Environment, EnvironmentConfiguration, TabularEnvironmentConfiguration, collection::CliffWalking, tape::Tape};
use bake::rl::tabular::logger::MovingAvgLogger;
use bake::rl::tabular::scheduler::{LinearScheduler, Scheduler};

pub fn main() {
    let state = NStepQLearning { n: 3, gamma: 0.99, alpha: 0.02 };
    let env = CliffWalking::new().init();
    let mut qtable = QTable::new(env.config().n_obs(), env.config().n_actions());
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
            logger.push_single("reward", tape.episode_reward, None);
            logger.push_single("steps", tape.steps as f32, None);
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