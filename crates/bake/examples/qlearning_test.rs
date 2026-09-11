use bake::tabular::algorithm::QLearning;
use bake::tabular::qtable::QTable;
use bake::tabular::explore::EpsGreedy;

use bake::logger::MovingAvgLogger;
use bake::scheduler::{LinearScheduler, Scheduler};
use bake_tabular::env::{MaskedCliffWalking, Tape};
use bake_tabular::explore::Exploration;

pub fn main() {
    let state = QLearning { gamma: 0.99, alpha: 0.4 };
    let env = MaskedCliffWalking::new();
    let mut qtable = QTable::new(env.n_states(), env.n_actions());
    let mut exploration = EpsGreedy::new(12, 1.0);
    
    let total_steps = 100000;
    let mut eps_sch = LinearScheduler::new(1.0, 0.0, total_steps, 0.4);
    let mut tape = Tape::new(env);

    let mut logger = MovingAvgLogger::new();
    logger.register("reward", 20);
    logger.register("steps", 20);

    for count in 0..=total_steps {
        let action = exploration.sample(&qtable, tape.obs, tape.constraint);
        let t = tape.step(action);
        QLearning::update(&state, &mut qtable, t);

        if tape.done() {
            logger.push_single("reward", tape.episode_reward);
            logger.push_single("steps", tape.steps as f32);
            tape.reset();
        }

        if count % 10000 == 0 {
            println!("count: {count}, reward: {}, steps: {}, eps: {}", logger.emit("reward"), logger.emit("steps"), exploration.eps());
        }

        *exploration.eps_mut() = eps_sch.step() as f32;
    }
}