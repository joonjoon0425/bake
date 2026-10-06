use bake::rl::tabular::algorithm::NStepSarsa;
use bake::rl::tabular::qtable::QTable;
use bake::rl::tabular::explore::{EpsGreedy, Exploration};
use bake::rl::tabular::logger::MovingAvgLogger;
use bake::rl::tabular::scheduler::{LinearScheduler, Scheduler};
use bake::rl::tabular::buffer::window::WindowBuffer;
use bake::rl::env::{Environment, EnvironmentConfiguration, TabularEnvironmentConfiguration, collection::MaskedCliffWalking, tape::Tape};

pub fn main() {
    let state = NStepSarsa { n: 1, gamma: 0.99, alpha: 0.4 };
    let env = MaskedCliffWalking::new().init();
    let mut qtable = QTable::new(env.config().n_obs(), env.config().n_actions());
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
            logger.push_single("reward", tape.episode_reward, None);
            logger.push_single("steps", tape.steps as f32, None);
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