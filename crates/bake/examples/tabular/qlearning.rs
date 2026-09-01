use bake::tabular::agent::*;
use bake::tabular::env::*;
use bake_tabular::policy::EpsGreedy;
use bake_tabular::types::Tape;

fn main() {
    let mut env = GridWorld::new();
    let mut agent = QLearningAgent::new(env.n_states(), env.n_actions(), 0.3, 0.99);
    let mut policy = EpsGreedy::new(2,1f32);
    let mut tape = Tape::new(&mut env);

    for i in 0..=100000 {
        let mut episode_reward = 0f32;
        let mut n_steps = 0usize;
        tape.reset(&mut env);
        loop {
            let action = agent.action(&mut policy, tape.obs, tape.mask);
            let t = tape.step(&mut env, action);
            agent.update(t.clone());
            
            episode_reward += t.reward;
            n_steps += 1;
            if t.terminated || t.truncated { break; }
        }
        *policy.eps_mut() *= 0.9996;
        if i % 10000 == 0 { println!("Episode {i}, Steps: {n_steps}, Reward: {}, Eps: {}", episode_reward, policy.eps()) }
    }
}