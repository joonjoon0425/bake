use bake::tabular::agent::*;
use bake::tabular::env::*;
use bake_tabular::policy::EpsGreedy;

fn main() {
    let mut env = GridWorld::new();
    let mut agent = QLearningAgent::new(env.n_states(), env.n_actions(), 0.3, 0.99);
    let mut policy = EpsGreedy::new(2,1f32);

    for i in 0..100000 {
        let mut obs = env.reset();
        let mut mask = env.action_mask();

        let mut reward = 0f32;
        let mut n_steps = 0usize;
        loop {
            let action = agent.action(&mut policy, obs, mask);
            let step = env.step(action);
            let next_mask = env.action_mask();
            agent.update(obs, action, next_mask, step);

            obs = step.obs;
            mask = next_mask;
            
            reward += step.reward;
            n_steps += 1;
            if step.done || step.truncated { break; }
        }
        *policy.eps_mut() *= 0.999;
        if i % 1000 == 0 { println!("Episode {i}, Steps: {n_steps}, Reward: {reward}, Eps: {}", policy.eps()) }
        
    }
    
}